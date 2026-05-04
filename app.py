import streamlit as st
from streamlit.components.v1 import html as renderHtml
import hashlib
import torch
import librosa
import numpy as np
import io
import os
import glob
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# SVG Kitten States (raw SVG strings)
# Cute chubby kawaii style — layered back-to-front so face renders on top.
# ---------------------------------------------------------------------------

try:
    with open("sleeping_kitten.svg", "r", encoding="utf-8") as f:
        sleepingSvg = f.read()
except FileNotFoundError:
    sleepingSvg = "<svg>Sleeping SVG Not Found</svg>"

try:
    with open("awake__kitten.svg", "r", encoding="utf-8") as f:
        awakeSvg = f.read()
except FileNotFoundError:
    awakeSvg = "<svg>Awake SVG Not Found</svg>"

# ---------------------------------------------------------------------------
# Predefined multi-language commands (Hindi, Tamil, Telugu)
# ---------------------------------------------------------------------------

commandsMap = {
    "hindi": {
        "बत्ती जलाओ": "Turn on the light",
        "batti jalao": "Turn on the light",
        "बत्ती बुझाओ": "Turn off the light",
        "batti bujhao": "Turn off the light",
        "पंखा चालू करो": "Turn on the fan",
        "pankha chalu karo": "Turn on the fan",
        "pankha chaalu karo": "Turn on the fan",
        "पंखा बंद करो": "Turn off the fan",
        "pankha band karo": "Turn off the fan",
        "गाना बजाओ": "Play music",
        "gana bajao": "Play music",
        "गाना बंद करो": "Stop music",
        "gana band karo": "Stop music",
        "आवाज़ बढ़ाओ": "Increase volume",
        "awaz badhao": "Increase volume",
        "aawaz badhao": "Increase volume",
        "आवाज़ कम करो": "Decrease volume",
        "awaz kam karo": "Decrease volume",
        "aawaz kam karo": "Decrease volume",
        "तापमान बताओ": "Tell the temperature",
        "tapman batao": "Tell the temperature",
        "taapmaan batao": "Tell the temperature",
        "समय बताओ": "Tell the time",
        "samay batao": "Tell the time",
    },
    "tamil": {
        "விளக்கை போடு": "Turn on the light",
        "vilakkai podu": "Turn on the light",
        "விளக்கை அணை": "Turn off the light",
        "vilakkai anai": "Turn off the light",
        "பாட்டு போடு": "Play music",
        "paattu podu": "Play music",
        "pattu podu": "Play music",
        "பாட்டு நிறுத்து": "Stop music",
        "paattu niruthu": "Stop music",
        "pattu niruthu": "Stop music",
        "விசிறி போடு": "Turn on the fan",
        "visiri podu": "Turn on the fan",
        "விசிறி நிறுத்து": "Turn off the fan",
        "visiri niruthu": "Turn off the fan",
        "ஒலி அதிகமாக்கு": "Increase volume",
        "oli athigamakku": "Increase volume",
        "ஒலி குறை": "Decrease volume",
        "oli kurai": "Decrease volume",
        "வெப்பநிலை சொல்": "Tell the temperature",
        "veppanilai sol": "Tell the temperature",
        "நேரம் சொல்": "Tell the time",
        "neram sol": "Tell the time",
    },
    "telugu": {
        "లైట్ వేయి": "Turn on the light",
        "light veyi": "Turn on the light",
        "లైట్ ఆపు": "Turn off the light",
        "light aapu": "Turn off the light",
        "ఫ్యాన్ వేయి": "Turn on the fan",
        "fan veyi": "Turn on the fan",
        "ఫ్యాన్ ఆపు": "Turn off the fan",
        "fan aapu": "Turn off the fan",
        "పాట వేయి": "Play music",
        "paata veyi": "Play music",
        "pata veyi": "Play music",
        "పాట ఆపు": "Stop music",
        "paata aapu": "Stop music",
        "pata aapu": "Stop music",
        "శబ్దం పెంచు": "Increase volume",
        "shabdam penchu": "Increase volume",
        "శబ్దం తగ్గించు": "Decrease volume",
        "shabdam tagginchu": "Decrease volume",
        "ఉష్ణోగ్రత చెప్పు": "Tell the temperature",
        "ushnogratha cheppu": "Tell the temperature",
        "సమయం చెప్పు": "Tell the time",
        "samayam cheppu": "Tell the time",
    },
}

# Whisper language tokens for FLEURS configs
fleursLangMap = {
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
}

# ---------------------------------------------------------------------------
# Model loading (cached so it loads only once)
# ---------------------------------------------------------------------------

@st.cache_resource
def loadModel():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.eval()
    return proc, model

# ---------------------------------------------------------------------------
# Audio ingestion & resampling
# ---------------------------------------------------------------------------

targetRate = 16000

def ingestAudio(rawBytes):
    """Load audio from raw bytes via librosa, resample to 16 kHz, return float32 numpy array."""
    buf = io.BytesIO(rawBytes)
    # librosa.load returns mono float32 at the requested sr; no ffmpeg needed for wav
    audioNp, sr = librosa.load(buf, sr=targetRate, mono=True)
    audioNp = audioNp.astype(np.float32)
    peak = np.max(np.abs(audioNp))
    if peak > 0:
        audioNp = audioNp / peak
    return audioNp

# ---------------------------------------------------------------------------
# Transcription pipeline
# ---------------------------------------------------------------------------

def transcribe(audioArray, proc, model, lang=None, prompt=None):
    """
    Run Whisper-tiny inference on a numpy float32 waveform at 16 kHz.
    If lang is provided it forces the decoder language token.
    If prompt is provided it biases the decoder toward expected vocabulary.
    """
    inputs = proc(
        audioArray,
        sampling_rate=targetRate,
        return_tensors="pt",
    )
    inputFeatures = inputs.input_features

    genKwargs = {"task": "transcribe"}
    if lang is not None:
        genKwargs["language"] = lang

    if prompt is not None:
        promptIds = proc.get_prompt_ids(prompt, return_tensors="pt")
        genKwargs["prompt_ids"] = promptIds

    with torch.no_grad():
        predicted = model.generate(inputFeatures, **genKwargs)

    text = proc.batch_decode(predicted, skip_special_tokens=True)[0]
    return text.strip()

# ---------------------------------------------------------------------------
# Fuzzy string matching (used by command matching below)
# ---------------------------------------------------------------------------

def fuzzyScore(a, b):
    """Return SequenceMatcher ratio between two lowercase strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ---------------------------------------------------------------------------
# Wakeword detection — Acoustic Embedding Cosine Similarity
# ---------------------------------------------------------------------------
# Instead of decoding audio to text (which hallucinates for non-English proper
# nouns), we use Whisper's ENCODER as an acoustic feature extractor:
#   1. Build an "anchor" embedding by averaging encoder outputs over reference
#      recordings of "Hey Bharat" (stored in help/yes/*.wav).
#   2. For incoming audio, extract encoder embedding of the speech portion
#      and compute cosine similarity against the anchor.
#   3. If similarity exceeds a threshold, the wakeword is detected.
# This completely bypasses the autoregressive decoder for the wakeword step.
# ---------------------------------------------------------------------------

WAKE_ENROLL_DIR = os.path.join(os.path.dirname(__file__) or ".", "help", "yes")
WAKE_SIM_THRESHOLD = 0.922  # statistically optimal across both domains


def _extractEncoderEmbedding(audioArray, proc, model):
    """
    Run audio through Whisper's encoder only (no autoregressive decoding).
    Mean-pool across active speech frames to get a fixed-size embedding.
    
    Whisper-tiny encoder output: (1, T, 384) where T=1500 (for 30s window).
    We only pool over frames corresponding to actual audio content, not
    the zero-padded silence that dominates short clips.
    """
    # Trim silence so we only encode active speech
    trimmed, _ = librosa.effects.trim(audioArray, top_db=25)
    nAudioFrames = len(trimmed) // 160  # mel spectrogram frames (~100 frames/sec)

    inputs = proc(trimmed, sampling_rate=targetRate, return_tensors="pt")
    encoder = model.get_encoder()
    with torch.no_grad():
        encOut = encoder(inputs.input_features)

    hidden = encOut.last_hidden_state.squeeze(0)  # (1500, 384)

    # Whisper's conv layers downsample by ~2x, so active frames in encoder space
    activeFrames = min(max(nAudioFrames // 2, 10), hidden.shape[0])
    activeHidden = hidden[:activeFrames]  # (activeFrames, 384)

    emb = activeHidden.mean(dim=0).numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-9)  # L2-normalise
    return emb


@st.cache_resource
def _buildWakeAnchor(_proc, _model):
    """
    Build the wakeword anchor embedding by averaging encoder embeddings
    from all reference recordings in help/yes/.
    Cached so it only runs once per session.
    """
    wavFiles = sorted(glob.glob(os.path.join(WAKE_ENROLL_DIR, "*.wav")))
    if not wavFiles:
        st.warning(f"No wakeword reference recordings found in {WAKE_ENROLL_DIR}/")
        return None

    embeddings = []
    for f in wavFiles:
        audio, _ = librosa.load(f, sr=targetRate, mono=True)
        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        emb = _extractEncoderEmbedding(audio, _proc, _model)
        embeddings.append(emb)

    anchor = np.mean(embeddings, axis=0)
    anchor = anchor / (np.linalg.norm(anchor) + 1e-9)
    return anchor


WAKE_WINDOW_SEC = 1.5   # sliding window duration (matches typical wakeword length)
WAKE_HOP_SEC    = 0.5   # hop between windows

def detectWake(audioArray, proc, model, anchor):
    """
    Acoustic wakeword detection via sliding-window encoder-embedding similarity.

    Scans overlapping 1.5 s windows across the (trimmed) audio and returns the
    MAX cosine similarity against the anchor.  This way a clip containing
    "Hey Bharat <command>" still triggers — the window that lands on just the
    wakeword portion will match even if the full-clip embedding is diluted.

    Returns (detected: bool, bestSimilarity: float).
    """
    if anchor is None:
        return False, 0.0

    # Trim silence first so we only scan speech
    trimmed, _ = librosa.effects.trim(audioArray, top_db=25)

    windowSamples = int(WAKE_WINDOW_SEC * targetRate)
    hopSamples    = int(WAKE_HOP_SEC * targetRate)

    # Short clip — just use the whole thing (no sliding needed)
    if len(trimmed) <= windowSamples:
        emb = _extractEncoderEmbedding(trimmed, proc, model)
        sim = float(np.dot(anchor, emb))
        return sim >= WAKE_SIM_THRESHOLD, sim

    # Slide windows and keep best similarity
    bestSim = -1.0
    start = 0
    while start + windowSamples <= len(trimmed):
        chunk = trimmed[start : start + windowSamples]
        emb = _extractEncoderEmbedding(chunk, proc, model)
        sim = float(np.dot(anchor, emb))
        if sim > bestSim:
            bestSim = sim
        start += hopSamples

    return bestSim >= WAKE_SIM_THRESHOLD, bestSim

# ---------------------------------------------------------------------------
# Command matching
# ---------------------------------------------------------------------------

def matchCommand(text):
    """
    Check every language's command list for a match inside the transcribed text.
    First tries an exact substring match, then falls back to sliding-window fuzzy matching
    to handle Whisper's spelling variations (e.g. "bhatty jalao" vs "batti jalao").
    Returns (language, nativeCmd, englishAction) or None.
    """
    cleaned = text.lower().strip()
    bestMatch = None
    highestScore = 0.0

    # Fast path: exact match
    for lang, cmds in commandsMap.items():
        for native, action in cmds.items():
            if native.lower() in cleaned:
                return lang, native, action

    # Fuzzy path: sliding window score
    words = cleaned.split()
    for lang, cmds in commandsMap.items():
        for native, action in cmds.items():
            nativeLower = native.lower()
            nativeWords = nativeLower.split()
            nLen = len(nativeWords)

            # Slide window of matching length
            for i in range(max(1, len(words) - nLen + 1)):
                window = " ".join(words[i : i + nLen])
                score = fuzzyScore(nativeLower, window)
                if score > highestScore:
                    highestScore = score
                    bestMatch = (lang, native, action)

            # Also check against the entire text if the transcript is very short
            fullScore = fuzzyScore(nativeLower, cleaned)
            if fullScore > highestScore:
                highestScore = fullScore
                bestMatch = (lang, native, action)

    if highestScore >= 0.7:
        return bestMatch

    return None

# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Indic Speech Command Recognizer",
    page_icon="🐱",
    layout="centered",
)

# ---- Custom CSS ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .block-container {
        max-width: 680px;
        padding-top: 1.5rem;
    }

    /* ---------- centred kitten area ---------- */
    .kittenWrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem 0 0.5rem 0;
    }

    .kittenWrap svg {
        filter: drop-shadow(0 4px 18px rgba(0,0,0,0.08));
        transition: transform 0.3s ease;
    }

    .kittenWrap:hover svg {
        transform: scale(1.03);
    }

    /* ---------- status pill ---------- */
    .statusPill {
        display: inline-block;
        padding: 0.35rem 1.2rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.04em;
        margin-top: 0.6rem;
    }

    .statusSleeping {
        background: linear-gradient(135deg, #e0e0e0 0%, #c4c4c4 100%);
        color: #555;
    }

    .statusAwake {
        background: linear-gradient(135deg, #FFB347 0%, #FF6F61 100%);
        color: #fff;
    }

    /* ---------- results card ---------- */
    .resultBox {
        background: #FAFAFA;
        border: 1.5px solid #E8E8E8;
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        margin-top: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }

    .resultBox h4 {
        margin: 0 0 0.5rem 0;
        font-weight: 700;
        color: #333;
    }

    .resultBox p {
        margin: 0.25rem 0;
        font-weight: 400;
        color: #555;
        font-size: 0.95rem;
    }

    .resultBox .actionTag {
        display: inline-block;
        margin-top: 0.6rem;
        padding: 0.3rem 0.9rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #FFB347 0%, #FF6F61 100%);
        color: #fff;
        font-weight: 600;
        font-size: 0.92rem;
    }

    /* ---------- subtitle ---------- */
    .subtitle {
        text-align: center;
        color: #999;
        font-size: 0.92rem;
        margin-top: -0.3rem;
        margin-bottom: 0.2rem;
    }

    /* ---------- header ---------- */
    .appTitle {
        text-align: center;
        font-size: 1.75rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Session state initialisation ----
if "isAwake" not in st.session_state:
    st.session_state["isAwake"] = False
if "lastTranscript" not in st.session_state:
    st.session_state["lastTranscript"] = ""
if "matchedCmd" not in st.session_state:
    st.session_state["matchedCmd"] = None
if "lastAudioHash" not in st.session_state:
    st.session_state["lastAudioHash"] = ""

# ---- Header (centred) ----
st.markdown('<p class="appTitle">🐱 Indic Speech Command Recognizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">'
    'Say <b>"Hey Bharat"</b> to wake the kitten, then speak a command in '
    'Hindi, Tamil, or Telugu.</p>',
    unsafe_allow_html=True,
)

# ---- Kitten display (centred via st.components.v1.html) ----
currSvg = awakeSvg if st.session_state["isAwake"] else sleepingSvg

if st.session_state["isAwake"]:
    pill = '<span class="statusPill statusAwake">🟢 Awake — Listening</span>'
else:
    pill = '<span class="statusPill statusSleeping">😴 Sleeping</span>'

kittenHtml = f'''
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:1rem 0;">
  {currSvg}
  <div style="margin-top:0.6rem;">{pill}</div>
</div>
<style>
svg {{ max-height: 280px; width: auto; }}
.statusPill {{display:inline-block;padding:0.35rem 1.2rem;border-radius:999px;font-weight:600;font-size:0.9rem;letter-spacing:0.04em;font-family:'Outfit',sans-serif;}}
.statusSleeping {{background:linear-gradient(135deg,#e0e0e0 0%,#c4c4c4 100%);color:#555;}}
.statusAwake {{background:linear-gradient(135deg,#FFB347 0%,#FF6F61 100%);color:#fff;}}
</style>
'''
renderHtml(kittenHtml, height=380)

st.divider()

# ---- Mic input (replaces file uploader) ----
audioValue = st.audio_input("🎤 Speak to the kitten")

if audioValue is not None:
    rawBytes = audioValue.read()

    st.download_button(
        label="💾 Download recording",
        data=rawBytes,
        file_name="recording.wav",
        mime="audio/wav"
    )

    # Guard: hash the audio so we never reprocess the same clip on rerun
    audioHash = hashlib.md5(rawBytes).hexdigest()
    if audioHash != st.session_state["lastAudioHash"]:
        st.session_state["lastAudioHash"] = audioHash

        with st.spinner("Loading Whisper-tiny model (CPU) …"):
            proc, model = loadModel()

        with st.spinner("Ingesting & resampling audio …"):
            audioArray = ingestAudio(rawBytes)

        # ------------------------------------------------------------------
        # Build / retrieve the wakeword anchor embedding (cached)
        # ------------------------------------------------------------------
        wakeAnchor = _buildWakeAnchor(proc, model)

        # ------------------------------------------------------------------
        # Phase 1: Wakeword detection via acoustic embedding similarity
        # Uses Whisper's encoder only — NO autoregressive text decoding.
        # ------------------------------------------------------------------
        with st.spinner("Checking wakeword (acoustic embedding) …"):
            wakeDetected, wakeSim = detectWake(audioArray, proc, model, wakeAnchor)

        st.session_state["lastTranscript"] = f"[Wakeword similarity: {wakeSim:.4f}]"

        if wakeDetected:
            st.session_state["isAwake"] = True

            # --------------------------------------------------------------
            # Phase 2: Multi-language command transcription
            # Bias the decoder with command vocabulary for each language.
            # --------------------------------------------------------------
            bestMatch = None

            for langName, langCode in fleursLangMap.items():
                with st.spinner(f"Transcribing in {langName} ({langCode}) …"):
                    langText = transcribe(audioArray, proc, model, lang=langCode)
                
                # Debug output to see what the model actually transcribed
                st.write(f"\U0001f50d **Raw {langName} Transcript:** `{langText}`")
                print(f"Raw {langName} Transcript: {langText}") # Also print to terminal

                result = matchCommand(langText)
                if result is not None:
                    bestMatch = result
                    st.session_state["lastTranscript"] = langText
                    break

            st.session_state["matchedCmd"] = bestMatch
        else:
            st.session_state["isAwake"] = False
            st.session_state["matchedCmd"] = None

        st.rerun()

# ---- Results box (clean card below kitten) ----
if st.session_state["lastTranscript"] or st.session_state["matchedCmd"] is not None:
    transcript = st.session_state["lastTranscript"]
    cmd = st.session_state["matchedCmd"]

    if cmd is not None:
        lang, native, action = cmd
        st.markdown(
            f"""
            <div class="resultBox">
                <h4>✦ Command Recognised</h4>
                <p><b>Transcript:</b> {transcript}</p>
                <p><b>Language:</b> {lang.capitalize()}</p>
                <p><b>Command:</b> {native}</p>
                <span class="actionTag">⚡ {action}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state["isAwake"]:
        st.markdown(
            f"""
            <div class="resultBox">
                <h4>✦ Wakeword Detected</h4>
                <p><b>Transcript:</b> {transcript}</p>
                <p style="color:#999;">No matching command found in the audio.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="resultBox">
                <h4>Transcript</h4>
                <p>{transcript}</p>
                <p style="color:#999;">Wakeword not detected. Say "Hey Bharat" first.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---- Reset button ----
if st.session_state["isAwake"]:
    st.markdown("")
    if st.button("🔄 Put kitten back to sleep", use_container_width=True):
        st.session_state["isAwake"] = False
        st.session_state["lastTranscript"] = ""
        st.session_state["matchedCmd"] = None
        st.rerun()

# ---- Sidebar: supported commands reference ----
with st.sidebar:
    st.markdown("### 📋 Supported Commands")
    for lang, cmds in commandsMap.items():
        with st.expander(f"{lang.capitalize()} ({fleursLangMap[lang]})"):
            for native, action in cmds.items():
                st.markdown(f"- **{native}** → _{action}_")
