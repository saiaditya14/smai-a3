"""
Shared pipeline module for the Indic Speech Command Recognizer (T11.7).

All four entry points (app.py, benchmarking.py, ablation_audio.py,
ablation_siamese.py) import from here. No Streamlit dependency lives in
this file - app.py wraps these functions with @st.cache_resource.

Two recognition paths are exposed:
  * Transcription path: forced-language Whisper-tiny + fuzzy command match.
  * Siamese path: contrastive projection head over Whisper encoder
    embeddings, classified by nearest prototype.
Both are gated by the same acoustic-embedding "Hey Bharat" wakeword.
"""

import os
import glob
import time
from difflib import SequenceMatcher

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

targetRate = 16000

BASE_DIR = os.path.dirname(__file__) or "."
MODELS_DIR = os.path.join(BASE_DIR, "models")

WAKE_ENROLL_DIR    = os.path.join(BASE_DIR, "help", "yes")
WAKE_SIM_THRESHOLD = 0.910  # Adjusted for pitch augmentation
WAKE_WINDOW_SEC    = 1.5
WAKE_HOP_SEC       = 0.5

SIAMESE_HEAD_PATH  = os.path.join(MODELS_DIR, "siamese_head.pt")
SIAMESE_PROTOS_PATH = os.path.join(MODELS_DIR, "siamese_prototypes.npz")
SIAMESE_SIM_THRESHOLD = 0.6  # below this, the siamese path returns None


# ---------------------------------------------------------------------------
# Command vocabulary (Hindi, Tamil, Telugu)
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

fleursLangMap = {
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
}


# ---------------------------------------------------------------------------
# Model loading & audio ingestion
# ---------------------------------------------------------------------------

def loadModel():
    """Load Whisper-tiny processor + model. Caller is responsible for caching."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.eval()
    return proc, model


def ingestAudio(rawBytes):
    """Bytes -> mono float32 numpy at 16 kHz, peak-normalised."""
    import io
    buf = io.BytesIO(rawBytes)
    audioNp, _ = librosa.load(buf, sr=targetRate, mono=True)
    audioNp = audioNp.astype(np.float32)
    peak = np.max(np.abs(audioNp))
    if peak > 0:
        audioNp = audioNp / peak
    return audioNp


def loadWavFile(filePath):
    """Path -> mono float32 numpy at 16 kHz, peak-normalised."""
    audio, _ = librosa.load(filePath, sr=targetRate, mono=True)
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


# ---------------------------------------------------------------------------
# Encoder embeddings & wakeword detection
# ---------------------------------------------------------------------------

def _extractEncoderEmbedding(audioArray, proc, model):
    """
    Whisper-encoder-only embedding: mean-pool over active speech frames.
    Output is L2-normalised (384-dim for whisper-tiny).
    """
    trimmed, _ = librosa.effects.trim(audioArray, top_db=25)
    nAudioFrames = len(trimmed) // 160
    inputs = proc(trimmed, sampling_rate=targetRate, return_tensors="pt")
    encoder = model.get_encoder()
    with torch.no_grad():
        encOut = encoder(inputs.input_features)
    hidden = encOut.last_hidden_state.squeeze(0)
    activeFrames = min(max(nAudioFrames // 2, 10), hidden.shape[0])
    emb = hidden[:activeFrames].mean(dim=0).numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb


def buildWakeAnchor(proc, model, enrollDir=None):
    """Average encoder embeddings across reference 'Hey Bharat' recordings with pitch augmentation."""
    enrollDir = enrollDir or WAKE_ENROLL_DIR
    wavFiles = sorted(glob.glob(os.path.join(enrollDir, "*.wav")))
    if not wavFiles:
        return None
    embeddings = []
    for f in wavFiles:
        audio = loadWavFile(f)
        embeddings.append(_extractEncoderEmbedding(audio, proc, model))
        # Augment with higher (female/child) and lower pitch variants
        for steps in [3, 6, -3]: 
            shifted = librosa.effects.pitch_shift(y=audio, sr=targetRate, n_steps=steps)
            embeddings.append(_extractEncoderEmbedding(shifted, proc, model))
            
    anchor = np.mean(embeddings, axis=0)
    anchor = anchor / (np.linalg.norm(anchor) + 1e-9)
    return anchor


def detectWake(audioArray, proc, model, anchor):
    """
    Sliding-window cosine similarity of encoder embeddings vs anchor.
    Returns (detected: bool, bestSimilarity: float).
    """
    if anchor is None:
        return False, 0.0
    trimmed, _ = librosa.effects.trim(audioArray, top_db=25)
    windowSamples = int(WAKE_WINDOW_SEC * targetRate)
    hopSamples    = int(WAKE_HOP_SEC * targetRate)

    if len(trimmed) <= windowSamples:
        emb = _extractEncoderEmbedding(trimmed, proc, model)
        sim = float(np.dot(anchor, emb))
        return sim >= WAKE_SIM_THRESHOLD, sim

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
# Transcription + fuzzy command matching (Path A)
# ---------------------------------------------------------------------------

def transcribe(audioArray, proc, model, lang=None, prompt=None):
    """Whisper-tiny inference with optional language forcing + prompt biasing."""
    inputs = proc(audioArray, sampling_rate=targetRate, return_tensors="pt")
    inputFeatures = inputs.input_features

    genKwargs = {"task": "transcribe"}
    if lang is not None:
        genKwargs["language"] = lang
    if prompt is not None:
        genKwargs["prompt_ids"] = proc.get_prompt_ids(prompt, return_tensors="pt")

    with torch.no_grad():
        predicted = model.generate(inputFeatures, **genKwargs)
    text = proc.batch_decode(predicted, skip_special_tokens=True)[0]
    return text.strip()


def fuzzyScore(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def matchCommand(text, threshold=0.7):
    """Exact substring + sliding-window fuzzy match across all language vocabs."""
    cleaned = text.lower().strip()
    bestMatch = None
    highestScore = 0.0

    for lang, cmds in commandsMap.items():
        for native, action in cmds.items():
            if native.lower() in cleaned:
                return lang, native, action

    words = cleaned.split()
    for lang, cmds in commandsMap.items():
        for native, action in cmds.items():
            nativeLower = native.lower()
            nativeWords = nativeLower.split()
            nLen = len(nativeWords)

            for i in range(max(1, len(words) - nLen + 1)):
                window = " ".join(words[i : i + nLen])
                score = fuzzyScore(nativeLower, window)
                if score > highestScore:
                    highestScore = score
                    bestMatch = (lang, native, action)

            fullScore = fuzzyScore(nativeLower, cleaned)
            if fullScore > highestScore:
                highestScore = fullScore
                bestMatch = (lang, native, action)

    if highestScore >= threshold:
        return bestMatch
    return None


def transcriptionPipeline(audioArray, proc, model):
    """
    Path A: try each FLEURS language code in order, return first matching command.
    Returns (result, perLangTimings) where result is (lang, native, action) or None.
    """
    timings = {}
    for langName, langCode in fleursLangMap.items():
        t0 = time.perf_counter()
        text = transcribe(audioArray, proc, model, lang=langCode)
        timings[langName] = time.perf_counter() - t0
        result = matchCommand(text)
        if result is not None:
            return result, timings, text
    return None, timings, ""


# ---------------------------------------------------------------------------
# Siamese projection head (Path B)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Lightweight MLP trained with contrastive loss over encoder embeddings."""
    def __init__(self, input_dim=384, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)


def loadSiameseAssets(headPath=None, protosPath=None):
    """
    Load trained projection head + per-command prototypes from disk.
    Returns (head, prototypes_dict, idx_to_action) or (None, None, None) if absent.
    """
    headPath = headPath or SIAMESE_HEAD_PATH
    protosPath = protosPath or SIAMESE_PROTOS_PATH
    if not (os.path.exists(headPath) and os.path.exists(protosPath)):
        return None, None, None

    head = ProjectionHead()
    head.load_state_dict(torch.load(headPath, map_location="cpu"))
    head.eval()

    data = np.load(protosPath, allow_pickle=True)
    prototypes = {int(k): torch.tensor(data["protos"][i], dtype=torch.float32)
                  for i, k in enumerate(data["idx"])}
    idx_to_action = {int(k): str(data["actions"][i])
                     for i, k in enumerate(data["idx"])}
    return head, prototypes, idx_to_action


def classifyBySiamese(audioArray, proc, model, head, prototypes, idx_to_action,
                      threshold=SIAMESE_SIM_THRESHOLD):
    """
    Path B: encode audio -> project -> nearest prototype.
    Returns ((lang, native, action), bestSim) or (None, bestSim).
    Language is reported as 'auto' since the head is language-agnostic.
    """
    emb = _extractEncoderEmbedding(audioArray, proc, model)
    with torch.no_grad():
        proj = head(torch.tensor(emb, dtype=torch.float32).unsqueeze(0)).squeeze(0)

    bestIdx = None
    bestSim = -1.0
    for idx, proto in prototypes.items():
        sim = float(torch.dot(proj, proto))
        if sim > bestSim:
            bestSim = sim
            bestIdx = idx

    if bestSim < threshold or bestIdx is None:
        return None, bestSim
    action = idx_to_action[bestIdx]  # e.g. "turn on the light"
    if action == "__reject__":
        return None, bestSim
    return ("auto", action, action.title()), bestSim


# ---------------------------------------------------------------------------
# Combined pipelines (wakeword gate + recognition path)
# ---------------------------------------------------------------------------

def runMultiLangPipeline(audioArray, proc, model, anchor, skip_wake=False):
    """
    Original pipeline used for benchmarking. Wakeword gate -> transcription path.
    skip_wake=True isolates command recognition (skips wakeword check).
    Returns (lang, native, action) or None.
    """
    if not skip_wake:
        wakeDetected, _ = detectWake(audioArray, proc, model, anchor)
        if not wakeDetected:
            return None
    result, _, _ = transcriptionPipeline(audioArray, proc, model)
    return result


def runSiamesePipeline(audioArray, proc, model, anchor, head, prototypes,
                       idx_to_action, skip_wake=False):
    """Wakeword gate -> siamese classifier."""
    if not skip_wake:
        wakeDetected, _ = detectWake(audioArray, proc, model, anchor)
        if not wakeDetected:
            return None
    result, _ = classifyBySiamese(audioArray, proc, model, head, prototypes,
                                  idx_to_action)
    return result
