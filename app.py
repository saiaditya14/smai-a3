import streamlit as st
from streamlit.components.v1 import html as renderHtml
import hashlib
import time
import os

import core
from core import (
    commandsMap, fleursLangMap, ingestAudio, detectWake,
    transcribe, matchCommand, loadSiameseAssets, classifyBySiamese,
)

# ---------------------------------------------------------------------------
# SVG Kitten States
# ---------------------------------------------------------------------------

try:
    with open(os.path.join("assets", "sleeping_kitten.svg"), "r", encoding="utf-8") as f:
        sleepingSvg = f.read()
except FileNotFoundError:
    sleepingSvg = "<svg>Sleeping SVG Not Found</svg>"

try:
    with open(os.path.join("assets", "awake__kitten.svg"), "r", encoding="utf-8") as f:
        awakeSvg = f.read()
except FileNotFoundError:
    awakeSvg = "<svg>Awake SVG Not Found</svg>"

# ---------------------------------------------------------------------------
# Cached resources (Whisper, anchor, siamese head)
# ---------------------------------------------------------------------------

@st.cache_resource
def loadModelCached():
    return core.loadModel()

@st.cache_resource
def buildWakeAnchorCached(_proc, _model):
    return core.buildWakeAnchor(_proc, _model)

@st.cache_resource
def loadSiameseAssetsCached():
    return loadSiameseAssets()

# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Indic Speech Command Recognizer",
    page_icon="🐱",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .block-container { max-width: 680px; padding-top: 1.5rem; }
    .kittenWrap { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem 0 0.5rem 0; }
    .kittenWrap svg { filter: drop-shadow(0 4px 18px rgba(0,0,0,0.08)); transition: transform 0.3s ease; }
    .kittenWrap:hover svg { transform: scale(1.03); }
    .statusPill { display: inline-block; padding: 0.35rem 1.2rem; border-radius: 999px; font-weight: 600; font-size: 0.9rem; letter-spacing: 0.04em; margin-top: 0.6rem; }
    .statusSleeping { background: linear-gradient(135deg, #e0e0e0 0%, #c4c4c4 100%); color: #555; }
    .statusAwake { background: linear-gradient(135deg, #FFB347 0%, #FF6F61 100%); color: #fff; }
    .resultBox { background: #FAFAFA; border: 1.5px solid #E8E8E8; border-radius: 16px; padding: 1.4rem 1.8rem; margin-top: 1rem; box-shadow: 0 2px 12px rgba(0,0,0,0.04); }
    .resultBox h4 { margin: 0 0 0.5rem 0; font-weight: 700; color: #333; }
    .resultBox p { margin: 0.25rem 0; font-weight: 400; color: #555; font-size: 0.95rem; }
    .resultBox .actionTag { display: inline-block; margin-top: 0.6rem; padding: 0.3rem 0.9rem; border-radius: 8px; background: linear-gradient(135deg, #FFB347 0%, #FF6F61 100%); color: #fff; font-weight: 600; font-size: 0.92rem; }
    .resultBox .latencyBadge { display: inline-block; margin-left: 0.5rem; padding: 0.2rem 0.7rem; border-radius: 8px; background: #eef; color: #336; font-weight: 500; font-size: 0.82rem; }
    .subtitle { text-align: center; color: #999; font-size: 0.92rem; margin-top: -0.3rem; margin-bottom: 0.2rem; }
    .appTitle { text-align: center; font-size: 1.75rem; font-weight: 700; color: #333; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Session state ----
for key, default in [("isAwake", False), ("lastTranscript", ""),
                     ("matchedCmd", None), ("lastAudioHash", ""),
                     ("lastLatencyMs", None), ("lastModeUsed", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---- Sidebar: mode toggle ----
with st.sidebar:
    st.markdown("### ⚙️ Recognition mode")
    mode = st.radio(
        "Pipeline",
        ["Transcription (fuzzy match)", "Siamese (projection head)", "Both — vote"],
        index=0,
        help="Transcription runs Whisper 3× (one per language). Siamese runs the encoder once.",
    )

    st.markdown("### 📋 Supported Commands")
    for lang, cmds in commandsMap.items():
        with st.expander(f"{lang.capitalize()} ({fleursLangMap[lang]})"):
            for native, action in cmds.items():
                st.markdown(f"- **{native}** → _{action}_")

# ---- Header ----
st.markdown('<p class="appTitle">🐱 Indic Speech Command Recognizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Say <b>"Hey Bharat"</b> to wake the kitten, then speak a command in '
    'Hindi, Tamil, or Telugu.</p>',
    unsafe_allow_html=True,
)

# ---- Kitten ----
currSvg = awakeSvg if st.session_state["isAwake"] else sleepingSvg
pill = ('<span class="statusPill statusAwake">🟢 Awake — Listening</span>'
        if st.session_state["isAwake"]
        else '<span class="statusPill statusSleeping">😴 Sleeping</span>')

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

audioValue = st.audio_input("🎤 Speak to the kitten")

if audioValue is not None:
    rawBytes = audioValue.read()
    st.download_button("💾 Download recording", data=rawBytes,
                       file_name="recording.wav", mime="audio/wav")

    audioHash = hashlib.md5(rawBytes).hexdigest()
    if audioHash != st.session_state["lastAudioHash"]:
        st.session_state["lastAudioHash"] = audioHash

        with st.spinner("Loading Whisper-tiny model (CPU) …"):
            proc, model = loadModelCached()
        with st.spinner("Ingesting & resampling audio …"):
            audioArray = ingestAudio(rawBytes)

        wakeAnchor = buildWakeAnchorCached(proc, model)

        # Phase 1: Wakeword
        t0 = time.perf_counter()
        with st.spinner("Checking wakeword …"):
            wakeDetected, wakeSim = detectWake(audioArray, proc, model, wakeAnchor)
        wakeMs = (time.perf_counter() - t0) * 1000

        st.session_state["lastTranscript"] = f"[Wakeword similarity: {wakeSim:.4f}]"

        if wakeDetected:
            st.session_state["isAwake"] = True

            # Phase 2: Recognition path(s)
            transcriptionResult = None
            siameseResult = None
            transcriptionMs = None
            siameseMs = None
            transcriptText = ""

            if mode in ("Transcription (fuzzy match)", "Both — vote"):
                t0 = time.perf_counter()
                with st.spinner("Transcribing across languages …"):
                    for langName, langCode in fleursLangMap.items():
                        langText = transcribe(audioArray, proc, model, lang=langCode)
                        st.write(f"🔍 **Raw {langName}:** `{langText}`")
                        result = matchCommand(langText)
                        if result is not None and transcriptionResult is None:
                            transcriptionResult = result
                            transcriptText = langText
                transcriptionMs = (time.perf_counter() - t0) * 1000

            if mode in ("Siamese (projection head)", "Both — vote"):
                head, prototypes, idx_to_action = loadSiameseAssetsCached()
                if head is None:
                    st.warning("Siamese head not found on disk. Run `python ablation_siamese.py` first.")
                else:
                    t0 = time.perf_counter()
                    with st.spinner("Classifying via siamese head …"):
                        sResult, sSim = classifyBySiamese(
                            audioArray, proc, model, head, prototypes, idx_to_action
                        )
                    siameseMs = (time.perf_counter() - t0) * 1000
                    siameseResult = sResult
                    st.write(f"🔍 **Siamese best similarity:** `{sSim:.4f}`")

            # Pick final answer based on mode
            if mode == "Transcription (fuzzy match)":
                bestMatch = transcriptionResult
                latency = (wakeMs or 0) + (transcriptionMs or 0)
                modeUsed = "Transcription"
            elif mode == "Siamese (projection head)":
                bestMatch = siameseResult
                latency = (wakeMs or 0) + (siameseMs or 0)
                modeUsed = "Siamese"
            else:  # Both — vote
                if (transcriptionResult is not None and siameseResult is not None
                        and transcriptionResult[2] == siameseResult[2]):
                    bestMatch = transcriptionResult  # they agree, prefer transcription's lang label
                    modeUsed = "Both (agree)"
                elif transcriptionResult is not None:
                    bestMatch = transcriptionResult
                    modeUsed = "Both (transcription only)"
                elif siameseResult is not None:
                    bestMatch = siameseResult
                    modeUsed = "Both (siamese only)"
                else:
                    bestMatch = None
                    modeUsed = "Both (no match)"
                latency = (wakeMs or 0) + (transcriptionMs or 0) + (siameseMs or 0)

            if transcriptText:
                st.session_state["lastTranscript"] = transcriptText
            elif mode == "Siamese (projection head)":
                st.session_state["lastTranscript"] = "No transcript (Siamese mode)"
            st.session_state["matchedCmd"] = bestMatch
            st.session_state["lastLatencyMs"] = latency
            st.session_state["lastModeUsed"] = modeUsed
        else:
            st.session_state["isAwake"] = False
            st.session_state["matchedCmd"] = None
            st.session_state["lastLatencyMs"] = wakeMs
            st.session_state["lastModeUsed"] = "Wakeword failed"

        st.rerun()


def get_action_ui(action_str):
    action = action_str.lower()
    animation_css = ""
    emoji_html = ""
    if "turn on the light" in action:
        animation_css = ".glow-bulb { font-size: 4rem; text-shadow: 0 0 20px #ffea00, 0 0 30px #ffea00; animation: pulse 2s infinite; } @keyframes pulse { 0% { text-shadow: 0 0 20px #ffea00; } 50% { text-shadow: 0 0 40px #ffea00, 0 0 60px #ffea00; } 100% { text-shadow: 0 0 20px #ffea00; } }"
        emoji_html = '<div style="text-align:center;"><span class="glow-bulb">💡</span><br><b style="color:#ffea00">LIGHT ON</b></div>'
    elif "turn off the light" in action:
        animation_css = ".dim-bulb { font-size: 4rem; filter: grayscale(100%) opacity(50%); }"
        emoji_html = '<div style="text-align:center;"><span class="dim-bulb">💡</span><br><b style="color:#666">LIGHT OFF</b></div>'
    elif "turn on the fan" in action:
        animation_css = ".spin-fan { font-size: 4rem; display: inline-block; animation: spin 0.5s linear infinite; } @keyframes spin { 100% { transform: rotate(360deg); } }"
        emoji_html = '<div style="text-align:center;"><span class="spin-fan">🌀</span><br><b style="color:#00e5ff">FAN ON</b></div>'
    elif "turn off the fan" in action:
        animation_css = ".still-fan { font-size: 4rem; display: inline-block; filter: grayscale(100%); }"
        emoji_html = '<div style="text-align:center;"><span class="still-fan">🌀</span><br><b style="color:#666">FAN OFF</b></div>'
    elif "play music" in action:
        animation_css = ".bounce-music { font-size: 4rem; display: inline-block; animation: bounce 1s infinite alternate; } @keyframes bounce { 0% { transform: translateY(0); } 100% { transform: translateY(-10px); } }"
        emoji_html = '<div style="text-align:center;"><span class="bounce-music">🎵</span> 🔊<br><b style="color:#e91e63">PLAYING MUSIC</b></div>'
    elif "stop music" in action:
        animation_css = ".stop-music { font-size: 4rem; filter: grayscale(100%); }"
        emoji_html = '<div style="text-align:center;"><span class="stop-music">🎵</span> 🔇<br><b style="color:#666">MUSIC STOPPED</b></div>'
    elif "increase volume" in action:
        animation_css = ".pump-vol { font-size: 4rem; display: inline-block; animation: pump 0.5s ease-out; } @keyframes pump { 50% { transform: scale(1.3); } 100% { transform: scale(1); } }"
        emoji_html = '<div style="text-align:center;"><span class="pump-vol">🔊</span> 📈<br><b style="color:#4caf50">VOLUME INCREASED</b></div>'
    elif "decrease volume" in action:
        animation_css = ".drop-vol { font-size: 4rem; display: inline-block; animation: drop 0.5s ease-out; } @keyframes drop { 50% { transform: scale(0.7); } 100% { transform: scale(1); } }"
        emoji_html = '<div style="text-align:center;"><span class="drop-vol">🔉</span> 📉<br><b style="color:#ff9800">VOLUME DECREASED</b></div>'
    elif "tell the temperature" in action:
        animation_css = ".temp-shake { font-size: 4rem; display: inline-block; animation: shake 2s infinite cubic-bezier(.36,.07,.19,.97) both; transform: translate3d(0, 0, 0); } @keyframes shake { 10%, 90% { transform: translate3d(-1px, 0, 0); } 20%, 80% { transform: translate3d(2px, 0, 0); } 30%, 50%, 70% { transform: translate3d(-4px, 0, 0); } 40%, 60% { transform: translate3d(4px, 0, 0); } }"
        emoji_html = '<div style="text-align:center;"><span class="temp-shake">🌡️</span><br><b style="color:#ff5722">TEMPERATURE</b></div>'
    elif "tell the time" in action:
        animation_css = ".tick-tock { font-size: 4rem; display: inline-block; animation: tick 1s steps(2, end) infinite; } @keyframes tick { 0% { transform: rotate(0deg); } 100% { transform: rotate(15deg); } }"
        emoji_html = '<div style="text-align:center;"><span class="tick-tock">🕒</span><br><b style="color:#9c27b0">TIME</b></div>'
    return f"<style>{animation_css}</style>\n<div style='margin: 20px 0; padding: 20px; background: rgba(0,0,0,0.1); border-radius: 15px;'>{emoji_html}</div>"


# ---- Result card ----
if st.session_state["lastTranscript"] or st.session_state["matchedCmd"] is not None:
    transcript = st.session_state["lastTranscript"]
    cmd = st.session_state["matchedCmd"]
    latency = st.session_state["lastLatencyMs"]
    modeUsed = st.session_state["lastModeUsed"]

    latencyBadge = (f'<span class="latencyBadge">⏱ {latency:.0f} ms · {modeUsed}</span>'
                    if latency is not None else "")

    if cmd is not None:
        lang, native, action = cmd
        st.markdown(
            f"""
            <div class="resultBox">
                <h4>✦ Command Recognised {latencyBadge}</h4>
                <p><b>Transcript:</b> {transcript}</p>
                <p><b>Language:</b> {lang.capitalize()}</p>
                <p><b>Command:</b> {native}</p>
                <span class="actionTag">⚡ {action}</span>
                {get_action_ui(action)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state["isAwake"]:
        st.markdown(
            f"""
            <div class="resultBox">
                <h4>✦ Wakeword Detected {latencyBadge}</h4>
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
                <h4>Transcript {latencyBadge}</h4>
                <p>{transcript}</p>
                <p style="color:#999;">Wakeword not detected. Say "Hey Bharat" first.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if st.session_state["isAwake"]:
    st.markdown("")
    if st.button("🔄 Put kitten back to sleep", use_container_width=True):
        st.session_state["isAwake"] = False
        st.session_state["lastTranscript"] = ""
        st.session_state["matchedCmd"] = None
        st.rerun()
