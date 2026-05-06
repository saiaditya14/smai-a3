"""
Test harness v3: compares full-clip embedding vs sliding-window approach.

The sliding window scans overlapping windows of wakeword-length (~1.5s)
across the audio and takes the MAX similarity. This handles clips where
"Hey Bharat" is followed by a command — the window over just the wakeword
portion will match the anchor even if the full clip doesn't.
"""

import os, glob, sys
import numpy as np
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

TARGET_SR = 16000
YES_DIR = os.path.join("help", "yes")
NO_DIR  = os.path.join("help", "no")

print("Loading whisper-tiny ...")
proc  = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.eval()
encoder = model.get_encoder()


def load_audio(path):
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    audio = audio.astype(np.float32)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def get_embedding(audio, pool="mean"):
    """Get encoder embedding from an audio array (already trimmed)."""
    n_frames = len(audio) // 160
    inputs = proc(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        enc_out = encoder(inputs.input_features)
    hidden = enc_out.last_hidden_state.squeeze(0)
    active = min(max(n_frames // 2, 10), hidden.shape[0])
    emb = hidden[:active].mean(dim=0).numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb


def cosine_sim(a, b):
    return float(np.dot(a, b))


def sliding_window_max_sim(audio, anchor, window_secs=1.5, hop_secs=0.5):
    """
    Slide a window across the audio, extract embedding for each window,
    return the MAX cosine similarity against the anchor.
    
    This means even if the audio is "Hey Bharat + batti jalao", the window
    that lands on just "Hey Bharat" will match.
    """
    window_samples = int(window_secs * TARGET_SR)
    hop_samples = int(hop_secs * TARGET_SR)
    
    # If audio is shorter than window, just use the whole thing
    if len(audio) <= window_samples:
        emb = get_embedding(audio)
        return cosine_sim(anchor, emb)
    
    best_sim = -1.0
    start = 0
    while start + window_samples <= len(audio):
        chunk = audio[start : start + window_samples]
        emb = get_embedding(chunk)
        sim = cosine_sim(anchor, emb)
        if sim > best_sim:
            best_sim = sim
        start += hop_samples
    
    # Also try the tail (last window_samples of audio)
    tail = audio[-window_samples:]
    emb = get_embedding(tail)
    sim = cosine_sim(anchor, emb)
    if sim > best_sim:
        best_sim = sim
    
    return best_sim


# -- Load all files ----------------------------------------------------------

yes_files = sorted(glob.glob(os.path.join(YES_DIR, "*.wav")))
no_files  = sorted(glob.glob(os.path.join(NO_DIR, "*.wav")))

print(f"\nFound {len(yes_files)} YES, {len(no_files)} NO samples\n")

yes_audios = []
for f in yes_files:
    a = load_audio(f)
    yes_audios.append(a)
    print(f"  [YES] {os.path.basename(f):30s}  dur={len(a)/TARGET_SR:.2f}s")

no_audios = []
for f in no_files:
    a = load_audio(f)
    no_audios.append(a)
    print(f"  [NO]  {os.path.basename(f):30s}  dur={len(a)/TARGET_SR:.2f}s")

# -- Build anchor from YES embeddings ----------------------------------------

yes_embs = [get_embedding(a) for a in yes_audios]
anchor = np.mean(yes_embs, axis=0)
anchor = anchor / (np.linalg.norm(anchor) + 1e-9)

# =============================================================================
# METHOD 1: Full-clip embedding (current approach)
# =============================================================================

print(f"\n{'='*65}")
print(f"  METHOD 1: Full-clip embedding")
print(f"{'='*65}")

pos_full = []
for f, a in zip(yes_files, yes_audios):
    emb = get_embedding(a)
    sim = cosine_sim(anchor, emb)
    pos_full.append(sim)
    marker = "PASS" if sim >= 0.950 else "FAIL"
    print(f"  [YES] {os.path.basename(f):30s}  sim={sim:.6f}  {marker}")

neg_full = []
for f, a in zip(no_files, no_audios):
    emb = get_embedding(a)
    sim = cosine_sim(anchor, emb)
    neg_full.append(sim)
    marker = "REJECT" if sim < 0.950 else "FALSE POS"
    print(f"  [NO]  {os.path.basename(f):30s}  sim={sim:.6f}  {marker}")

print(f"  Min pos={min(pos_full):.6f}  Max neg={max(neg_full):.6f}  Gap={min(pos_full)-max(neg_full):.6f}")

# =============================================================================
# METHOD 2: Sliding window (1.5s window, 0.5s hop)
# =============================================================================

print(f"\n{'='*65}")
print(f"  METHOD 2: Sliding window (1.5s window, 0.5s hop)")
print(f"{'='*65}")

pos_sw = []
for f, a in zip(yes_files, yes_audios):
    sim = sliding_window_max_sim(a, anchor, window_secs=1.5, hop_secs=0.5)
    pos_sw.append(sim)
    marker = "PASS" if sim >= 0.950 else "FAIL"
    print(f"  [YES] {os.path.basename(f):30s}  max_sim={sim:.6f}  {marker}")

neg_sw = []
for f, a in zip(no_files, no_audios):
    sim = sliding_window_max_sim(a, anchor, window_secs=1.5, hop_secs=0.5)
    neg_sw.append(sim)
    marker = "REJECT" if sim < 0.950 else "FALSE POS"
    print(f"  [NO]  {os.path.basename(f):30s}  max_sim={sim:.6f}  {marker}")

print(f"  Min pos={min(pos_sw):.6f}  Max neg={max(neg_sw):.6f}  Gap={min(pos_sw)-max(neg_sw):.6f}")

# =============================================================================
# METHOD 3: Sliding window (2.0s window, 0.5s hop)
# =============================================================================

print(f"\n{'='*65}")
print(f"  METHOD 3: Sliding window (2.0s window, 0.5s hop)")
print(f"{'='*65}")

pos_sw2 = []
for f, a in zip(yes_files, yes_audios):
    sim = sliding_window_max_sim(a, anchor, window_secs=2.0, hop_secs=0.5)
    pos_sw2.append(sim)
    marker = "PASS" if sim >= 0.950 else "FAIL"
    print(f"  [YES] {os.path.basename(f):30s}  max_sim={sim:.6f}  {marker}")

neg_sw2 = []
for f, a in zip(no_files, no_audios):
    sim = sliding_window_max_sim(a, anchor, window_secs=2.0, hop_secs=0.5)
    neg_sw2.append(sim)
    marker = "REJECT" if sim < 0.950 else "FALSE POS"
    print(f"  [NO]  {os.path.basename(f):30s}  max_sim={sim:.6f}  {marker}")

print(f"  Min pos={min(pos_sw2):.6f}  Max neg={max(neg_sw2):.6f}  Gap={min(pos_sw2)-max(neg_sw2):.6f}")

print("\nDone.")
