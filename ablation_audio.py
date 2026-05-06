"""
Audio preprocessing ablation: baseline vs telephony (8kHz round-trip)
vs strict bandpass (300-3400 Hz). Whisper resamples internally to 16 kHz
either way - what we're testing is whether removing high-frequency content
before the feature extractor helps for short commands.
"""

import os
import librosa
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, sosfilt

import core
from core import loadModel, transcribe, matchCommand, fleursLangMap, targetRate

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def get_audio_files():
    files = []
    for root, _, fnames in os.walk(DATA_DIR):
        for f in fnames:
            if f.endswith(".wav"):
                cmd_dir = os.path.basename(root)
                if cmd_dir == "none":
                    continue
                files.append((cmd_dir.replace("_", " ").lower(), os.path.join(root, f)))
    return files


def apply_bandpass(audio, sr=16000, lowcut=300, highcut=3400, order=5):
    nyq = 0.5 * sr
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='bandpass', output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def evaluate(strategy, proc, model):
    files = get_audio_files()
    correct = 0
    total = len(files)

    for expected_cmd, fpath in tqdm(files, desc=strategy):
        if strategy == "baseline":
            audio, _ = librosa.load(fpath, sr=targetRate, mono=True)
        elif strategy == "telephony":
            audio_8k, _ = librosa.load(fpath, sr=8000, mono=True)
            audio = librosa.resample(audio_8k, orig_sr=8000, target_sr=targetRate)
        elif strategy == "bandpass":
            audio_raw, _ = librosa.load(fpath, sr=targetRate, mono=True)
            audio = apply_bandpass(audio_raw, sr=targetRate)

        for langName, langCode in fleursLangMap.items():
            text = transcribe(audio, proc, model, lang=langCode)
            result = matchCommand(text)
            if result is not None and result[2].lower() == expected_cmd:
                correct += 1
                break

    print(f"{strategy} Accuracy: {correct}/{total} = {correct/total:.3f}")
    return correct, total


if __name__ == "__main__":
    proc, model = loadModel()
    print("Running Audio Preprocessing Ablation")
    evaluate("baseline", proc, model)
    evaluate("telephony", proc, model)
    evaluate("bandpass", proc, model)
