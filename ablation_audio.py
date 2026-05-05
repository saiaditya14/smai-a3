import os
import glob
import librosa
import torch
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, sosfilt
import sys
# make sure we can import benchmarking functions
sys.path.append(os.path.dirname(__file__))
from benchmarking import loadModel, transcribe, matchCommand, commandsMap, fleursLangMap

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def get_audio_files():
    files = []
    for root, _, fnames in os.walk(DATA_DIR):
        for f in fnames:
            if f.endswith(".wav"):
                cmd_dir = os.path.basename(root)
                if cmd_dir == "none": continue
                files.append((cmd_dir, os.path.join(root, f)))
    return files

def apply_bandpass(audio, sr=16000, lowcut=300, highcut=3400, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    filtered = sosfilt(sos, audio)
    return filtered.astype(np.float32)

def evaluate(preprocessing_strategy, proc, model):
    files = get_audio_files()
    correct = 0
    total = len(files)
    
    for expected_cmd, fpath in tqdm(files, desc=preprocessing_strategy):
        # Baseline: sr=16000 directly
        if preprocessing_strategy == "baseline":
            audio, _ = librosa.load(fpath, sr=16000, mono=True)
            
        elif preprocessing_strategy == "telephony":
            # Resample to 8kHz then back to 16kHz
            audio_8k, _ = librosa.load(fpath, sr=8000, mono=True)
            audio = librosa.resample(audio_8k, orig_sr=8000, target_sr=16000)
            
        elif preprocessing_strategy == "bandpass":
            audio_raw, _ = librosa.load(fpath, sr=16000, mono=True)
            audio = apply_bandpass(audio_raw, sr=16000)
            
        # evaluate using whisper over all configured languages (forcing)
        best_match = None
        for langName, langCode in fleursLangMap.items():
            text = transcribe(audio, proc, model, lang=langCode)
            result = matchCommand(text)
            if result is not None:
                _, _, action = result
                # convert action string to cmd_dir string
                action_k = action.lower().replace(" ", "_")
                if action_k == expected_cmd:
                    best_match = True
                    break
        
        if best_match:
            correct += 1

    print(f"{preprocessing_strategy} Accuracy: {correct}/{total} = {correct/total:.3f}")

if __name__ == "__main__":
    proc, model = loadModel()
    print("Running Audio Preprocessing Ablation")
    evaluate("baseline", proc, model)
    evaluate("telephony", proc, model)
    evaluate("bandpass", proc, model)

