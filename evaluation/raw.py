import os
import csv
import librosa
import torch
import numpy as np
import argparse
#running raw model on our dataset
def loadModel():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.eval()
    return proc, model

def transcribe(audioArray, proc, model):
    inputs = proc(audioArray, sampling_rate=16000, return_tensors="pt")
    inputFeatures = inputs.input_features
    with torch.no_grad():
        predicted = model.generate(inputFeatures, task="transcribe")
    text = proc.batch_decode(predicted, skip_special_tokens=True)[0]
    return text.strip()

def evaluateRawLocal(dataDir):
    proc, model = loadModel()
    outCsv = "benchmark_results_raw.csv"
    with open(outCsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fileName", "expectedAction", "rawTranscription"])
        if not os.path.exists(dataDir):
            return
        for subDir in os.listdir(dataDir):
            subDirPath = os.path.join(dataDir, subDir)
            if not os.path.isdir(subDirPath):
                continue
            expectedAction = subDir.replace("_", " ").strip().lower()
            for fileName in os.listdir(subDirPath):
                if not fileName.lower().endswith(".wav"):
                    continue
                filePath = os.path.join(subDirPath, fileName)
                audioData, _ = librosa.load(filePath, sr=16000, mono=True)
                audioData = audioData.astype(np.float32)
                peak = np.max(np.abs(audioData))
                if peak > 0:
                    audioData = audioData / peak
                rawResult = transcribe(audioData, proc, model)
                writer.writerow([fileName, expectedAction, rawResult])
                f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", type=str, default="data")
    args = parser.parse_args()
    evaluateRawLocal(args.dataDir)
