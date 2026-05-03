import argparse
import os
import csv
import librosa
import torch
import numpy as np
from datasets import load_dataset
from jiwer import wer
from difflib import SequenceMatcher

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

targetRate = 16000

def loadModel():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.eval()
    return proc, model

def transcribe(audioArray, proc, model, lang=None, prompt=None):
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

def fuzzyScore(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def matchCommand(text):
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

    if highestScore >= 0.7:
        return bestMatch

    return None

def ensureNewline(filePath):
    if os.path.exists(filePath) and os.path.getsize(filePath) > 0:
        with open(filePath, "rb+") as fp:
            fp.seek(-1, os.SEEK_END)
            if fp.read(1) != b'\n':
                fp.write(b'\n')

def runMultiLangPipeline(audioData, proc, model):
    for langName, langCode in fleursLangMap.items():
        langText = transcribe(audioData, proc, model, lang=langCode)
        result = matchCommand(langText)
        if result is not None:
            return result
    return None

def evaluateFleursWer():
    proc, model = loadModel()
    outCsv = "benchmark_results_fleurs.csv"
    
    ensureNewline(outCsv)
    with open(outCsv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if os.stat(outCsv).st_size == 0:
            writer.writerow(["language", "reference", "prediction", "wer"])
        
        for langName, langCode in fleursLangMap.items():
            print(f"Evaluating FLEURS for {langName} ({langCode})...")
            datasetName = f"{langCode}_in"
            dataset = load_dataset("google/fleurs", name=datasetName, split="test", streaming=True, trust_remote_code=True)
            
            totalWer = 0.0
            count = 0
            
            for item in dataset:
                if count >= 500:
                    break
                    
                audioData = item["audio"]["array"]
                origSr = item["audio"]["sampling_rate"]
                if origSr != targetRate:
                    audioData = librosa.resample(y=audioData, orig_sr=origSr, target_sr=targetRate)
                
                audioData = audioData.astype(np.float32)
                peak = np.max(np.abs(audioData))
                if peak > 0:
                    audioData = audioData / peak
                    
                reference = item["transcription"].strip()
                prediction = transcribe(audioData, proc, model, lang=langCode)
                
                currentWer = wer(reference, prediction)
                totalWer += currentWer
                count += 1
                
                if count % 10 == 0:
                    print(f"Heartbeat: Processed {count} samples for {langCode}...")
                
                writer.writerow([langName, reference, prediction, currentWer])
                
            avgWer = totalWer / count if count > 0 else 0
            print(f"Completed {langName} | Evaluated: {count} | Avg WER: {avgWer:.4f}")

def evaluateLocalCommands(dataDir):
    proc, model = loadModel()
    outCsv = "benchmark_results_local.csv"
    
    ensureNewline(outCsv)
    with open(outCsv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if os.stat(outCsv).st_size == 0:
            writer.writerow(["fileName", "expectedAction", "predictedAction", "success"])
            
        total = 0
        successes = 0
        
        for subDir in os.listdir(dataDir):
            subDirPath = os.path.join(dataDir, subDir)
            if not os.path.isdir(subDirPath):
                continue
                
            expectedAction = subDir.replace("_", " ").strip().lower()
            
            for fileName in os.listdir(subDirPath):
                if not fileName.lower().endswith(".wav"):
                    continue
                    
                filePath = os.path.join(subDirPath, fileName)
                audioData, _ = librosa.load(filePath, sr=targetRate, mono=True)
                audioData = audioData.astype(np.float32)
                peak = np.max(np.abs(audioData))
                if peak > 0:
                    audioData = audioData / peak
                    
                matchResult = runMultiLangPipeline(audioData, proc, model)
                
                predictedAction = matchResult[2].lower() if matchResult else "none"
                isSuccess = (predictedAction == expectedAction)
                
                if isSuccess:
                    successes += 1
                total += 1
                
                if total % 10 == 0:
                    print(f"Heartbeat: Processed {total} local samples...")
                
                writer.writerow([fileName, expectedAction, predictedAction, isSuccess])
                
        accuracy = (successes / total) * 100 if total > 0 else 0
        print(f"\nLocal Commands Evaluation Summary")
        print(f"Total Samples: {total}")
        print(f"Success Rate: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Headless Benchmarking")
    parser.add_argument("--mode", choices=["fleurs", "local"], required=True, help="Evaluation mode to run")
    parser.add_argument("--dataDir", type=str, default="data", help="Directory containing local evaluation subfolders")
    args = parser.parse_args()

    if args.mode == "fleurs":
        evaluateFleursWer()
    elif args.mode == "local":
        if not os.path.exists(args.dataDir):
            print(f"Directory not found: {args.dataDir}")
            exit(1)
        evaluateLocalCommands(args.dataDir)
