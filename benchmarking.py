"""
Headless benchmark runner. Two modes:
  --mode fleurs : Word Error Rate on Google FLEURS (hi/ta/te) test split, 500/lang.
  --mode local  : Intent accuracy on self-recorded data/ + false-positive rate
                  on data/none/ and help/no/ negatives.

Both transcription and siamese paths are evaluated in --mode local for the
head-to-head comparison the report needs.
"""

import argparse
import csv
import glob
import os
import time

import librosa
import numpy as np

import core
from core import (
    fleursLangMap, targetRate, loadModel, buildWakeAnchor, detectWake,
    transcribe, runMultiLangPipeline, transcriptionPipeline,
    classifyBySiamese, loadSiameseAssets, loadWavFile,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__) or ".", "results")


def resultPath(fileName):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, fileName)


def ensureNewline(filePath):
    if os.path.exists(filePath) and os.path.getsize(filePath) > 0:
        with open(filePath, "rb+") as fp:
            fp.seek(-1, os.SEEK_END)
            if fp.read(1) != b'\n':
                fp.write(b'\n')


# ---------------------------------------------------------------------------
# FLEURS WER
# ---------------------------------------------------------------------------

def evaluateFleursWer():
    from datasets import load_dataset
    from jiwer import wer
    proc, model = loadModel()
    outCsv = resultPath("benchmark_results_fleurs.csv")
    ensureNewline(outCsv)

    with open(outCsv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if os.stat(outCsv).st_size == 0:
            writer.writerow(["language", "reference", "prediction", "wer"])

        for langName, langCode in fleursLangMap.items():
            print(f"Evaluating FLEURS for {langName} ({langCode})...")
            datasetName = f"{langCode}_in"
            dataset = load_dataset(
                "google/fleurs", name=datasetName, split="test",
                streaming=True, trust_remote_code=True,
            )

            totalWer = 0.0
            count = 0
            for item in dataset:
                if count >= 500:
                    break
                audioData = item["audio"]["array"]
                origSr = item["audio"]["sampling_rate"]
                if origSr != targetRate:
                    audioData = librosa.resample(
                        y=audioData, orig_sr=origSr, target_sr=targetRate
                    )
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
                    print(f"Heartbeat: {count} samples for {langCode}...")
                writer.writerow([langName, reference, prediction, currentWer])

            avgWer = totalWer / count if count > 0 else 0
            print(f"Completed {langName} | Evaluated: {count} | Avg WER: {avgWer:.4f}")


# ---------------------------------------------------------------------------
# Local commands: accuracy + per-path latency + FPR on negatives
# ---------------------------------------------------------------------------

def _iterCommandFiles(dataDir):
    """Yield (expectedAction, filePath) for every *.wav under dataDir/<command>/,
    excluding the 'none' folder (which is handled separately)."""
    for subDir in sorted(os.listdir(dataDir)):
        if subDir == "none":
            continue
        subPath = os.path.join(dataDir, subDir)
        if not os.path.isdir(subPath):
            continue
        expected = subDir.replace("_", " ").strip().lower()
        for fileName in sorted(os.listdir(subPath)):
            if fileName.lower().endswith(".wav"):
                yield expected, os.path.join(subPath, fileName)


def _iterNegatives(dataDir):
    """Yield (poolName, filePath) for every negative clip across data/none/ and help/no/."""
    nonePath = os.path.join(dataDir, "none")
    if os.path.isdir(nonePath):
        for fileName in sorted(os.listdir(nonePath)):
            if fileName.lower().endswith(".wav"):
                yield "data_none", os.path.join(nonePath, fileName)
    helpNoPath = os.path.join(os.path.dirname(__file__) or ".", "help", "no")
    if os.path.isdir(helpNoPath):
        for fileName in sorted(os.listdir(helpNoPath)):
            if fileName.lower().endswith(".wav"):
                yield "help_no", os.path.join(helpNoPath, fileName)
    helpNoTestPath = os.path.join(os.path.dirname(__file__) or ".", "help", "no_test")
    if os.path.isdir(helpNoTestPath):
        for fileName in sorted(os.listdir(helpNoTestPath)):
            if fileName.lower().endswith(".wav"):
                yield "help_no_test", os.path.join(helpNoTestPath, fileName)


def evaluateLocalCommands(dataDir, skip_wake=False, run_siamese=True):
    proc, model = loadModel()
    anchor = buildWakeAnchor(proc, model)

    siamese_head, siamese_protos, siamese_idx2action = (None, None, None)
    if run_siamese:
        siamese_head, siamese_protos, siamese_idx2action = loadSiameseAssets()
        if siamese_head is None:
            print("[!] siamese assets not found on disk - skipping siamese path. "
                  "Run `python ablation_siamese.py` to generate them.")
            run_siamese = False

    outCsv = resultPath("benchmark_results_local.csv")
    fields = [
        "fileName", "expectedAction",
        "transcriptionPredicted", "transcriptionSuccess", "transcriptionMs",
        "siamesePredicted", "siameseSuccess", "siameseMs",
        "wakeDetected", "wakeSim", "wakeMs",
    ]
    with open(outCsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)

        total = 0
        tx_correct = 0
        sm_correct = 0
        for expected, fpath in _iterCommandFiles(dataDir):
            audio = loadWavFile(fpath)

            t0 = time.perf_counter()
            wakeOk, wakeSim = detectWake(audio, proc, model, anchor)
            wakeMs = (time.perf_counter() - t0) * 1000

            tx_pred, tx_ms = "none", 0.0
            sm_pred, sm_ms = "none", 0.0

            wake_pass = wakeOk or skip_wake
            if wake_pass:
                t0 = time.perf_counter()
                result, _, _ = transcriptionPipeline(audio, proc, model)
                tx_ms = (time.perf_counter() - t0) * 1000
                tx_pred = result[2].lower() if result else "none"

                if run_siamese:
                    t0 = time.perf_counter()
                    sResult, _ = classifyBySiamese(
                        audio, proc, model, siamese_head, siamese_protos,
                        siamese_idx2action,
                    )
                    sm_ms = (time.perf_counter() - t0) * 1000
                    sm_pred = sResult[2].lower() if sResult else "none"

            tx_ok = tx_pred == expected
            sm_ok = sm_pred == expected
            tx_correct += int(tx_ok)
            sm_correct += int(sm_ok)
            total += 1

            writer.writerow([
                os.path.basename(fpath), expected,
                tx_pred, tx_ok, f"{tx_ms:.1f}",
                sm_pred, sm_ok, f"{sm_ms:.1f}",
                wakeOk, f"{wakeSim:.4f}", f"{wakeMs:.1f}",
            ])
            f.flush()
            if total % 10 == 0:
                print(f"Heartbeat: {total} samples processed...")

    print()
    print(f"Local command evaluation (skip_wake={skip_wake})")
    print(f"  Total clips: {total}")
    print(f"  Transcription accuracy: {tx_correct}/{total} = {tx_correct/total*100:.2f}%")
    if run_siamese:
        print(f"  Siamese accuracy:       {sm_correct}/{total} = {sm_correct/total*100:.2f}%")
    print(f"  Wrote per-clip results to {outCsv}")


def evaluateNegatives(dataDir, run_siamese=True):
    """False-positive rate on data/none/ and help/no/."""
    proc, model = loadModel()
    anchor = buildWakeAnchor(proc, model)

    siamese_head, siamese_protos, siamese_idx2action = (None, None, None)
    if run_siamese:
        siamese_head, siamese_protos, siamese_idx2action = loadSiameseAssets()
        if siamese_head is None:
            run_siamese = False

    outCsv = resultPath("benchmark_results_negatives.csv")
    fields = ["pool", "fileName", "wakeDetected", "wakeSim",
              "transcriptionFiredCommand", "siameseFiredCommand"]
    with open(outCsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)

        from collections import defaultdict
        wake_fp = defaultdict(int)
        tx_fp = defaultdict(int)
        sm_fp = defaultdict(int)
        totals = defaultdict(int)

        for pool, fpath in _iterNegatives(dataDir):
            audio = loadWavFile(fpath)
            wakeOk, wakeSim = detectWake(audio, proc, model, anchor)
            tx_fired = False
            sm_fired = False
            if wakeOk:
                wake_fp[pool] += 1
                result, _, _ = transcriptionPipeline(audio, proc, model)
                tx_fired = result is not None
                if tx_fired:
                    tx_fp[pool] += 1
                if run_siamese:
                    sResult, _ = classifyBySiamese(
                        audio, proc, model, siamese_head, siamese_protos,
                        siamese_idx2action,
                    )
                    sm_fired = sResult is not None
                    if sm_fired:
                        sm_fp[pool] += 1
            totals[pool] += 1
            writer.writerow([pool, os.path.basename(fpath),
                             wakeOk, f"{wakeSim:.4f}", tx_fired, sm_fired])

    print()
    print("Negative-pool false-positive rates")
    for pool, n in totals.items():
        if n == 0:
            continue
        print(f"  [{pool}] n={n} | wake_fp={wake_fp[pool]}/{n}={wake_fp[pool]/n*100:.1f}%"
              f" | tx_cmd_fp={tx_fp[pool]}/{n}={tx_fp[pool]/n*100:.1f}%"
              f" | siamese_cmd_fp={sm_fp[pool]}/{n}={sm_fp[pool]/n*100:.1f}%")
    print(f"  Wrote per-clip results to {outCsv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Headless Benchmarking")
    parser.add_argument("--mode", choices=["fleurs", "local", "negatives"], required=True)
    parser.add_argument("--dataDir", type=str, default="data")
    parser.add_argument("--skipWake", action="store_true",
                        help="Skip wakeword gating (isolates command recognition).")
    parser.add_argument("--noSiamese", action="store_true",
                        help="Skip the siamese path (use if weights not yet trained).")
    args = parser.parse_args()

    if args.mode == "fleurs":
        evaluateFleursWer()
    elif args.mode == "local":
        if not os.path.exists(args.dataDir):
            print(f"Directory not found: {args.dataDir}")
            exit(1)
        evaluateLocalCommands(args.dataDir, skip_wake=args.skipWake,
                              run_siamese=not args.noSiamese)
    elif args.mode == "negatives":
        if not os.path.exists(args.dataDir):
            print(f"Directory not found: {args.dataDir}")
            exit(1)
        evaluateNegatives(args.dataDir, run_siamese=not args.noSiamese)
