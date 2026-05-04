# Proposed Systems-Level Evaluations for Pipeline Upscaling

## 1. Component-Level Latency Profiling
**Objective:** Prove the architecture is viable for real-time edge deployment by isolating compute bottlenecks.
**Methodology:** 
Instead of a single End-to-End (E2E) latency metric that gets penalized by disk I/O, we track three distinct micro-benchmarks using high-precision timers (`time.perf_counter()`):
*   $T_{preprocess}$: Audio array normalization.
*   $T_{inference}$: The raw forward pass of the `whisper-tiny` model.
*   $T_{algorithm}$: The execution time of our custom logic (language routing, fuzzy matching, dictionary lookups).
**Expected Analysis:** This will empirically demonstrate that our algorithmic interventions (fuzzy matching, routing) add negligible overhead (e.g., < 5ms) to the pipeline, proving the software layer is highly efficient and that the primary bottleneck remains the base model's compute time.

## 2. Confusion Matrix and Error Analysis
**Objective:** Identify specific phonetic or linguistic failure points beyond top-line accuracy.
**Methodology:** 
Map the predictions from `benchmark_results_local.csv` into a standard Confusion Matrix.
**Expected Analysis:** Move beyond aggregate percentages to diagnose exact failure modes. For example, analyzing if the system consistently confuses "turn *on* the light" with "turn *off* the light" in specific languages due to phonetic similarities in the trailing syllables. 

## 3. Word Error Rate (WER) vs. Intent Accuracy
**Objective:** Justify the semantic pipeline architecture over a strict Speech-to-Text approach.
**Methodology:** 
Calculate the standard Word Error Rate (WER) for the raw model output, contrasting it directly with the final Intent Accuracy (API trigger success rate).
**Expected Analysis:** The model will likely exhibit a high WER due to spelling variations and transliteration inconsistencies. However, demonstrating a high Intent Accuracy (~75%) proves that the sliding-window fuzzy matching effectively bypasses the traditional constraints of STT systems, bridging the gap between imperfect phonetic recognition and strict programmatic execution.