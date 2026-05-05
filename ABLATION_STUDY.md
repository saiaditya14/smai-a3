# Ablation Study Results

## Overview
To validate the necessity of our engineering choices (fuzzy matching, forced language tokens, wakeword detection), we conducted an ablation study comparing the raw, unguided `openai/whisper-tiny` model against our custom pipeline.

## Results
- **Our Pipeline Accuracy:** ~75% (up to ~80% under optimal conditions)
- **Raw Baseline Accuracy:** ~3.7%

## Analysis
The massive drop in accuracy (from ~75% to 3.7%) when removing our custom logic perfectly demonstrates why the base model cannot be used out-of-the-box for this task. 

The raw baseline struggles significantly due to:
1. **Lack of Language Forcing:** Without explicitly providing the language token, Whisper-tiny often guesses the wrong language or attempts to translate the audio into English instead of transcribing the native words.
2. **Spelling Variations:** Whisper frequently produces minor spelling variations for Indic phonetic words (e.g., "bhatty jalao" vs "batti jalao"). The raw exact-match baseline fails here, proving the necessity of our sliding-window fuzzy matching algorithm.
3. **Prompt Biasing:** Our pipeline uses prompt biasing during the English wakeword phase to prime the model, which the raw baseline lacks.

## Conclusion
The massive ~20x improvement in accuracy mathematically justifies the architecture of `app.py`. The fuzzy matching and language-forcing pipeline is absolutely critical for making small, edge-friendly models like `whisper-tiny` viable for multilingual command recognition.


0.908 had 80%

---

### Ablation : Audio Preprocessing

**Concept:** 
Whisper internally resamples everything to 16 kHz. We evaluate what happens to the audio before it reaches Whisper using the best performing baseline (Whisper Tiny with Language Forcing).

**Hypothesis:**
Removing high-frequency noise (like simulating telephony) could actually help recognition for short commands by filtering out background features that are irrelevant. 

**Tested Combinations:**
1.  **Baseline**: Native sample rate, converted to 16kHz.
2.  **Telephony Simulation**: Resampled to 8 kHz to remove high frequencies (>4kHz), then upsampled to 16kHz. 
3.  **Bandpass Filter**: 300–3400 Hz strict bandpass using scipy `butter` and `sosfilt`.

**Results:**
-   **Baseline Accuracy**: `68.5%`
-   **Telephony Accuracy**: `59.1%`
-   **Bandpass Filter Accuracy**: `67.8%`

**Conclusion:**
Contrary to the hypothesis, removing frequencies above 4 kHz (telephony) severely degraded performance. The strict bandpass filter performed very closely to the baseline but did not top it. This indicates Whisper's feature extraction implicitly utilizes high-frequency textures even for speech commands. The raw baseline is best.

---

### Ablation : Siamese Network

**Concept:**
Instead of transcribing audio to text and string-matching, we directly compare Whisper's internal audio embeddings. 

**Implementation Strategy:**
We extract encoder embeddings from Whisper before the decoder. 5 reference recordings per command and language are passed. We train using a subset of references and evaluate on a validation set. At inference, cosine similarity dictates the matched command.

**Tested Conditions:**
1. **Frozen Whisper Encoder**: We take the out-of-the-box frozen encoder weights, extract hidden states, and evaluate similarities.
2. **Learned Projection Head**: We train a Multi-Layer Perceptron projection head over the embeddings with Contrastive Loss initialized on augmented pairing labels (1 for same command, 0 for different command).

**Hypothesis:**
The learned projection head will map the embeddings into a tighter cluster per command, bringing together multilingual audio files into a single intent representation faster compared to frozen embeddings. 

**Results:**
-   **Frozen Encoder Accuracy**: `44.1%`
-   **Learned Projection Head Accuracy**: `94.9%`

**Conclusion:**
The standard Euclidean/Cosine space from Whisper's bare encoder performs poorly for direct intent mapping since Whisper embeddings are structured for autoregressive transcription rather than direct acoustic classification. However, learning a lightweight contrastive projection transforms the embeddings remarkably, shooting accuracy up to 94.9%, vastly outperforming string matching transcription architectures.
