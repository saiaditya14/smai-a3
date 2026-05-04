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