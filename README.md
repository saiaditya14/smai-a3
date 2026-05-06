# Indic Speech Command Recognizer (SMAI A3 — T11.7)

Tier 2 implementation of the *Wakeword + multi-language commands* variant: a
Streamlit app that listens for the English wakeword **"Hey Bharat"** and then
executes one of 11 smart-home commands spoken in **Hindi, Tamil, or Telugu**.

The system runs end-to-end on **Whisper-tiny (39 M params, CPU)**. No paid
inference. Two recognition heads are compared:

1. **Transcription path** — forced-language Whisper decoding + sliding-window
   fuzzy match against a curated `commandsMap` (native script + romanized).
2. **Siamese path** — a contrastive projection head trained on Whisper
   encoder embeddings, classified by nearest prototype.

Both share the same acoustic-embedding wakeword detector (Whisper encoder +
cosine similarity to a "Hey Bharat" anchor).

## Repo layout

```
core.py                shared pipeline module — wakeword, transcribe, siamese
app.py                 Streamlit UI (sidebar toggles transcription / siamese / both)
benchmarking.py        FLEURS WER + local-command accuracy + negatives FPR
ablation_audio.py      Audio preprocessing ablation (baseline / telephony / bandpass)
ablation_siamese.py    Siamese ablation: headline + k-fold + cross-language splits
test_wakeword.py       Standalone wakeword threshold sweep
data/                  Self-recorded multilingual command clips (~174 wav)
help/yes/              "Hey Bharat" reference recordings (anchor source)
help/no/               Hard negatives that should NOT trigger the wakeword
ABLATION_STUDY.md      Ablation numbers + analysis
REPORT.md              6-8 page Tier 2 technical report
pitch.md               One-slide LinkedIn-shareable pitch
```

## Run

```bash
pip install -r requirements.txt

# 1. Train + persist the siamese head (caches embeddings to _siamese_embeddings.npz)
python ablation_siamese.py

# 2. Run benchmarks
python benchmarking.py --mode local       # transcription + siamese accuracy
python benchmarking.py --mode negatives   # FPR on data/none + help/no
python benchmarking.py --mode fleurs      # WER on Google FLEURS test split

# 3. Audio preprocessing ablation
python ablation_audio.py

# 4. Launch the demo app
streamlit run app.py
```

## Headline numbers

See `ABLATION_STUDY.md` for the full table. Architecture comparison and the
recommended production default are documented there.

## Acknowledgements

- OpenAI Whisper (whisper-tiny) for the base ASR + acoustic encoder.
- Google FLEURS for held-out Indic transcription evaluation.
- Self-recordings by team members across Hindi, Tamil, and Telugu.

LLM usage (per assignment honesty rules): Claude was used for code scaffolding
of `core.py`, ablation harness structure, and report drafting. All evaluation
runs and analysis are our own.
