# Indic Speech Command Recognizer (SMAI A3 — T11.7)

**Repository:** https://github.com/saiaditya14/smai-a3  
**Deployed prototype:** https://engja9hyuz4sjstbxdka2e.streamlit.app/

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
app.py                    Streamlit UI (sidebar toggles transcription / siamese / both)
core.py                   shared pipeline module — wakeword, transcribe, siamese
benchmarking.py           FLEURS WER + local-command accuracy + negatives FPR
ablation_audio.py         audio preprocessing ablation
ablation_siamese.py       siamese ablation: headline + k-fold + cross-language splits
test_wakeword.py          standalone wakeword threshold sweep
data/                     self-recorded multilingual command clips
help/                     wakeword positives + hard negatives
models/                   saved siamese head/prototypes and embedding cache
assets/                   UI SVG/image assets
results/                  benchmark CSVs and run logs
docs/REPORT.md            6-8 page Tier 2 technical report
docs/SUBMISSION.md        concise PDF assembly source with screenshots/links
docs/ABLATION_STUDY.md    ablation numbers + analysis
docs/screenshots/         deployed prototype screenshots
```

## Run

```bash
pip install -r requirements.txt

# 1. Train + persist the siamese head (caches embeddings to models/_siamese_embeddings.npz)
python ablation_siamese.py --reject

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

| System | Result |
|---|---|
| Path A: transcription + fuzzy match | 70.47% skip-wake intent accuracy |
| Path B: siamese projection head | 93.22% headline / 91.24% ± 6.25% 5-fold CV |
| Hard-negative command FPR (`help/no_test`) | 20.0% Path A / 0.0% Path B |

For the submitted T11.7 prototype, **Path B is the recommended default**:
it is faster, more accurate on the fixed Indic command set, and has stronger
hard-negative rejection. Path A remains available as an interpretable fallback
and extension path for new commands/languages without retraining.

See `docs/ABLATION_STUDY.md` for the full table and `docs/REPORT.md` for the
final writeup.

## Acknowledgements

- OpenAI Whisper (whisper-tiny) for the base ASR + acoustic encoder.
- Google FLEURS for held-out Indic transcription evaluation.
- Self-recordings by team members across Hindi, Tamil, and Telugu.

LLM usage (per assignment honesty rules): Claude was used for code scaffolding
of `core.py`, ablation harness structure, and report drafting. All evaluation
runs and analysis are our own.
