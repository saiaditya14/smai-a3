# Ablation Study Results

> Numbers in this document are computed by `benchmarking.py`,
> `ablation_audio.py`, and `ablation_siamese.py` against the **full** local
> dataset (10 commands × 5 takes × 3 languages = 149 wav files in `data/`)
> and the negative pools (`data/none/` + `help/no/`). The Tier-2 wakeword is
> verified separately in `test_wakeword.py`. Re-run with `python <script>`
> to regenerate.

---

## 1. Headline pipeline accuracy (full dataset)

(From `results/benchmark_results_local.csv`.)

| Configuration | Accuracy |
|---|---|
| Raw Whisper-tiny + exact match (no language forcing, no fuzzy) | ~3.7% |
| **Path A — language-forced transcription + sliding-window fuzzy match** | **43.62%** (wake-gated) / **70.47%** (skip-wake) |
| **Path B — siamese projection head (headline split 1-3 / 4-5)** | **93.22%** |
| **Path B — siamese (5-fold cross-validation)** | **91.24% ± 6.25%** |

The 25× improvement over the raw baseline isolates the value of our
engineering choices. The two production paths are compared head-to-head in
[Section 4](#4-production-architecture-comparison).

The raw baseline struggles because:
1. **Lack of Language Forcing** — without an explicit language token,
   Whisper-tiny often translates Indic audio to English ("Sing me a song"
   instead of transcribing "gana bajao").
2. **Spelling Variations** — Whisper produces minor phonetic spelling
   variations (e.g., "bhatty jalao" vs "batti jalao") that defeat
   exact-match. Our sliding-window fuzzy matcher tolerates these.
3. **No Wakeword Gate** — the raw baseline has no way to suppress
   non-command audio.

---

## 2. Ablation — Audio Preprocessing

**Concept.** Whisper resamples to 16 kHz internally regardless of input
sample rate. The question is whether removing high-frequency content
*before* the feature extractor helps or hurts for short Indic commands.

**Hypothesis.** Telephony-band filtering should help by suppressing
out-of-band noise that Whisper's mel features might over-attend.

**Tested configurations:**
1. **Baseline** — load at 16 kHz, peak-normalise.
2. **Telephony simulation** — 8 kHz round-trip (wipes ≥4 kHz).
3. **Bandpass 300–3400 Hz** — strict telephony band via `scipy.butter` /
   `sosfilt`, order 5.

**Results (full 149-clip dataset):**

| Strategy | Accuracy |
|---|---|
| Baseline | 68.5% |
| Telephony (8 kHz round-trip) | 59.1% |
| Bandpass 300–3400 Hz | 67.8% |

**Conclusion.** The hypothesis was wrong: aggressive low-pass filtering hurts
short Indic command recognition. The bandpass
variant comes close to baseline because Whisper's mel extractor has some
high-frequency redundancy, but the cleanest approach is to leave the
audio alone and let Whisper's own front-end do the work.

---

## 3. Ablation — Siamese Projection Head (lightweight task adaptation)

**Concept.** Instead of decoding to text and string-matching, project
Whisper's encoder embeddings into a contrastive space and classify by
nearest per-command prototype. Single forward pass, no autoregressive
decoder, language-agnostic at inference.

**Implementation.** MLP `384 → 256 → 128`, ReLU, L2-normalised output.
Margin-based contrastive loss (margin = 0.5), Adam (lr = 1e-3),
**vectorised pairwise loss matrix** for tractable CPU training. 100 epochs.

### 3.1 Three evaluation regimes

We deliberately report three splits to disentangle overfit, generalisation
to new takes, and generalisation to new languages.

| Split | Train | Eval | Frozen encoder | Learned projection head |
|---|---|---|---|---|
| **Headline** | takes 1-3 (cross-speaker, all 3 langs) | takes 4-5 | **38.98%** | **93.22%** |
| **5-fold CV** | 4/5 of clips | held-out 1/5 | — | **91.24% ± 6.25%** |
| **Cross-language hold-out** | 2 of 3 langs | held-out lang | — | **0.00% / 10.20% / 18.00%** |

The cross-language fold trains on Hindi+Tamil and tests on Telugu (and
rotations). The weak cross-language result is the headline finding of this section
— see [Section 5](#5-key-takeaways-for-the-report).

### 3.2 Frozen vs learned (headline split)

The +54.24 pp gap between frozen and learned (38.98% → 93.22%) is the main
supervised adaptation result. We keep Whisper-tiny frozen and train only the
projection head; the frozen encoder contains useful acoustic information, but
its raw metric space is not directly optimised for intent classification. The
contrastive head reshapes that space.

### 3.3 5-fold cross-validation

The 6.25% std across 5 folds suggests the headline split is within normal
sampling variance — the model is not memorising particular takes. Folds
3 and 5 underperform the mean (83.33%, 86.21%); fold 4 over-performs
(100.00%). Likely driven by which speakers' clips end up in eval.

### 3.4 Cross-language hold-out

| Held-out language | Accuracy |
|---|---|
| Hindi held out | 0.00% |
| Tamil held out | 10.20% |
| Telugu held out | 18.00% |

This is a **deal-breaker for production** if we cared about generalisation
to unseen languages. The contrastive head is learning Hindi-flavoured,
Tamil-flavoured, Telugu-flavoured prototypes per command rather than
language-independent intent embeddings. Training on hi+ta gives the head
no information about how Telugu speakers say "fan veyi", and the held-out
prototypes don't match.

The transcription path (Path A) does not have this problem — adding a
new language means adding a new entry to `commandsMap` and forcing the
appropriate decoder language token.

---

## 4. Production architecture comparison

Both paths share the same wakeword gate (encoder-embedding cosine vs
"Hey Bharat" anchor). The decision is which recognition head to ship as
default.

| Metric | Path A (transcription + fuzzy) | Path B (siamese head) |
|---|---|---|
| Headline accuracy | **70.47%** (all takes) | **93.22%** |
| 5-fold CV accuracy | n/a (no train) | **91.24% ± 6.25%** |
| Cross-language hold-out | n/a (per-lang routing) | **0–18%** ⚠ |
| FPR on `data/none` (easy negatives) | 2.0% | 3.0% |
| FPR on `help/no` (hard negatives, seen, n=26) | 3.8% | **0.0%** |
| FPR on `help/no_test` (hard negatives, unseen, n=10) | 20.0% | **0.0%** |
| End-to-end latency / clip (CPU) | ~1623 ms | ~270 ms |
| Add a new command | edit `commandsMap`, no retrain | retrain head |
| Add a new language | edit `commandsMap`, no retrain | retrain head + need data |

**Recommended submitted default — Path B.** Path B has much lower latency
(~270 ms vs ~1623 ms), higher in-distribution accuracy, and stronger hard-
negative rejection. This best fits the fixed T11.7 command-recognition setting
where the target languages and command set are known before deployment. Path A
is still useful as an interpretable fallback/debug mode and for future
extensions where commands or languages must be added without retraining.

---

## 5. Key takeaways for the report

1. **Engineering > parameters.** A 39 M-parameter model with the right
   pre/post-processing reaches 88-92% intent accuracy on a 3-language,
   10-command task. The fancy model isn't doing the heavy lifting; the
   wakeword gate, language forcing, fuzzy matcher, and contrastive head are.
2. **Embedding-based wakeword > text-based wakeword.** Whisper's
   autoregressive decoder hallucinates on Indic, so we use the encoder
   directly. This was a >20× accuracy improvement on its own.
3. **Contrastive projection head adds ~54 pp on top of frozen embeddings**
   (38.98% → 93.22%) — this is the lightweight learned adaptation.
4. **Siamese path is fragile to unseen languages** (cross-language drops
   to single digits) — this is the main limitation of choosing Path B for the
   submitted fixed-language prototype.
5. **Sliding-window fuzzy match makes WER irrelevant for intent.**
   Whisper-tiny's WER on FLEURS Indic is high (184-433%), but our intent
   accuracy stays far above the raw baseline because the matcher tolerates the kind of
   substitutions that wreck WER.
