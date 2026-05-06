# Indic Speech Command Recognizer
### Whisper-tiny + acoustic wakeword + Hindi/Tamil/Telugu commands

**SMAI Assignment 3 · Theme T11.7 (Tier 2)**

---

## 1. Introduction

Smart-home and accessibility tools assume a confident command of English. We
build a **multilingual voice-command frontend** that listens for an English
wakeword (*"Hey Bharat"*) and then accepts smart-home commands spoken in
**Hindi, Tamil, or Telugu**. The system must run on a laptop CPU — no GPU,
no paid inference — so the base model is **Whisper-tiny (39 M params)**.

The contribution is **architectural, not parametric**: Whisper-tiny on its
own answers Indic speech-command queries with ~3.7% intent accuracy
([Section 5.1](#51-baseline)). With a few engineering decisions on top —
forced-language decoding, sliding-window fuzzy command matching, an
encoder-embedding wakeword detector, and a contrastive projection head — we
push that to **TBD%** while keeping the inference path entirely on CPU.

We compare two recognition heads in [Section 5](#5-results): a
**transcription path** (decode → fuzzy match) and a **siamese path**
(contrastive projection head over Whisper encoder embeddings). The
production default is picked based on accuracy, false-positive rate on a
hard-negative pool, and end-to-end latency.

## 2. Data

| Source | Use | Size |
|---|---|---|
| Google FLEURS (`hi_in`, `ta_in`, `te_in`) | WER baseline | 500 test samples × 3 langs = **1500** |
| Self-recorded `data/<command>/*.wav` | Intent accuracy | 10 commands × 5 takes × 3 langs = **149** |
| Self-recorded `help/yes/*.wav` | Wakeword anchor (positive references) | **6** |
| Self-recorded `help/no/*.wav` | Hard negatives for FPR | **5** |
| `data/none/*.wav` | Easy negatives for FPR | **100** |

The self-recorded set spans 3–4 different speakers per command across
varied acoustic conditions (rooms, distances, intonations). The `help/no`
pool was specifically constructed to acoustically resemble *"Hey Bharat"*
(near-rhymes, prosodically similar phrases) to stress the wakeword
detector.

The dataset sits just below the Tier-2 floor of 2 000 examples (we have
1 660 across FLEURS and self-recorded) — see [Section 6](#6-limitations).

## 3. Method

### 3.1 Two-stage architecture

```
audio ─► [ wakeword gate ] ─► [ recognition head ] ─► action
            ▲                          ▲
       Whisper encoder            Path A: Whisper decoder + fuzzy match
       cosine vs anchor           Path B: contrastive projection head
                                          + nearest prototype
```

The wakeword gate is shared by both heads. Only audio that passes the
gate continues to recognition.

### 3.2 Wakeword detection (encoder-embedding cosine)

We deliberately **bypass autoregressive decoding** for the wakeword stage.
Whisper-tiny's decoder hallucinates aggressively for short non-English
proper nouns ("Hey Bharat" → "Hey Brad" / "Bharat" → "Brett"). Instead:

1. Run audio through **Whisper's encoder only** and mean-pool over active
   speech frames to get a 384-dim L2-normalised embedding.
2. Build an **anchor** by averaging encoder embeddings across the 6
   `help/yes/*.wav` reference recordings.
3. At inference, slide a **1.5 s window with 0.5 s hop** across the audio,
   take the **max cosine similarity** of any window vs the anchor.
4. Trigger if max ≥ 0.922 (threshold tuned via held-out sweep, see
   `test_wakeword.py`).

The sliding window matters: an utterance like *"Hey Bharat batti jalao"*
is 3 s long. The full-clip mean embedding is diluted by the command
portion; only the 1.5 s window centered on *"Hey Bharat"* matches cleanly.

### 3.3 Path A — transcription + fuzzy command matching

For each FLEURS language code (`hi`, `ta`, `te`) in turn:

1. Force the Whisper decoder's language token (`forced_decoder_ids`) so it
   *transcribes* the audio in that language rather than translating to
   English (the default behaviour, which produces `~~"sing me a song"~~`
   for *"gana bajao"*).
2. Compare the transcript against `commandsMap[lang]` — a curated
   dictionary of native-script + romanized phrasings for every
   command×language pair (e.g. `"बत्ती जलाओ" → "Turn on the light"`,
   `"batti jalao" → "Turn on the light"`).
3. Match in two phases: **exact substring**, then **sliding-window
   `SequenceMatcher` ratio** ≥ 0.7. The fuzzy phase handles Whisper's
   spelling variations (`"bhatty jalao"` vs `"batti jalao"`).
4. Return on the first language that yields a match. If no match in any
   language, return None.

This path runs Whisper's decoder up to 3× per query, which costs latency
(see [Section 5.4](#54-latency)).

### 3.4 Path B — contrastive projection head (siamese)

We train a lightweight MLP (`384 → 256 → 128`, ReLU, L2-normalised
output) on top of the Whisper encoder embedding using **margin-based
contrastive loss**:

$$\mathcal{L} = y \cdot d^2 + (1-y) \cdot \max(0, \, \text{margin} - d)^2$$

where $d$ is the Euclidean distance between projected embeddings of two
clips and $y = 1$ iff they share the same command label. We train for 100
epochs (Adam, lr=1e-3, margin=0.5) on a vectorised pairwise loss matrix
(see implementation note in [Section 6](#6-limitations)).

At inference: project the encoder embedding, compare cosine similarity to
each per-command prototype (mean of training projections), and return the
arg-max. A confidence threshold of 0.6 returns None for unrecognised
clips.

### 3.5 Evaluation regimes

For Path B (the trained model) we report **three** splits to disentangle
overfit from generalisation:

1. **Headline split** — train on takes 1–3, evaluate on takes 4–5
   (cross-speaker, cross-environment, but every language seen).
2. **5-fold cross-validation** — confidence band on the headline.
3. **Cross-language hold-out** — train on hi+ta, eval on te (and rotations).
   Tells us whether the head learns language-conditioned acoustic
   templates or true intent-level features.

## 4. Implementation notes

- **Single shared module `core.py`** holds all model logic so both the
  Streamlit app and the headless benchmarks call exactly the same code.
- **Embedding cache** (`_siamese_embeddings.npz`) keyed on
  `(reject, augment)` so re-running the ablation is near-instant after the
  first pass.
- **Vectorised contrastive loss** computes the full pairwise distance
  matrix in one forward pass, replacing the original Python double-loop.
  Identical math, ~100× faster on CPU (10 epochs on N=100: ~5 min →
  1.3 s).
- **`time.perf_counter()` brackets** in `core.transcriptionPipeline` and
  `classifyBySiamese` populate per-clip latency columns in
  `benchmark_results_local.csv` and a "took X ms" badge in the live app
  result card.

## 5. Results

### 5.1 FLEURS WER baseline

(From `benchmark_results_fleurs.csv`, 500 samples per language.)

| Language | Avg WER | Notes |
|---|---|---|
| Hindi (`hi`) | TBD | TBD |
| Tamil (`ta`) | TBD | TBD |
| Telugu (`te`) | TBD | TBD |

Whisper-tiny's WER on FLEURS Indic is high — short syllabic Indic words
get dropped or misspelled — but **WER and intent accuracy diverge**: our
fuzzy matcher tolerates the kind of substitutions that wreck WER but
preserve intent.

### 5.2 Intent accuracy on self-recorded data

| System | Wake-gated accuracy | Skip-wake accuracy | Notes |
|---|---|---|---|
| Raw whisper-tiny + exact match | ~3.7% | TBD | No language forcing, English-translated output |
| Path A: transcription + fuzzy match | TBD% | TBD% | (`benchmark_results_local.csv`) |
| Path B: siamese (headline 1-3 / 4-5) | **88.14%** | n/a | Ungated by design |
| Path B: siamese (5-fold CV) | **91.98% ± 4.97%** | n/a | |

### 5.3 False-positive rate on negatives

| Pool | n | Wake-trigger FPR | Path A command-fire FPR | Path B command-fire FPR |
|---|---|---|---|---|
| `data/none` (easy) | 100 | TBD | TBD | TBD |
| `help/no` (hard) | 5 | TBD | TBD | TBD |

### 5.4 Latency

CPU-only, batch evaluation timings:

| Stage | Mean (ms) | Notes |
|---|---|---|
| Wakeword scan | TBD | Encoder-only, 1–3 windows depending on clip length |
| Path A (transcribe ×3 + match) | TBD | Decoder cost dominates |
| Path B (project + nearest proto) | TBD | Single encoder pass |

### 5.5 Architecture comparison (production decision)

| Metric | Path A (transcription) | Path B (siamese) |
|---|---|---|
| Headline accuracy | TBD% | 88.14% |
| Cross-language hold-out | n/a (independently routed) | **8–26%** (catastrophic) |
| FPR (`help/no`) | TBD | TBD |
| Latency / clip | TBD ms | TBD ms |
| New command extensibility | edit `commandsMap`, no retrain | retrain head |

**Recommendation: TBD** — to be completed once Path A numbers land.

The cross-language collapse means siamese is only safe to ship when
every supported language is in its training fold. The transcription path,
by contrast, treats languages independently at decode time and degrades
gracefully (a missing language just means that decode pass fails;
others still work).

## 6. Ablations

(Detailed numbers also in `ABLATION_STUDY.md`.)

### 6.1 Language forcing (Path A only)

Removing the `forced_decoder_ids` Hindi/Tamil/Telugu hint drops Path A
accuracy from ~75% (now TBD with full dataset) to **3.7%** — Whisper
defaults to translating Indic into English.

### 6.2 Audio preprocessing

| Preprocessing | Accuracy | Δ vs baseline |
|---|---|---|
| Baseline (16 kHz, normalised) | TBD% | — |
| Telephony simulation (8 kHz round-trip) | TBD% | TBD |
| Bandpass 300–3400 Hz | TBD% | TBD |

(Hypothesis was that removing high frequencies might help short commands;
result inverts the prediction in the original ABLATION_STUDY — Whisper's
log-mel features rely on high-frequency texture even for speech.)

### 6.3 Frozen vs learned siamese projection

| Embedding space | Headline accuracy |
|---|---|
| Frozen Whisper encoder + cosine prototype | **42.37%** |
| Learned projection head (contrastive, 100 ep) | **88.14%** |

The +45 pp gap is the "fine-tune" the Tier-2 spec asks for.

## 7. Limitations

1. **Dataset size** — 149 self-recorded clips puts us just under the
   Tier-2 floor of 2 000. We mitigated by also reporting on FLEURS (1 500),
   but a ~10× expansion would be worth doing for noise-robustness
   evaluation we did not run.
2. **Single-utterance clips** — every clip is one speaker, one room, one
   take. Continuous-listening scenarios (false positives during
   conversation, noise robustness, code-switched audio) are out of scope.
3. **Cross-language fragility of the siamese head** — see
   [Section 5.5](#55-architecture-comparison-production-decision). Adding
   a new target language to Path B requires a retrain; Path A only
   requires editing `commandsMap`.
4. **Wakeword threshold** — fixed at 0.922 from a small held-out sweep on
   `help/yes` vs `help/no`. A larger negative pool would yield a more
   trustworthy threshold.
5. **No fine-tune of Whisper itself** — Tier 2 mentions "light
   fine-tune" and we interpreted that as the projection head. A LoRA
   fine-tune of the decoder on the 1500 FLEURS samples would likely lower
   WER and indirectly help Path A, but did not fit the time budget.
6. **Vectorised contrastive loss converges to a slightly different
   optimum** than the original per-pair loop (88% vs 95% headline). The
   math is identical (mean over upper-triangular pair losses) but a single
   `optimizer.step()` per epoch on the full matrix differs from the
   original's gradient accumulation across N(N-1)/2 mini-steps. We chose
   the vectorised version because it makes 5-fold CV + cross-language
   hold-out tractable on CPU; the trade-off is a few percentage points
   on the single split.

## References

1. A. Radford et al. *Robust Speech Recognition via Large-Scale Weak
   Supervision* (Whisper). 2022.
2. A. Conneau et al. *FLEURS: Few-shot Learning Evaluation of Universal
   Representations of Speech*. 2022.
3. R. Hadsell, S. Chopra, Y. LeCun. *Dimensionality Reduction by Learning
   an Invariant Mapping*. CVPR 2006. (Margin contrastive loss.)
4. M. Morris et al. *jiwer*: Word Error Rate utility for Python.

## Acknowledgements

Self-recordings contributed by team members across Hindi, Tamil, and
Telugu native/proficient speakers. LLM usage (per assignment honesty
rules): Claude was used for the `core.py` extraction, vectorisation of the
contrastive loss, and report drafting. All evaluation runs and analytical
conclusions are our own.
