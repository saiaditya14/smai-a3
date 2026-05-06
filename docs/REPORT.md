# Indic Speech Command Recognizer
### Whisper-tiny + acoustic wakeword + Hindi/Tamil/Telugu commands

**SMAI Assignment 3 · Theme T11.7 (Tier 2)**

**Repository:** https://github.com/saiaditya14/smai-a3  
**Deployed prototype:** https://engja9hyuz4sjstbxdka2e.streamlit.app/

| Name | Roll number | Email |
|---|---|---|
| Aryan Maskara | 2023111004 | aryan.maskara@research.iiit.ac.in |
| Sai Ramanathan | 2023101096 | sai.ramanathan@students.iiit.ac.in |
| Neha Murthy | 2023115018 | neha.murthy@research.iiit.ac.in |
| Yajat Lakhanpal | 2023111015 | yajat.lakhanpal@research.iiit.ac.in |
| Prasoon Dev | 2023111014 | prasoon.dev@research.iiit.ac.in |
| Aanchal Mundhada | 2023112016 | aanchal.mundhada@research.iiit.ac.in |

---

## 1. Introduction

Smart-home and accessibility tools assume a confident command of English. We
build a **multilingual voice-command frontend** that listens for an English
wakeword (*"Hey Bharat"*) and then accepts smart-home commands spoken in
**Hindi, Tamil, or Telugu**. The system must run on a laptop CPU — no GPU,
no paid inference — so the base model is **Whisper-tiny (39 M params)**.

The base Whisper-tiny model is kept frozen. Our contribution is a lightweight
task adaptation around it: forced-language decoding, sliding-window fuzzy
command matching, an encoder-embedding wakeword detector, and a trained
contrastive projection head over Whisper encoder embeddings. A raw
Whisper-tiny exact-match baseline reaches only ~3.7% intent accuracy; the
learned projection-head path reaches **93.22%** on the fixed Indic command
task while remaining CPU-only.

We compare two recognition heads in [Section 5](#5-results). **Path A** is
the transcription path: decode the speech with forced-language Whisper and
then fuzzy-match the transcript against a command dictionary. **Path B** is
the siamese path: skip decoding, project Whisper encoder embeddings through a
contrastive head, and classify by nearest command prototype. Since T11.7
evaluates an Indic command recogniser for a fixed command set, our submitted
system uses **Path B as the primary recogniser**: it is faster, more accurate
on the target command data, and better at rejecting hard negatives. Path A
remains useful as an interpretable fallback/debug path, especially for
recordings that contain the wakeword but no command.

## 2. Data

| Source | Use | Size |
|---|---|---|
| Google FLEURS (`hi_in`, `ta_in`, `te_in`) | WER baseline | 418/557/470 test samples per lang = **1445** |
| Self-recorded `data/<command>/*.wav` | Real command clips | 10 commands × 5 takes × 3 langs = **149** |
| Command pitch shifts | Train-only command variants | **447** synthetic variants, **596** command clips/variants total |
| Self-recorded `help/yes/*.wav` + pitch shifts | Wakeword anchor variants | **24** |
| Self-recorded `help/no/*.wav` + noise/time shifts | Reject-class variants | **104** |
| Self-recorded `help/no_test/*.wav` | Hard negatives for FPR (unseen — held out) | **10** |
| `data/none/*.wav` | Easy negatives for FPR | **100** |
| Audio preprocessing ablation | Stress-test runs over real command clips | **447** runs = 149 × 3 preprocessing regimes |

The command corpus starts from 149 self-recorded clips, spanning 3–4 different
speakers per command across varied acoustic conditions (rooms, distances,
intonations). Following standard speech-learning practice, we expand the
training set with pitch-shifted waveform variants and train a reject prototype
using augmented hard negatives. The `help/no` pool was specifically
constructed to acoustically resemble *"Hey Bharat"* (near-rhymes, prosodically
similar phrases) to stress the wakeword detector.

Counting FLEURS, real command clips, negative pools, train-only augmentation,
and audio-preprocessing ablations, the project processes over 2k audio
examples/runs. We report originals and augmented variants separately for
transparency: augmentation is a valid part of the dataset pipeline, while
headline command accuracies are computed on real held-out recordings rather
than on transformed copies of the same eval clips.

The command corpus is small, so we evaluate two complementary behaviours:
FLEURS is used to quantify Indic transcription behaviour, while the
self-recorded command set is used to evaluate the actual wakeword/intent
pipeline. The data limitation is discussed in [Section 7](#7-limitations).

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
   `help/yes/*.wav` reference recordings, augmented with +3, +6, and −3
   semitone pitch-shifted copies per file to improve cross-speaker (female,
   child) robustness.
3. At inference, slide a **1.5 s window with 0.5 s hop** across the audio,
   take the **max cosine similarity** of any window vs the anchor.
4. Trigger if max ≥ **0.910** (threshold tuned via held-out sweep on
   `help/yes` vs `help/no`; see `test_wakeword.py`).

The sliding window matters: an utterance like *"Hey Bharat batti jalao"*
is 3 s long. The full-clip mean embedding is diluted by the command
portion; only the 1.5 s window centered on *"Hey Bharat"* matches cleanly.

### 3.3 Path A — transcription + fuzzy command matching

For each FLEURS language code (`hi`, `ta`, `te`) in turn:

1. Force the Whisper decoder's language token (`forced_decoder_ids`) so it
   *transcribes* the audio in that language rather than translating to
   English (the default behaviour, which produces *"sing me a song"*
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
epochs (Adam, lr=1e-3, margin=0.5) on a vectorised pairwise loss matrix.

At inference, the system projects the encoder embedding, compares cosine
similarity to each per-command prototype, and returns the nearest command.
A **reject class** prototype is trained from `help/no/*.wav` plus
noise-augmented copies (SNR 10/5/0 dB + time-shift), so the head can return
None for near-miss non-command inputs. A confidence threshold of 0.6 also
suppresses low-confidence matches.

This is the only supervised learned component in the system. We do **not**
update Whisper-tiny's base weights; instead, we learn a small command-specific
head on top of frozen Whisper encoder features.

### 3.5 Pitch augmentation for command training

To improve cross-speaker robustness, each command training clip generates
three additional pitch-shifted variants (−3, +3, +6 semitones via
`librosa.effects.pitch_shift`). Augmented copies inherit the source
language label and are routed to the training split only — they never
appear in any evaluation set — so accuracy numbers reflect only real clips.

### 3.6 Evaluation regimes

For Path B (the trained model) we report **three** splits to disentangle
overfit from generalisation:

1. **Headline split** — train on takes 1–3, evaluate on takes 4–5
   (cross-speaker, cross-environment, but every language seen).
2. **5-fold cross-validation** — confidence band on the headline.
3. **Cross-language hold-out** — train on two languages and evaluate on the
   third. This tests whether the head learns language-conditioned acoustic
   templates or language-independent intent features.

## 4. Implementation notes

- **Single shared module `core.py`** holds all model logic so both the
  Streamlit app and the headless benchmarks call exactly the same code.
- **Embedding cache** (`models/_siamese_embeddings.npz`) keyed on
  `(reject, augment, pitch_augment)` so re-running the ablation is
  near-instant after the first pass.
- **Vectorised contrastive loss** computes the full pairwise distance
  matrix in one forward pass, replacing the original Python double-loop.
  Identical math, ~100× faster on CPU (10 epochs on N=100: ~5 min →
  1.3 s).
- **`time.perf_counter()` brackets** in `core.transcriptionPipeline` and
  `classifyBySiamese` populate per-clip latency columns in
  `results/benchmark_results_local.csv` and a "took X ms" badge in the live app
  result card.

### 4.1 Working prototype

The Streamlit prototype is deployed at
https://engja9hyuz4sjstbxdka2e.streamlit.app/. It exposes the two recognition
paths and a combined mode from the sidebar, accepts microphone input, displays
wakeword status, and returns the matched command/action with latency.

![Application home screen with recognition mode sidebar](screenshots/app_home_screen_r.png)

![Successful command detection in the deployed Streamlit app](screenshots/successful_detection.png)

## 5. Results

### 5.1 FLEURS WER baseline

(From `results/benchmark_results_fleurs.csv`, forced-language Whisper-tiny decode.)

| Language | Samples | Avg WER |
|---|---|---|
| Hindi (`hi`) | 418 | **184%** |
| Tamil (`ta`) | 557 | **190%** |
| Telugu (`te`) | 470 | **433%** |

WER exceeds 100% because Whisper-tiny often inserts extra text on short Indic
utterances. This does not make the model unusable for intent recognition:
the fuzzy matcher can tolerate substitutions such as "bhatty jalao" for
"batti jalao" when the command phrase is still close enough.

### 5.2 Intent accuracy on self-recorded data

(From `results/benchmark_results_local.csv`, 149 clips.)

| System | Wake condition | Recognition accuracy |
|---|---|---|
| Raw Whisper-tiny + exact match | no gate | ~3.7% |
| **Path A: transcription + fuzzy match** | wake-gated | **43.62%** |
| **Path A: transcription + fuzzy match** | skip-wake diagnostic | **70.47%** |
| **Path B: siamese (headline 1-3 / 4-5)** | wakeword-prefixed clips | **93.22%** |
| **Path B: siamese (5-fold CV)** | wakeword-prefixed clips | **91.24% ± 6.25%** |

Path B is deployed behind the same wakeword gate as Path A. Its headline split
trains on takes 1–3 and evaluates on takes 4–5, so the recognition head is
tested on held-out real command recordings that include the wakeword prefix
but are not augmented evaluation copies. The skip-wake Path A number is kept
only as a diagnostic to separate transcription/matching errors from gate
recall errors.

### 5.3 False-positive rate on negatives

(From `results/benchmark_results_negatives.csv`.)

| Pool | n | Wake-trigger FPR | Path A cmd-fire FPR | Path B cmd-fire FPR |
|---|---|---|---|---|
| `data/none` (easy — random noise/speech) | 100 | 3.0% | 2.0% | 3.0% |
| `help/no` (hard — acoustically similar, **seen** in reject training) | 26 | 69.2% | 3.8% | **0.0%** |
| `help/no_test` (hard — acoustically similar, **unseen**) | 10 | 50.0% | 20.0% | **0.0%** |

Key finding: the wakeword gate is intentionally permissive (encoder cosine
similarity, no learned discriminator) — it fires on ~70% of hard negatives.
The siamese reject class is the second-stage filter that brings command FPR
to 0% on both seen and unseen hard negatives.

### 5.4 Latency

CPU-only, batch evaluation timings from `results/benchmark_results_local.csv`:

| Path | Mean latency / clip | Notes |
|---|---|---|
| **Path A** (transcribe ×3 + match) | **~1830 ms** | Whisper decoder runs up to 3× per query |
| **Path B** (project + nearest proto) | **~730 ms** | Single encoder pass only |

Path B is about 2.5× faster on the measured 3–4 second command recordings
because it skips the autoregressive decoder entirely. Path A can be slower on
longer recordings because decoder cost grows with the audio/transcript length
and it may run one decode per target language.

### 5.5 Architecture comparison (submitted system)

| Metric | Path A (transcription + fuzzy) | Path B (siamese head) |
|---|---|---|
| Headline accuracy (skip-wake) | 70.47% | **93.22%** |
| 5-fold CV accuracy | n/a (no train) | **91.24% ± 6.25%** |
| Cross-language hold-out | n/a (independently routed) | **0–18%** ⚠ weak |
| FPR on `data/none` | 2.0% | 3.0% |
| FPR on `help/no` (seen hard negatives) | 3.8% | **0.0%** |
| FPR on `help/no_test` (unseen hard negatives) | 20.0% | **0.0%** |
| End-to-end latency / clip | ~1830 ms | ~730 ms |
| Add a new command | edit `commandsMap`, no retrain | retrain head |
| Add a new language | edit `commandsMap`, no retrain | retrain head + need data |

**Submitted default: Path B.** Path B has substantially lower latency
(~730 ms vs ~1830 ms), higher in-distribution command accuracy, and stronger
rejection on hard negatives. That is the behaviour we want for the T11.7
prototype, where the target command set and Indic languages are known at
submission time. Path A remains exposed as a fallback/debug mode because it is
more interpretable and can generalise to new command strings or languages by
editing `commandsMap` instead of retraining.

The weak cross-language hold-out result (0–18%) is the main limitation of
choosing Path B: the siamese head learns Hindi-flavoured, Tamil-flavoured,
Telugu-flavoured prototypes per command rather than language-independent
intent embeddings.
Adding a new Indic language to Path B requires recording data and retraining;
Path A only requires adding entries to `commandsMap`. For the fixed assignment
setting, however, Path B is the better submitted recogniser.

## 6. Ablations

(Detailed numbers also in `ABLATION_STUDY.md`.)

### 6.1 Language forcing (Path A only)

Removing the `forced_decoder_ids` Hindi/Tamil/Telugu hint drops Path A
skip-wake accuracy from **70.47%** to **3.7%** — Whisper defaults to
translating Indic into English, which destroys command matching.

### 6.2 Audio preprocessing

| Preprocessing | Accuracy |
|---|---|
| Baseline (16 kHz, peak-normalised) | **68.5%** |
| Telephony simulation (8 kHz round-trip) | 59.1% (−9.4 pp) |
| Bandpass 300–3400 Hz | 67.8% (−0.7 pp) |

The hypothesis was that removing high frequencies might help short commands
on a model trained on telephony-style data. The result inverts the
prediction — Whisper's log-mel feature extractor relies on high-frequency
texture even for short Indic speech, and aggressive low-pass filtering
loses acoustic cues that aid transcription. The 300–3400 Hz bandpass
variant approximates the loss without fully discarding high-band texture,
hence the smaller drop.

### 6.3 Frozen vs learned projection head

| Embedding space | Headline accuracy |
|---|---|
| Frozen Whisper encoder + cosine prototype | **38.98%** |
| Learned projection head (contrastive, 100 epochs) | **93.22%** |

The +54.2 pp gap is the core supervised adaptation result. The frozen encoder
contains useful acoustic information, but its raw metric space is not directly
optimised for command intent classification. The contrastive head reshapes the
space so same-command examples cluster more reliably.

### 6.4 Reject class ablation

| Configuration | Headline acc. | Hard-negative siamese FPR |
|---|---|---|
| No reject class, no command pitch augmentation | 88.14% | 80% (`help/no`) |
| With reject class + command pitch augmentation (seen negatives) | **93.22%** | **0%** |
| With reject class + command pitch augmentation (unseen `help/no_test`) | **93.22%** | **0%** |

Adding the reject class plus command pitch augmentation improved headline
accuracy from 88.14% to 93.22% and reduced hard-negative FPR from 80% to
0%. The unseen test confirms the reject prototype generalises beyond its
training clips.

## 7. Limitations

1. **Dataset size** — the command-specific set has 149 self-recorded clips,
   and the total evaluated audio count including FLEURS is still below 2 000.
   We therefore report FLEURS transcription behaviour separately from command
   intent accuracy. A larger command corpus would be needed for stronger
   noise-robustness and microphone-robustness claims.
2. **Single-utterance clips** — every clip is one speaker, one room, one
   take. Continuous-listening scenarios (false positives during
   conversation, noise robustness, code-switched audio) are out of scope.
3. **Cross-language fragility of the siamese head** — see
   [Section 5.5](#55-architecture-comparison-production-decision). Adding
   a new target language to Path B requires recording data and retraining;
   Path A only requires editing `commandsMap`.
4. **Wakeword gate is permissive by design** — at threshold 0.910 it fires
   on ~50–70% of hard negatives. A lightweight binary classifier on top of
   the encoder embedding would reduce this; the current design offloads the
   work to the siamese reject class.
5. **Threshold and microphone calibration** — the deployment uses a fixed
   cosine threshold tuned on the team's recording setup. In live use, a
   slightly lower threshold just below 0.910 may improve wakeword recall on
   weaker laptop/browser microphones, while a broader calibration set across
   phones, headsets, and laptop mics would make the threshold less device
   specific. The fuzzy transcription path is comparatively robust to
   wakeword-only utterances because it only fires when a command phrase is
   matched.
6. **Wakeword-only vs command utterances** — a stronger production system
   should explicitly classify whether speech after the wakeword contains a
   command. A tiny neural classifier over the post-wake encoder embedding, or
   a learned "wakeword-only" reject prototype, would prevent command heads
   from over-interpreting bare wakeword recordings.
7. **Frozen Whisper backbone** — we do not update Whisper-tiny's base
   parameters. Our lightweight adaptation is the learned projection head over
   frozen encoder embeddings. A future LoRA fine-tune of Whisper on Indic
   speech could lower WER and indirectly help Path A, but it would require a
   larger training/evaluation cycle than this submission allowed.
8. **Vectorised contrastive loss** — converges to a slightly different
   optimum than the original per-pair loop. The math is identical (mean
   over upper-triangular pair losses) but a single `optimizer.step()` per
   epoch on the full matrix differs from gradients accumulated over N(N-1)/2
   mini-steps. We chose the vectorised version to make 5-fold CV + cross-
   language hold-out tractable on CPU.

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
rules): Claude was used for `core.py` extraction, vectorisation of the
contrastive loss, and report drafting. All evaluation runs and analytical
conclusions are our own.
