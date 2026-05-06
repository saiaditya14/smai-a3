# Progress & Handoff — SMAI A3 / T11.7

Living document so any agent (or the user) can pick up mid-task. Updated as
work proceeds.

Plan file: `C:\Users\Lenovo\.claude\plans\smai-assignment-3-topics-1-pdf-has-the-expressive-rainbow.md`

---

## Plan summary (7 steps)

1. **Refactor** — extract a shared `core.py` and dedupe `app.py` /
   `benchmarking.py` / `ablation_audio.py` / `ablation_siamese.py`.
2. **Local benchmark** — re-run on the full ~174-clip `data/` set with both
   wake-gated and `skip_wake` modes, plus FPR pool from `data/none/` ∪
   `help/no/`.
3. **Ablations** — refresh audio preprocessing numbers; refresh siamese with
   headline split + 5-fold CV + cross-language hold-out. Optional reject
   class trained on `help/no` + augmented copies if siamese FP rate is high.
4. **App.py comparison path** — load saved siamese weights, sidebar toggle
   for transcription / siamese / both, latency badge in result card.
5. **Polish** — `requirements.txt`, `.gitignore`, smoke test
   `streamlit run app.py` (deployment was rolled back).
6. **`ABLATION_STUDY.md`** — refresh stale numbers + add "Production
   architecture comparison" section.
7. **Docs** — `README.md`, `REPORT.md` (6–8 pp), one-slide pitch.

---

## Status

| Step | Status | Notes |
|---|---|---|
| 1 | ✅ done | `core.py` written, all four consumers refactored, syntax + import smoke-checked |
| 2 | ✅ done | Local benchmarks run with both paths (tx and siamese) and both modes (gated and skipWake), plus FPR on negative pool. |
| 3 | ✅ done | Siamese trained with reject class (n=26 hard negatives). Pitch augmentation added to wakeword anchor and command training data. FPR re-evaluated on unseen test pool (n=10). |
| 4 | ✅ done | `app.py` has sidebar toggle (Transcription / Siamese / Both), latency badge, and loads siamese weights from disk. Smoke tested live at localhost:8501. |
| 5 | ✅ done | `requirements.txt` written; `.gitignore` extended; `streamlit run app.py` smoke tested successfully. |
| 6 | ✅ done | `ABLATION_STUDY.md` fully refreshed: stale numbers replaced, FPR table updated with unseen test pool results, Production Architecture Comparison section added. |
| 7 | ✅ done | `README.md`, `REPORT.md` (6–8 pp), and `pitch.md` drafted and synced to final pitch-augmented numbers. |

---

## Files created / modified so far

**Created**
- `core.py` — shared pipeline (commandsMap, loadModel, ingestAudio, wakeword,
  transcribe, matchCommand, ProjectionHead, classifyBySiamese,
  runMultiLangPipeline, runSiamesePipeline). No streamlit dependency.
- `requirements.txt`
- `README.md`
- `docs/PROGRESS.md` (this file)
- `docs/REPORT.md`
- `docs/SUBMISSION.md`
- `docs/ABLATION_STUDY.md`
- `docs/pitch.md`
- `archive/intermediary_results_pre_telugu_tamil.csv` (moved from root)
- `help/no_test/` — 10-clip unseen hard-negative test pool

**Modified**
- `app.py` — imports from `core`, adds sidebar mode toggle (Transcription /
  Siamese / Both), latency badge in result card. Wraps `loadModel`,
  `buildWakeAnchor`, `loadSiameseAssets` in `@st.cache_resource`.
- `benchmarking.py` — imports from `core`. New CSV
  `results/benchmark_results_local.csv` with per-clip transcription + siamese rows
  (predicted/success/ms each), plus wake info. Three modes: `--mode local`
  (with `--skipWake`, `--noSiamese` flags), `--mode negatives` (FPR on
  `data/none` + `help/no` + `help/no_test`), `--mode fleurs` (unchanged).
- `ablation_audio.py` — imports from `core`, otherwise unchanged behaviour.
- `ablation_siamese.py` — imports from `core`. Adds three eval regimes
  (headline 1-3/4-5, 5-fold CV, cross-language hold-out). Disk cache for
  embeddings (`models/_siamese_embeddings.npz`). `--reject` flag enables training a
  reject class from `help/no` + 4 augmented copies (3 SNR levels +
  time-shift) per clip. Command clips are pitch-augmented at -3/+3/+6
  semitones for training. Saves `models/siamese_head.pt` +
  `models/siamese_prototypes.npz`
  (including `__reject__` prototype) unless `--no-save`.
- `core.py` — `buildWakeAnchor` now generates +3, +6, −3 semitone
  pitch-shifted copies of each `help/yes` clip before averaging the anchor,
  improving cross-speaker/gender wakeword robustness. `WAKE_SIM_THRESHOLD`
  adjusted to 0.910. `classifyBySiamese` now returns `None` when best match
  is `__reject__`. Siamese assets now live in `models/`.
- `.gitignore` — appended project-specific ignores.
- Repository cleanup — docs moved to `docs/`, screenshots to
  `docs/screenshots/`, CSVs/logs to `results/`, UI images to `assets/`, and
  siamese weights/cache to `models/`.

---

## Running jobs (background)

| Task ID | Command | Status |
|---|---|---|
| `bh01wz394` | `python ablation_siamese.py --kfold 5 --epochs 100` | ✅ done |
| `brmp6vgzx` | `python benchmarking.py --mode negatives` | ✅ done |
| `bwbamai20` | `python ablation_audio.py` | ✅ done |
| `b2eto0n3t` | `python ablation_siamese.py --reject --kfold 5 --epochs 100` | ✅ done |
| `902ea0aa` | `python ablation_siamese.py --reject` + `benchmarking.py --mode negatives` (26 train / 10 test negatives) | ✅ done |

## Verified results so far

### Siamese ablation (without reject class)

```
HEADLINE SPLIT  (train idx 1-3, eval idx 4-5)
  Frozen encoder + cosine prototype:  42.37%
  Learned projection head:            88.14%

K-FOLD CROSS-VALIDATION  (k=5)
  fold 1: 83.33%   fold 2: 93.33%   fold 3: 90.00%
  fold 4: 96.67%   fold 5: 96.55%
  Mean ± std: 91.98% ± 4.97%

CROSS-LANGUAGE HOLD-OUT  ← documented limitation; not a bug
  held-out hi: 8.00%
  held-out ta: 10.20%
  held-out te: 26.00%
```

### Siamese ablation (with reject class + command pitch augmentation — final saved weights)

```
HEADLINE SPLIT:  93.22%
K-FOLD CV:       91.24% ± 6.25%
CROSS-LANGUAGE:  held-out hi 0.00% | ta 10.20% | te 18.00%
```

Adding the reject class plus command pitch augmentation improved headline
accuracy (88.14% → 93.22%) while dropping siamese CMD FPR on hard negatives
from 80% → 0%.

### Negatives FPR (final — reject class + wakeword pitch augmentation)

```
[data_none]    n=100 | wake_fp=3.0%  | tx_cmd_fp=2.0%  | siamese_cmd_fp=3.0%
[help_no]      n=26  | wake_fp=69.2% | tx_cmd_fp=3.8%  | siamese_cmd_fp=0.0%
[help_no_test] n=10  | wake_fp=50.0% | tx_cmd_fp=20.0% | siamese_cmd_fp=0.0%
```

Wakeword gate is intentionally permissive (encoder cosine sim, no learned
discriminator). Siamese model acts as second filter — 0% CMD FPR on both
seen and unseen hard negatives after reject-class retraining.

### Key findings

- **Path B beats Path A on accuracy** (93.22% vs 70.47%) and latency (~270ms vs ~1623ms) when all training languages are present.
- **Path B is the submitted default** for the fixed T11.7 command setting.
- **Path A generalises better** — transcription handles new command strings/languages with dictionary edits; siamese needs retraining.
- **Wakeword gate** improved with pitch augmentation across +3/+6/−3 semitones; now works for female speakers.
- **Vectorised training** gave ~100× speedup (1.3s vs ~5 min per 10 epochs).

---

## Resolved handoff items

| # | Task | Priority |
|---|---|---|
| A | ✅ done — command training data pitch-augmented in `ablation_siamese.py`, embeddings rebuilt, head/prototypes retrained and saved. | 🔴 high |
| B | ✅ done — `evaluation/` and empty `hindi`/`tamil`/`telugu` root stubs deleted. | 🟡 medium |
| C | ✅ done — Siamese-only mode now shows "No transcript (Siamese mode)" after wakeword success. | 🟡 medium |
| D | ✅ done — `REPORT.md` drafted and updated with final pitch-augmented metrics. | 🔴 high |
| E | ✅ done — `pitch.md` drafted and updated with final Path A/Path B numbers. | 🟡 medium |

---

## Final submission checklist

| Item | Status | Owner |
|---|---|---|
| Add team names, roll numbers, emails to report | ✅ done | Codex |
| Add GitHub repository link | ✅ done (`https://github.com/saiaditya14/smai-a3`) | Codex |
| Add deployed prototype link | ✅ done (`https://engja9hyuz4sjstbxdka2e.streamlit.app/`) | Codex |
| Add screenshots from `docs/screenshots/` | ✅ done in `docs/REPORT.md` and `docs/SUBMISSION.md` | Codex |
| Explain Path A vs Path B in plain language | ✅ done | Codex |
| Document live wakeword limitations instead of retraining/redeploying | ✅ done | Codex |
| Refactor repo into clean submission layout | ✅ done | Codex |
| Generate final single consolidated PDF | ⬜ pending | User/Codex after choosing export method |
| Paste submission textbox details | ⬜ pending | User |

### Submission textbox draft

```
Project: Indic Speech Command Recognizer (T11.7)

Team:
Aryan Maskara, 2023111004, aryan.maskara@research.iiit.ac.in
Sai Ramanathan, 2023101096, sai.ramanathan@students.iiit.ac.in
Neha Murthy, 2023115018, neha.murthy@research.iiit.ac.in
Yajat Lakhanpal, 2023111015, yajat.lakhanpal@research.iiit.ac.in
Prasoon Dev, 2023111014, prasoon.dev@research.iiit.ac.in
Aanchal Mundhada, 2023112016, aanchal.mundhada@research.iiit.ac.in

GitHub repository:
https://github.com/saiaditya14/smai-a3

Deployed working prototype / video of working:
https://engja9hyuz4sjstbxdka2e.streamlit.app/
```

---

## Verification snippets

```bash
# Confirm core imports cleanly
python -c "import core; print(len([k for k in dir(core) if not k.startswith('_')]), 'symbols')"

# Confirm syntax of all modified files
python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in ['app.py','benchmarking.py','ablation_audio.py','ablation_siamese.py','core.py']]; print('OK')"

# Sanity check siamese assets exist before running app
python -c "import os; print('head:', os.path.exists('models/siamese_head.pt')); print('protos:', os.path.exists('models/siamese_prototypes.npz'))"
```

---

## Notes for whoever picks this up

- `models/siamese_head.pt` and `models/siamese_prototypes.npz` are persisted to disk and
  include the `__reject__` prototype. The app loads them at startup via
  `loadSiameseAssetsCached()`. If you retrain (e.g. after pitch augmentation),
  just re-run `python ablation_siamese.py --reject` and restart Streamlit.
- Embedding cache lives at `models/_siamese_embeddings.npz`. Delete it before
  retraining if you change the data (new clips, augmentation flags etc.).
- `results/benchmark_results_local.csv` and
  `results/benchmark_results_negatives.csv` are overwritten each run, not
  appended.
- Wakeword threshold is `WAKE_SIM_THRESHOLD = 0.910` in `core.py`. The gate
  is intentionally permissive — the Siamese reject class is the real filter.
- No open implementation/docs items remain from the prior handoff.
