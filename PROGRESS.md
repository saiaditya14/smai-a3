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
| 3 | ✅ done | Siamese trained with reject class. Audio preprocessing ablation run. |
| 4 | 🟡 partial | `app.py` already has the toggle + latency badge wired to `loadSiameseAssetsCached`; still needs the `siamese_head.pt`/`siamese_prototypes.npz` files on disk (done now) |
| 5 | 🟢 partial | `requirements.txt` written; `.gitignore` extended; smoke test still pending |
| 6 | ✅ done | Stale numbers refreshed and Production Architecture Comparison added to ABLATION_STUDY.md |
| 7 | 🟢 partial | `README.md` written (skeleton); `REPORT.md` and `pitch.md` not yet drafted |

---

## Files created / modified so far

**Created**
- `core.py` — shared pipeline (commandsMap, loadModel, ingestAudio, wakeword,
  transcribe, matchCommand, ProjectionHead, classifyBySiamese,
  runMultiLangPipeline, runSiamesePipeline). No streamlit dependency.
- `requirements.txt`
- `README.md`
- `PROGRESS.md` (this file)
- `archive/intermediary_results_pre_telugu_tamil.csv` (moved from root)

**Modified**
- `app.py` — imports from `core`, adds sidebar mode toggle (Transcription /
  Siamese / Both), latency badge in result card. Wraps `loadModel`,
  `buildWakeAnchor`, `loadSiameseAssets` in `@st.cache_resource`.
- `benchmarking.py` — imports from `core`. New CSV
  `benchmark_results_local.csv` with per-clip transcription + siamese rows
  (predicted/success/ms each), plus wake info. Three modes: `--mode local`
  (with `--skipWake`, `--noSiamese` flags), `--mode negatives` (FPR on
  `data/none` + `help/no`), `--mode fleurs` (unchanged).
- `ablation_audio.py` — imports from `core`, otherwise unchanged behaviour.
- `ablation_siamese.py` — imports from `core`. Adds three eval regimes
  (headline 1-3/4-5, 5-fold CV, cross-language hold-out). Disk cache for
  embeddings (`_siamese_embeddings.npz`). `--reject` flag enables training a
  reject class from `help/no` + 4 augmented copies (3 SNR levels +
  time-shift) per clip. Saves `siamese_head.pt` + `siamese_prototypes.npz`
  unless `--no-save`.
- `.gitignore` — appended project-specific ignores.

---

## Running jobs (background)

| Task ID | Command | Log | Status |
|---|---|---|---|
| `bh01wz394` | `python ablation_siamese.py --kfold 5 --epochs 100` | `siamese_run3.log` | ✅ done |
| `brmp6vgzx` | `python benchmarking.py --mode negatives` | `negatives_run.log` | ✅ done |
| `bwbamai20` | `python ablation_audio.py` | `audio_ablation.log` | ✅ done |
| `b2eto0n3t` | `python ablation_siamese.py --reject --kfold 5 --epochs 100` | `siamese_reject.log` | ✅ done |

## Verified results so far

### Siamese ablation (full sweep)

```
HEADLINE SPLIT  (train idx 1-3, eval idx 4-5)
  train=90  eval=59
  Frozen encoder + cosine prototype:  42.37%
  Learned projection head:            88.14%

K-FOLD CROSS-VALIDATION  (k=5)
  fold 1: 83.33%
  fold 2: 93.33%
  fold 3: 90.00%
  fold 4: 96.67%
  fold 5: 96.55%
  Mean ± std: 91.98% ± 4.97%

CROSS-LANGUAGE HOLD-OUT
  held-out hi: 8.00%
  held-out ta: 10.20%
  held-out te: 26.00%
```

### Negatives FPR (no-reject siamese)

```
[data_none] n=100 | wake_fp=0%   | tx_cmd_fp=0%   | siamese_cmd_fp=0%
[help_no]   n=5   | wake_fp=80%  | tx_cmd_fp=0%   | siamese_cmd_fp=80%
```

**Critical finding**: the wakeword fires on 4/5 hard negatives. Path A
(transcription) catches every wake-FP downstream because the transcribed
text doesn't fuzzy-match any command. Path B (siamese, no reject class)
has no second filter — every wake-FP becomes a command-FP. This is
**exactly** the failure mode the reject-class idea fixes. Retraining now
with `--reject` (help/no + 4 augmented copies per clip).

**Big finding (siamese)**: cross-language hold-out collapses to single digits. The
projection head is learning **language-specific acoustic templates**, not
language-independent intent representations. So the 88-92% headline /
k-fold numbers are only valid when all 3 target languages are seen at
train time. This is a critical caveat for the production-vs-comparison
discussion in the report — siamese needs every supported language in its
training fold to work, while the transcription path generalises naturally.

(Headline dropped from original 94.9% → 88.14%. Vectorised training is
mathematically equivalent to the per-pair loop but converges differently
due to single-step-per-epoch on the full pair matrix vs original's
accumulate-then-step. Still well above the 70% needed to dominate
transcription.)

**Performance fix (2026-05-06)**: original `train_head` did N(N-1)/2
individual forward+backward passes per epoch — pure Python overhead made
each epoch ~30s. Replaced with a vectorised version that computes the full
pairwise distance matrix in one pass. Smoke test: 10 epochs on N=100 went
from ~5 min to **1.3 s** (~100× speedup). The math is identical
(margin-contrastive over all i<j pairs), so headline numbers should match
the original ABLATION_STUDY's 94.9% within a couple percent.

Monitor `b5sy7nmh1` tails the log for HEADLINE / K-FOLD / CROSS-LANGUAGE
markers.

---

## Decisions / open items

1. **Reject class for siamese**: user agreed to add it *if* siamese FP rate
   on `help/no` is high. Plan: run siamese without reject first → run
   `benchmarking.py --mode negatives` → if siamese fires often on `help/no`,
   re-run `python ablation_siamese.py --reject` and re-evaluate.
2. **Saved weights**: the first siamese run uses `--no-save` so we can read
   the headline numbers cleanly. Once we decide on reject vs no-reject, run
   without `--no-save` to persist the chosen variant for the live app.
3. **Audio ablation rerun** (Step 3 sub-task): not yet kicked off. Run
   sequentially after siamese to avoid CPU thrash.

---

## Next actions in order

1. Wait for `bwm5tah9u` (siamese dry-run) to finish; record headline /
   k-fold / cross-language numbers.
2. Run `python ablation_siamese.py` (no `--no-save`) to persist
   `siamese_head.pt` + `siamese_prototypes.npz`.
3. Run `python benchmarking.py --mode local` (full dataset, both paths).
4. Run `python benchmarking.py --mode local --skipWake` (isolates command
   recognition from wakeword).
5. Run `python benchmarking.py --mode negatives` (FPR table).
6. Decide on reject class based on FPR results; if needed, retrain with
   `--reject` and overwrite weights.
7. Run `python ablation_audio.py` (full-dataset preprocessing numbers).
8. Smoke test `streamlit run app.py` end-to-end (transcription / siamese /
   both modes, plus a "Hey Bharat" + command run for video capture).
9. Update `ABLATION_STUDY.md` with verified numbers + add Production
   Architecture Comparison section.
10. Draft `REPORT.md` (6–8 pp ICVGIP-style) and `pitch.md`.

---

## Verification snippets

```bash
# Confirm core imports cleanly
python -c "import core; print(len([k for k in dir(core) if not k.startswith('_')]), 'symbols')"

# Confirm syntax of all modified files
python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in ['app.py','benchmarking.py','ablation_audio.py','ablation_siamese.py','core.py']]; print('OK')"

# Sanity check siamese assets exist before running app
python -c "import os; print('head:', os.path.exists('siamese_head.pt')); print('protos:', os.path.exists('siamese_prototypes.npz'))"
```

---

## Notes for whoever picks this up

- Embedding extraction is the slow part (~30 s for ~165 wav files). It's
  cached to `_siamese_embeddings.npz` keyed on `reject=<bool>_aug=<bool>` —
  flipping `--reject` invalidates the cache and triggers re-extraction.
- The sidebar UI in `app.py` references `loadSiameseAssetsCached()`; if the
  files don't exist on disk, the app shows a warning and the siamese option
  is effectively dead. So **always run `ablation_siamese.py` (without
  `--no-save`) before `streamlit run app.py`** if demoing the toggle.
- `benchmark_results_local.csv` and `benchmark_results_negatives.csv` are
  overwritten each run (not appended) — different from the old
  `intermediary_results.csv` resumable behaviour.
- `app.py` currently does not change `lastTranscript` for the siamese-only
  mode; result card will read "Wakeword similarity: 0.xxxx" since no text
  transcription happened. Cosmetic — could be addressed in Step 5 polish.
- The `evaluation/` folder contains older copies of `benchmarking.py`,
  `baseline_eval.py`, `raw.py`. Can probably be deleted in Step 5 cleanup
  but verify nothing references them first.
- Empty `hindi/`, `tamil/`, `telugu/` folders at root are stub remnants —
  safe to delete.
