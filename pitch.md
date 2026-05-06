# Indic Speech Command Recognizer — One-slide pitch

> *"Hey Bharat... batti jalao."*
> A 39 M-parameter Whisper-tiny on a laptop CPU, running smart-home commands
> in **Hindi, Tamil & Telugu**.

---

### What it does
Wakeword (English *"Hey Bharat"*) → 11 smart-home commands in 3 Indic
languages → real-time UI feedback. No GPU, no paid inference.

### Why it works
- **Encoder-embedding wakeword** — bypass Whisper's autoregressive decoder
  (which hallucinates on Indic) and use a 1.5 s sliding window of cosine
  similarity vs an anchor.
- **Forced-language transcription + sliding-window fuzzy match** — turns
  Whisper's high WER into high *intent* accuracy.
- **Contrastive projection head** over Whisper encoder embeddings — direct
  acoustic intent classification, single forward pass.

### Headline numbers (CPU only)

| Approach | Intent accuracy |
|---|---|
| Raw whisper-tiny + exact match | 3.7 % |
| Path A: transcription + fuzzy match | **TBD %** |
| Path B: siamese projection head (5-fold CV) | **91.98 % ± 4.97 %** |

**Cross-language hold-out** (train hi+ta, test te) collapses Path B to
8–26 % — the projection head learns language-specific acoustic templates,
not language-independent intent. Lesson: ship Path A as default, Path B
as a faster alternative when all target languages are seen at train time.

### Built with
Whisper-tiny · FLEURS · Streamlit · PyTorch (contrastive loss, vectorised)

---

*SMAI Assignment 3 · Theme T11.7 · IIIT-H · 2025–26*
