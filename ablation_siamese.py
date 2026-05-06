"""
Siamese projection-head ablation.

Three evaluation regimes are reported:
  1. Headline split: train on recordings 1-3 per language, eval on 4-5
     (this matches the original ABLATION_STUDY result; spans multiple
     speakers/environments per language).
  2. K-fold cross-validation (5-fold): mean ± std for confidence band.
  3. Cross-language hold-out: train on hi+ta, eval on te (and rotations).

Trained head + per-command prototypes are persisted under models/ so app.py
can load them without retraining.

Optional reject class (--reject) trains an extra prototype from
help/no/*.wav plus noise-augmented copies, so the head can return None
on inputs that don't acoustically resemble any command.
"""

import argparse
import glob
import os
import random
import re
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import core
from core import (
    loadModel, _extractEncoderEmbedding, ProjectionHead,
    targetRate, SIAMESE_HEAD_PATH, SIAMESE_PROTOS_PATH,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HELP_NO_DIR = os.path.join(os.path.dirname(__file__), "help", "no")

REJECT_LABEL = "__reject__"


# ---------------------------------------------------------------------------
# Filename parsing: extract language and recording index
# ---------------------------------------------------------------------------

LANG_PATTERN = re.compile(r"^(hi|ta|te)[_]", re.IGNORECASE)
INDEX_PATTERN = re.compile(r"(\d+)")

def parse_filename(fname):
    """Return (language_code, recording_index) or (None, None) if unparseable."""
    m = LANG_PATTERN.match(fname)
    lang = m.group(1).lower() if m else None
    nums = INDEX_PATTERN.findall(fname)
    idx = int(nums[-1]) if nums else None
    return lang, idx


# ---------------------------------------------------------------------------
# Embedding extraction with caching to disk
# ---------------------------------------------------------------------------

EMBEDDING_CACHE = os.path.join(os.path.dirname(__file__), "models", "_siamese_embeddings.npz")

PITCH_SEMITONES = (-3, 3, 6)

def extract_all_embeddings(proc, model, include_reject=False, augment=True,
                           pitch_augment=True):
    """
    Iterate over data/<cmd>/*.wav and (optionally) help/no/*.wav, extract
    encoder embeddings. Cache to disk so repeated runs are fast.

    With pitch_augment=True, each command clip generates 3 additional
    pitch-shifted variants (-3, +3, +6 semitones) marked with idx=None and
    is_aug=True. Augmented copies inherit the source clip's language so
    cross-language splits stay honest, but they are routed to train and
    held out of every eval set (see *_split helpers below).

    Returns (embeddings: (N, 384) float32, labels, langs, indices, is_aug).
    """
    cache_key = f"reject={include_reject}_aug={augment}_pitch={pitch_augment}"
    if os.path.exists(EMBEDDING_CACHE):
        data = np.load(EMBEDDING_CACHE, allow_pickle=True)
        if str(data.get("cache_key", "")) == cache_key:
            print(f"Loaded cached embeddings: {data['embs'].shape[0]} clips")
            is_aug = list(data["is_aug"]) if "is_aug" in data.files else [False] * len(data["labels"])
            return (data["embs"], list(data["labels"]),
                    list(data["langs"]), list(data["indices"]), is_aug)

    embs, labels, langs, indices, is_aug = [], [], [], [], []

    cmds = [d for d in sorted(os.listdir(DATA_DIR))
            if os.path.isdir(os.path.join(DATA_DIR, d)) and d != "none"]
    for cmd in tqdm(cmds, desc="Extracting command embeddings"):
        cmd_path = os.path.join(DATA_DIR, cmd)
        for fname in sorted(os.listdir(cmd_path)):
            if not fname.lower().endswith(".wav"):
                continue
            audio, _ = librosa.load(os.path.join(cmd_path, fname),
                                    sr=targetRate, mono=True)
            audio = audio.astype(np.float32)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
            lang, idx = parse_filename(fname)

            # Original clip
            emb = _extractEncoderEmbedding(audio, proc, model)
            embs.append(emb); labels.append(cmd); langs.append(lang)
            indices.append(idx); is_aug.append(False)

            # Pitch-shifted copies — train-only, idx=None marks as augmented
            if pitch_augment:
                for n_steps in PITCH_SEMITONES:
                    shifted = librosa.effects.pitch_shift(
                        audio, sr=targetRate, n_steps=n_steps,
                    ).astype(np.float32)
                    aug_emb = _extractEncoderEmbedding(shifted, proc, model)
                    embs.append(aug_emb); labels.append(cmd); langs.append(lang)
                    indices.append(None); is_aug.append(True)

    if include_reject and os.path.isdir(HELP_NO_DIR):
        no_files = sorted(glob.glob(os.path.join(HELP_NO_DIR, "*.wav")))
        for fname in tqdm(no_files, desc="Extracting reject embeddings"):
            audio, _ = librosa.load(fname, sr=targetRate, mono=True)
            audio = audio.astype(np.float32)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

            variants = [audio]
            if augment:
                rng = np.random.default_rng(42 + len(embs))
                for snr_db in (10, 5, 0):
                    noise = rng.standard_normal(len(audio)).astype(np.float32)
                    sig_p = (audio ** 2).mean() + 1e-9
                    noi_p = (noise ** 2).mean() + 1e-9
                    scale = np.sqrt(sig_p / (noi_p * (10 ** (snr_db / 10))))
                    variants.append(np.clip(audio + scale * noise, -1, 1))
                shift = int(0.1 * targetRate)
                variants.append(np.concatenate([np.zeros(shift, dtype=np.float32),
                                                audio])[:len(audio)])

            for v in variants:
                emb = _extractEncoderEmbedding(v, proc, model)
                embs.append(emb); labels.append(REJECT_LABEL); langs.append(None)
                indices.append(None); is_aug.append(False)

    embs = np.stack(embs).astype(np.float32)
    np.savez(EMBEDDING_CACHE, embs=embs, labels=np.array(labels),
             langs=np.array(langs, dtype=object),
             indices=np.array(indices, dtype=object),
             is_aug=np.array(is_aug, dtype=bool),
             cache_key=cache_key)
    print(f"Cached {embs.shape[0]} embeddings to {EMBEDDING_CACHE}")
    return embs, labels, langs, indices, is_aug


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_head(train_embs, train_labels, epochs=100, seed=0, desc="train",
               margin=0.5):
    """
    Train ProjectionHead with contrastive loss vectorised across all pairs.

    Per epoch:
      proj = head(train_embs)                       # (N, D)
      D    = pairwise_distance(proj_i, proj_j)      # (N, N)
      same = label[i] == label[j]                   # (N, N) {0,1}
      loss = mean( same * D^2 + (1-same) * relu(margin - D)^2 )

    Eliminates the Python double-loop (~100x faster on CPU).
    """
    torch.manual_seed(seed)
    head = ProjectionHead()
    optimizer = optim.Adam(head.parameters(), lr=1e-3)
    head.train()

    label_to_idx = {lab: i for i, lab in enumerate(sorted(set(train_labels)))}
    label_ids = torch.tensor([label_to_idx[l] for l in train_labels],
                             dtype=torch.long)
    same = (label_ids.unsqueeze(0) == label_ids.unsqueeze(1)).float()
    n = len(train_labels)
    # Mask the diagonal and lower triangle (i < j only) so each pair counts once
    triu_mask = torch.triu(torch.ones(n, n), diagonal=1).bool()

    for _ in tqdm(range(epochs), desc=desc, leave=False):
        optimizer.zero_grad()
        proj = head(train_embs)                   # (N, D)
        # Pairwise euclidean distances; small eps to avoid sqrt(0) gradient issues
        diff = proj.unsqueeze(0) - proj.unsqueeze(1)
        dist = (diff.pow(2).sum(-1) + 1e-9).sqrt()  # (N, N)
        pos_loss = same * dist.pow(2)
        neg_loss = (1 - same) * F.relu(margin - dist).pow(2)
        loss_mat = pos_loss + neg_loss
        loss = loss_mat[triu_mask].mean()
        loss.backward()
        optimizer.step()
    head.eval()
    return head


def build_prototypes(embs, labels, label_set):
    """Mean-pool projected embeddings per class, L2-normalise."""
    protos = {}
    for lab in label_set:
        mask = [l == lab for l in labels]
        sel = embs[mask]
        if len(sel) == 0:
            continue
        proto = sel.mean(dim=0)
        protos[lab] = F.normalize(proto, p=2, dim=0)
    return protos


def evaluate_split(head, train_embs_t, train_labels, eval_embs_t, eval_labels,
                   reject_threshold=None):
    """Project embeddings, classify by nearest prototype, return accuracy."""
    label_set = sorted(set(train_labels))
    with torch.no_grad():
        proj_train = head(train_embs_t)
        proj_eval = head(eval_embs_t)

    protos = build_prototypes(proj_train, train_labels, label_set)
    correct = 0
    for i in range(len(proj_eval)):
        emb = proj_eval[i]
        best_lab, best_sim = None, -1.0
        for lab, p in protos.items():
            sim = float(torch.dot(emb, p))
            if sim > best_sim:
                best_sim = sim
                best_lab = lab
        if reject_threshold is not None and best_sim < reject_threshold:
            best_lab = REJECT_LABEL
        if best_lab == eval_labels[i]:
            correct += 1
    return correct / len(proj_eval) if len(proj_eval) > 0 else 0.0, protos


def evaluate_frozen_split(train_embs_t, train_labels, eval_embs_t, eval_labels):
    """Same as evaluate_split but using raw encoder embeddings (no projection)."""
    label_set = sorted(set(train_labels))
    protos = {}
    for lab in label_set:
        mask = [l == lab for l in train_labels]
        sel = train_embs_t[mask]
        if len(sel) == 0:
            continue
        protos[lab] = F.normalize(sel.mean(dim=0), p=2, dim=0)

    correct = 0
    for i in range(len(eval_embs_t)):
        emb = F.normalize(eval_embs_t[i], p=2, dim=0)
        best_lab, best_sim = None, -1.0
        for lab, p in protos.items():
            sim = float(torch.dot(emb, p))
            if sim > best_sim:
                best_sim = sim
                best_lab = lab
        if best_lab == eval_labels[i]:
            correct += 1
    return correct / len(eval_embs_t) if len(eval_embs_t) > 0 else 0.0


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def headline_split(labels, indices, is_aug=None):
    """Train on idx 1-3, eval on 4-5. Reject + augmented samples → train."""
    if is_aug is None:
        is_aug = [False] * len(labels)
    train_mask, eval_mask = [], []
    for lab, idx, aug in zip(labels, indices, is_aug):
        if lab == REJECT_LABEL or aug or idx is None:
            train_mask.append(True); eval_mask.append(False)
        elif idx in (4, 5):
            train_mask.append(False); eval_mask.append(True)
        else:
            train_mask.append(True); eval_mask.append(False)
    return np.array(train_mask), np.array(eval_mask)


def kfold_splits(labels, indices, is_aug=None, k=5, seed=0):
    """Stratified k-fold. Reject + augmented always train (excluded from folds)."""
    if is_aug is None:
        is_aug = [False] * len(labels)
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    by_label = {}
    for i, lab in enumerate(labels):
        if lab == REJECT_LABEL or is_aug[i]:
            continue
        by_label.setdefault(lab, []).append(i)
    for lab, idx_list in by_label.items():
        rng.shuffle(idx_list)
        for i, sample_idx in enumerate(idx_list):
            folds[i % k].append(sample_idx)
    n = len(labels)
    for fold_id in range(k):
        eval_idx = set(folds[fold_id])
        train_mask = np.array([(i not in eval_idx) for i in range(n)])
        eval_mask = np.array([(i in eval_idx) for i in range(n)])
        yield train_mask, eval_mask


def cross_language_splits(labels, langs, is_aug=None):
    """
    Train on two langs, eval on the third.
    - Reject samples → always train (no language bias).
    - Augmented samples of held-out language → excluded from train AND eval
      (otherwise we leak held-out language into training).
    - Augmented samples of seen languages → train (data diversity).
    - All eval samples are non-augmented (real clips only).
    """
    if is_aug is None:
        is_aug = [False] * len(labels)
    all_langs = ["hi", "ta", "te"]
    n = len(labels)
    for held_out in all_langs:
        train_mask = []
        eval_mask = []
        for i in range(n):
            lab, lang, aug = labels[i], langs[i], is_aug[i]
            if lab == REJECT_LABEL:
                train_mask.append(True); eval_mask.append(False); continue
            if aug:
                # Augmented clips: only included in train if their language is seen
                train_mask.append(lang != held_out)
                eval_mask.append(False)
                continue
            if lang == held_out:
                train_mask.append(False); eval_mask.append(True)
            else:
                train_mask.append(True); eval_mask.append(False)
        yield held_out, np.array(train_mask), np.array(eval_mask)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(include_reject=False, save=True, kfold=5, epochs=100, pitch_augment=True):
    proc, model = loadModel()
    embs_np, labels, langs, indices, is_aug = extract_all_embeddings(
        proc, model, include_reject=include_reject, pitch_augment=pitch_augment,
    )
    embs_t = torch.tensor(embs_np, dtype=torch.float32)
    n_aug = sum(is_aug)
    print(f"Total clips: {len(labels)} ({n_aug} pitch-augmented)")

    print()
    print("=" * 70)
    print("  HEADLINE SPLIT  (train idx 1-3, eval idx 4-5)")
    print("=" * 70)
    train_mask, eval_mask = headline_split(labels, indices, is_aug)
    train_embs_t = embs_t[train_mask]
    eval_embs_t = embs_t[eval_mask]
    train_lab = [l for l, m in zip(labels, train_mask) if m]
    eval_lab = [l for l, m in zip(labels, eval_mask) if m]
    print(f"  train={len(train_lab)}  eval={len(eval_lab)}")

    frozen_acc = evaluate_frozen_split(train_embs_t, train_lab, eval_embs_t, eval_lab)
    print(f"  Frozen encoder + cosine prototype:  {frozen_acc*100:.2f}%")

    head = train_head(train_embs_t, train_lab, epochs=epochs)
    learned_acc, headline_protos = evaluate_split(
        head, train_embs_t, train_lab, eval_embs_t, eval_lab
    )
    print(f"  Learned projection head:            {learned_acc*100:.2f}%")

    if save:
        os.makedirs(os.path.dirname(SIAMESE_HEAD_PATH), exist_ok=True)
        torch.save(head.state_dict(), SIAMESE_HEAD_PATH)
        # Index mapping for app.py: filter out reject from the saved prototypes
        # so the live app uses command prototypes only (reject is enforced via
        # the SIAMESE_SIM_THRESHOLD instead).
        idx_to_action = {}
        protos_arr = []
        idx_arr = []
        actions_arr = []
        i = 0
        for lab, p in headline_protos.items():
            idx_to_action[i] = lab.replace("_", " ") if lab != REJECT_LABEL else REJECT_LABEL
            idx_arr.append(i)
            actions_arr.append(lab.replace("_", " ") if lab != REJECT_LABEL else REJECT_LABEL)
            protos_arr.append(p.numpy())
            i += 1
        np.savez(
            SIAMESE_PROTOS_PATH,
            idx=np.array(idx_arr, dtype=np.int64),
            protos=np.stack(protos_arr).astype(np.float32),
            actions=np.array(actions_arr, dtype=object),
        )
        print(f"  Saved head -> {SIAMESE_HEAD_PATH}")
        print(f"  Saved prototypes ({len(idx_arr)} classes) -> {SIAMESE_PROTOS_PATH}")

    print()
    print("=" * 70)
    print(f"  K-FOLD CROSS-VALIDATION  (k={kfold})")
    print("=" * 70)
    accs = []
    for fold_id, (tr_mask, ev_mask) in enumerate(kfold_splits(labels, indices, is_aug, k=kfold)):
        tr_embs = embs_t[tr_mask]; ev_embs = embs_t[ev_mask]
        tr_lab = [l for l, m in zip(labels, tr_mask) if m]
        ev_lab = [l for l, m in zip(labels, ev_mask) if m]
        h = train_head(tr_embs, tr_lab, epochs=epochs, seed=fold_id)
        acc, _ = evaluate_split(h, tr_embs, tr_lab, ev_embs, ev_lab)
        print(f"  fold {fold_id+1}: {acc*100:.2f}%   (train={len(tr_lab)}, eval={len(ev_lab)})")
        accs.append(acc)
    print(f"  Mean ± std: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")

    print()
    print("=" * 70)
    print("  CROSS-LANGUAGE HOLD-OUT  (train hi+ta, eval te; etc.)")
    print("=" * 70)
    for held, tr_mask, ev_mask in cross_language_splits(labels, langs, is_aug):
        tr_embs = embs_t[tr_mask]; ev_embs = embs_t[ev_mask]
        tr_lab = [l for l, m in zip(labels, tr_mask) if m]
        ev_lab = [l for l, m in zip(labels, ev_mask) if m]
        if len(ev_lab) == 0:
            continue
        h = train_head(tr_embs, tr_lab, epochs=epochs, seed=42)
        acc, _ = evaluate_split(h, tr_embs, tr_lab, ev_embs, ev_lab)
        print(f"  held-out {held}: {acc*100:.2f}%   (train={len(tr_lab)}, eval={len(ev_lab)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reject", action="store_true",
                        help="Include a reject class trained on help/no + augmented copies.")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the trained head/prototypes to disk.")
    parser.add_argument("--no-pitch-augment", action="store_true",
                        help="Disable pitch-shift augmentation of command clips "
                             "(default: enabled, ±3/+6 semitones).")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    run(include_reject=args.reject, save=not args.no_save,
        kfold=args.kfold, epochs=args.epochs,
        pitch_augment=not args.no_pitch_augment)
