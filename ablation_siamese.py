import os
import glob
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(__file__))
from benchmarking import loadModel, fleursLangMap, commandsMap, _extractEncoderEmbedding

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def get_reference_files():
    refs = {}
    for cmd in os.listdir(DATA_DIR):
        cmd_path = os.path.join(DATA_DIR, cmd)
        if cmd == "none" or not os.path.isdir(cmd_path): continue
        wavs = glob.glob(os.path.join(cmd_path, "*.wav"))
        refs[cmd] = wavs
    return refs

def create_dataset(proc, model):
    files = get_reference_files()
    
    # We will use all 5 hi_recordings + 5 ta + 5 te. Total 15 per command.
    # To have a clean train/test split, we use hi 1-3, ta 1-3, te 1-3 for prototyping and training
    # And hi 4-5, ta 4-5, te 4-5 for evaluation.
    
    train_embs = []
    train_labels = []
    
    eval_embs = []
    eval_labels = []
    
    cmd_to_idx = {cmd: i for i, cmd in enumerate(files.keys())}
    idx_to_cmd = {i: cmd for cmd, i in cmd_to_idx.items()}
    
    for cmd, wav_list in tqdm(files.items(), desc="Extracting Encodings"):
        idx = cmd_to_idx[cmd]
        for f in wav_list:
            audio, _ = librosa.load(f, sr=16000, mono=True)
            emb = _extractEncoderEmbedding(audio, proc, model)
            
            # Simple heuristic to split
            fname = os.path.basename(f)
            if "4" in fname or "5" in fname:
                eval_embs.append(emb)
                eval_labels.append(idx)
            else:
                train_embs.append(emb)
                train_labels.append(idx)

    train_embs = torch.tensor(np.array(train_embs), dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    eval_embs = torch.tensor(np.array(eval_embs), dtype=torch.float32)
    eval_labels = torch.tensor(eval_labels, dtype=torch.long)
                
    return train_embs, train_labels, eval_embs, eval_labels, idx_to_cmd

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)

def contrastive_loss(v1, v2, label, margin=0.5):
    # label 1 if same class, 0 if different
    euclidean_distance = F.pairwise_distance(v1, v2, keepdim=True)
    loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                  (1-label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

def train_projection(train_embs, train_labels):
    head = ProjectionHead()
    optimizer = optim.Adam(head.parameters(), lr=1e-3)
    
    head.train()
    epochs = 100
    
    # generate pairs
    n = len(train_embs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss = 0
        pairs = 0
        for i in range(n):
            for j in range(i+1, n):
                is_same = 1.0 if train_labels[i] == train_labels[j] else 0.0
                v1 = head(train_embs[i].unsqueeze(0))
                v2 = head(train_embs[j].unsqueeze(0))
                
                l = contrastive_loss(v1, v2, torch.tensor([[is_same]], dtype=torch.float32))
                loss += l
                pairs += 1
                
        loss = loss / pairs
        loss.backward()
        optimizer.step()
        
    return head

def evaluate_models(proc, model):
    train_embs, train_labels, eval_embs, eval_labels, idx_to_cmd = create_dataset(proc, model)
    
    # 1. Frozen Setup
    print("Evaluating Frozen Siamese...")
    # build prototypes from training set
    prototypes_frozen = {}
    for idx_val in torch.unique(train_labels):
        mask = (train_labels == idx_val)
        proto = train_embs[mask].mean(dim=0)
        proto = F.normalize(proto, p=2, dim=0)
        prototypes_frozen[idx_val.item()] = proto
        
    correct_frozen = 0
    for i in range(len(eval_embs)):
        emb = eval_embs[i]
        best_sim = -1
        best_idx = None
        for p_idx, p_emb in prototypes_frozen.items():
            sim = float(torch.dot(emb, p_emb))
            if sim > best_sim:
                best_sim = sim
                best_idx = p_idx
        
        if best_idx == eval_labels[i].item():
            correct_frozen += 1
            
    print(f"Frozen Whisper encoder + cosine similarity Accuracy: {correct_frozen}/{len(eval_embs)} = {correct_frozen/len(eval_embs):.3f}")
    
    # 2. Learned setup
    print("\nTraining projection head...")
    head = train_projection(train_embs, train_labels)
    head.eval()
    
    # build projected prototypes
    with torch.no_grad():
        proj_train_embs = head(train_embs)
        proj_eval_embs = head(eval_embs)
        
    prototypes_learned = {}
    for idx_val in torch.unique(train_labels):
        mask = (train_labels == idx_val)
        proto = proj_train_embs[mask].mean(dim=0)
        proto = F.normalize(proto, p=2, dim=0)
        prototypes_learned[idx_val.item()] = proto
        
    correct_learned = 0
    for i in range(len(proj_eval_embs)):
        emb = proj_eval_embs[i]
        best_sim = -1
        best_idx = None
        for p_idx, p_emb in prototypes_learned.items():
            sim = float(torch.dot(emb, p_emb))
            if sim > best_sim:
                best_sim = sim
                best_idx = p_idx
                
        if best_idx == eval_labels[i].item():
            correct_learned += 1
            
    print(f"Learned projection head (Contrastive) Accuracy: {correct_learned}/{len(eval_embs)} = {correct_learned/len(eval_embs):.3f}")

if __name__ == "__main__":
    proc, model = loadModel()
    evaluate_models(proc, model)
