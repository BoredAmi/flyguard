"""A small trained CNN baseline (the project's task 4), second comparison point
alongside Lee's tau. Unlike the connectome-derived detector and unlike tau,
this one is deliberately a **learned** model -- the point of the comparison
is "what does training buy you over a zero-parameter geometric cue or an
anatomical circuit," not to build another zero-training detector. Per the
working agreement, learned parameters only ever live in this baseline path,
never in the connectome path.

Input: a pair of consecutive grayscale frames (2-channel, [0,1]) from
`flyguard.mujoco_world` -- the same rendered frames the tau baseline and
(eventually) the encoder consume, so all three detectors see the same raw
input. Output: binary looming-vs-translation logits per frame pair.
Deliberately tiny (3 conv layers, global average pool, one linear layer) --
this is a baseline, not a research model.

See `flyguard.validate_baseline_cnn` for the training/evaluation script that
generates data, trains a model, and reports accuracy, the shortcut check,
and inference latency.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONDITION_LABEL = {"looming": 1, "translation": 0}


class TinyLoomNet(nn.Module):
    """~19k parameters. Two grayscale frames in, 2-class logits out."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=(2, 3))  # global average pool
        return self.fc(x)


def frame_pairs_from_dataset(npz_path: Path):
    """Load a `mujoco_world.generate_dataset` `.npz` and return consecutive
    grayscale frame pairs.

    Returns (X [N, 2, H, W] float32 in [0,1], y [N] int64, trial_id [N] int64).
    """
    z = np.load(npz_path)
    frames, condition, trial_id = z["frames"], z["condition"], z["trial_id"]
    gray = frames.astype(np.float32).mean(axis=-1) / 255.0  # [n_trials, n_frames, H, W]

    X_list, y_list, tid_list = [], [], []
    n_trials, n_frames = gray.shape[0], gray.shape[1]
    for i in range(n_trials):
        label = CONDITION_LABEL[str(condition[i])]
        for f in range(n_frames - 1):
            X_list.append(np.stack([gray[i, f], gray[i, f + 1]]))
            y_list.append(label)
            tid_list.append(trial_id[i])
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    tid = np.array(tid_list, dtype=np.int64)
    return X, y, tid


def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 64,
                 lr: float = 1e-3, device: str = "cpu", seed: int = 0):
    torch.manual_seed(seed)
    model = TinyLoomNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    n = len(X)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        perm = rng.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        yield epoch, total_loss / n, model


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, device: str = "cpu") -> float:
    model.eval()
    Xt = torch.from_numpy(X).to(device)
    logits = model(Xt)
    pred = logits.argmax(dim=1).cpu().numpy()
    model.train()
    return float((pred == y).mean())


@torch.no_grad()
def static_shortcut_accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray, device: str = "cpu") -> float:
    """Validity check: duplicate each pair's first frame into both channels,
    zeroing out real motion/expansion while keeping absolute object size and
    depth-implied scale exactly as in the real data. If accuracy here is
    still high, the model has learned to read off absolute size/depth
    (looming and translation trials are drawn from different depth ranges)
    rather than the actual expansion cue -- a shortcut that would make this
    baseline invalid as a "motion-based" comparison point. Should land near
    chance (0.5) for a model doing the intended thing.
    """
    X_static = X.copy()
    X_static[:, 1] = X_static[:, 0]
    return evaluate(model, X_static, y, device=device)


@torch.no_grad()
def inference_latency_ms(model: nn.Module, X: np.ndarray, device: str = "cpu", n_reps: int = 200) -> float:
    """Mean per-frame-pair inference latency in ms, batch size 1 -- the
    realistic online-detection regime, not batched throughput."""
    model.eval()
    x1 = torch.from_numpy(X[:1]).to(device)
    for _ in range(10):  # warm-up
        model(x1)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_reps):
        model(x1)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    model.train()
    return elapsed / n_reps * 1000


