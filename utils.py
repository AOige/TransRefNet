"""
utils.py
--------
Shared utilities for TransRefNet training and evaluation.
"""

import os
import random
import time
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seeding(seed: int = 42):
    """Fix all random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN (slightly slower; set to False for max speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# File / directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_dir(path: str):
    """Create directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save(path: str, text: str):
    """Print text to stdout and append it to a log file."""
    print(text)
    with open(path, "a") as f:
        f.write(text + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

def epoch_time(start_time: float, end_time: float):
    """Return elapsed time as (minutes, seconds) integers."""
    elapsed = end_time - start_time
    return int(elapsed / 60), int(elapsed % 60)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample metric computation (operates on a single prediction + ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute Jaccard (IoU), Dice (F1), Recall, and Precision for a single
    prediction–mask pair.

    Parameters
    ----------
    y_true : Tensor  – ground-truth binary mask, shape [1, H, W] or [H, W]
    y_pred : Tensor  – raw network output (logits or probabilities),
                       same shape as y_true

    Returns
    -------
    (jaccard, f1, recall, precision)  –  floats in [0, 1]
    """
    # Binarise prediction at 0.5
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()
    y_true = y_true.float()

    # Flatten
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    # TP, FP, FN
    tp = (y_true * y_pred).sum().item()
    fp = ((1 - y_true) * y_pred).sum().item()
    fn = (y_true * (1 - y_pred)).sum().item()

    eps = 1e-7

    jaccard  = tp / (tp + fp + fn + eps)
    f1       = (2 * tp) / (2 * tp + fp + fn + eps)
    recall   = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)

    return jaccard, f1, recall, precision
