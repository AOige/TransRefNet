"""
metrics.py
----------
Loss functions for TransRefNet training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Dice Loss
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice Loss.  Operates on raw logits (applies sigmoid internally).
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs   = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


# ─────────────────────────────────────────────────────────────────────────────
# Combined Dice + BCE Loss  (used throughout training)
# ─────────────────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """
    Linear combination of Dice Loss and Binary Cross-Entropy Loss.

    Args:
        dice_weight : weight for the Dice component (default 0.5).
        bce_weight  : weight for the BCE component  (default 0.5).
        smooth      : Dice smoothing constant.
    """
    def __init__(self, dice_weight: float = 0.5,
                 bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice_loss   = DiceLoss(smooth=smooth)
        self.bce_loss    = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce  = self.bce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce
