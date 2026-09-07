"""Segmentation 손실 함수와 config 기반 factory."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Binary Dice loss on sigmoid(logits)."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()
        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.w_bce = bce_weight
        self.w_dice = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w_bce * self.bce(logits, targets) + self.w_dice * self.dice(logits, targets)


def build_loss(loss_cfg: dict) -> nn.Module:
    """config의 training.loss 블록으로부터 손실 함수를 만든다."""
    name = loss_cfg.get("name", "bce_dice").lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
        )
    raise ValueError(f"Unknown loss: {name}")
