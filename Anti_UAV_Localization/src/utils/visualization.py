"""
예측 결과 시각화 유틸리티.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def save_prediction_comparison(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    save_path: str,
    denormalize: bool = True,
):
    """
    원본 이미지, GT 마스크, 예측 마스크를 나란히 시각화하여 저장한다.

    Args:
        image: (3, H, W) 정규화된 이미지
        gt_mask: (1, H, W) 바이너리 GT 마스크
        pred_mask: (1, H, W) 예측 logit (sigmoid 전)
        save_path: 저장 경로
        denormalize: ImageNet 정규화 역변환 여부
    """
    image = image.cpu().float()
    gt_mask = gt_mask.cpu().float()
    pred_mask = pred_mask.cpu().float()

    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()

    gt_mask = gt_mask.squeeze(0).numpy()
    pred_mask = (torch.sigmoid(pred_mask).squeeze(0) >= 0.5).float().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
