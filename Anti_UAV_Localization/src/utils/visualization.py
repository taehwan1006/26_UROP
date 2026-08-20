"""
예측 결과 시각화 유틸리티 (학술대회 논문 Figure용)
"""

import cv2
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
    alpha: float = 0.6,
):
    """
    원본 이미지 위에 GT 마스크와 예측 마스크를 겹쳐서 고화질 시각화하여 저장한다.
    
    Args:
        image: (3, H, W) 정규화된 이미지 텐서
        gt_mask: (1, H, W) 바이너리 GT 마스크 텐서
        pred_mask: (1, H, W) 예측 logit (sigmoid 적용 전)
        save_path: 저장 경로
        denormalize: ImageNet 정규화 역변환 여부
        alpha: 컬러 오버레이 투명도 (0.0 ~ 1.0)
    """
    # 1. 텐서를 CPU로 이동
    image = image.cpu().float()
    gt_mask = gt_mask.cpu().float()
    pred_mask = pred_mask.cpu().float()

    # 2. 이미지 정규화 역변환 및 Numpy 변환 (0~255)
    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        
    img_np = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # 3. 마스크 이진화 (Threshold = 0.5)
    gt_np = gt_mask.squeeze(0).numpy()
    # pred_mask는 logit 상태이므로 sigmoid 적용 후 이진화
    pred_np = (torch.sigmoid(pred_mask).squeeze(0) >= 0.5).float().numpy()

    gt_bin = (gt_np >= 0.5).astype(bool)
    pred_bin = (pred_np >= 0.5).astype(bool)

    # 4. 오버레이용 빈 캔버스 생성 (RGB)
    overlay = np.zeros_like(img_np)
    
    # 색상 정의 (RGB 기준)
    COLOR_GT = [0, 255, 0]        # Green (정답이지만 못 찾은 미탐)
    COLOR_PRED = [255, 0, 0]      # Red (드론이 아닌데 드론이라고 오탐)
    COLOR_OVERLAP = [255, 255, 0] # Yellow (완벽하게 정답을 맞춘 영역)
    
    overlay[gt_bin & ~pred_bin] = COLOR_GT
    overlay[~gt_bin & pred_bin] = COLOR_PRED
    overlay[gt_bin & pred_bin] = COLOR_OVERLAP

    # 5. 투명도(Alpha) 블렌딩 적용
    mask = gt_bin | pred_bin # 색칠해야 할 전체 영역
    output = img_np.copy()
    
    # 색이 칠해질 영역만 원본 배경이 살짝 비치도록 블렌딩
    if mask.any():
        output[mask] = cv2.addWeighted(img_np[mask], 1 - alpha, overlay[mask], alpha, 0)

    # 6. 논문용 Figure 생성 ([원본 이미지 | 결과 오버레이] 가로 배치)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=18, pad=12)
    axes[0].axis("off")
    
    axes[1].imshow(output)
    axes[1].set_title("Ours (Green:GT, Red:Pred, Yellow:Overlap)", fontsize=18, pad=12)
    axes[1].axis("off")

    # 배경 여백 최소화 및 고해상도(300 dpi) 저장
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)