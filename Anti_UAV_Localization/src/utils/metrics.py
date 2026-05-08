"""
Semantic Segmentation 평가 메트릭.
Precision, Recall, Dice Coefficient, IoU 다양한 정의를 모두 계산한다.
"""

import torch


class SegmentationMetrics:
    """
    바이너리 semantic segmentation 메트릭 (배치 누적 방식).

    여러 mIoU 정의를 모두 제공하여 논문 비교 시 정의 차이로 인한 혼선을 방지:
      - uav_iou: UAV 클래스의 IoU (TP/(TP+FP+FN)) — 보통 논문에서 의미하는 값
      - miou_pixel: 픽셀 풀링 후 (BG_IoU + UAV_IoU) / 2
      - miou_per_image: 이미지별 (BG_IoU + UAV_IoU)/2 의 평균 — 일부 논문 정의
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        # 픽셀 풀링용 누적값
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        # 이미지별 평균용 누적
        self._per_image_uav_iou_sum = 0.0
        self._per_image_miou_sum = 0.0
        self._n_images = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: (B, 1, H, W) logits
            targets: (B, 1, H, W) 바이너리 마스크
        """
        if preds.requires_grad:
            preds = preds.detach()
        preds_bin = (torch.sigmoid(preds) >= self.threshold).float()
        targets = targets.float()

        # 전체 픽셀 누적
        tp_all = (preds_bin * targets).sum().item()
        fp_all = (preds_bin * (1 - targets)).sum().item()
        fn_all = ((1 - preds_bin) * targets).sum().item()
        tn_all = ((1 - preds_bin) * (1 - targets)).sum().item()

        self.tp += tp_all
        self.fp += fp_all
        self.fn += fn_all
        self.tn += tn_all

        # 이미지별 IoU 누적
        # 각 이미지 (1, H, W)별 TP/FP/FN/TN 계산
        per_img_tp = (preds_bin * targets).sum(dim=(1, 2, 3))
        per_img_fp = (preds_bin * (1 - targets)).sum(dim=(1, 2, 3))
        per_img_fn = ((1 - preds_bin) * targets).sum(dim=(1, 2, 3))
        per_img_tn = ((1 - preds_bin) * (1 - targets)).sum(dim=(1, 2, 3))

        # UAV IoU per image (UAV 픽셀이 없으면 1로 처리: GT/예측 모두 비어있으면 perfect)
        uav_denom = per_img_tp + per_img_fp + per_img_fn
        uav_iou = torch.where(uav_denom > 0, per_img_tp / uav_denom, torch.ones_like(uav_denom))

        bg_denom = per_img_tn + per_img_fp + per_img_fn
        bg_iou = torch.where(bg_denom > 0, per_img_tn / bg_denom, torch.ones_like(bg_denom))

        per_img_miou = (uav_iou + bg_iou) / 2.0

        self._per_image_uav_iou_sum += uav_iou.sum().item()
        self._per_image_miou_sum += per_img_miou.sum().item()
        self._n_images += preds_bin.size(0)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def dice(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp) / denom if denom > 0 else 0.0

    @property
    def uav_iou(self) -> float:
        """UAV 클래스 IoU (픽셀 풀링)."""
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def bg_iou(self) -> float:
        """배경 클래스 IoU (픽셀 풀링)."""
        denom = self.tn + self.fp + self.fn
        return self.tn / denom if denom > 0 else 0.0

    @property
    def miou_pixel(self) -> float:
        """픽셀 풀링 후 (BG_IoU + UAV_IoU) / 2."""
        return (self.bg_iou + self.uav_iou) / 2.0

    @property
    def miou_per_image(self) -> float:
        """이미지별 mIoU의 평균."""
        return self._per_image_miou_sum / max(self._n_images, 1)

    @property
    def uav_iou_per_image(self) -> float:
        """이미지별 UAV IoU의 평균."""
        return self._per_image_uav_iou_sum / max(self._n_images, 1)

    # 하위 호환: miou는 픽셀 풀링 정의를 유지
    @property
    def miou(self) -> float:
        return self.miou_pixel

    def compute(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "dice": self.dice,
            "uav_iou": self.uav_iou,                    # UAV-only IoU (픽셀)
            "bg_iou": self.bg_iou,                      # BG IoU (픽셀)
            "miou_pixel": self.miou_pixel,              # (BG+UAV)/2 픽셀 풀링
            "miou_per_image": self.miou_per_image,      # 이미지별 평균
            "uav_iou_per_image": self.uav_iou_per_image,
            "miou": self.miou_pixel,                    # 하위 호환
        }
