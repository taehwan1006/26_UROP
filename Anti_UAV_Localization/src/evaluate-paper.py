"""
ThinDyUNet 평가 스크립트 (논문 공식 코드 완벽 모사 버전).
주의: 논문 공식 코드의 특이점(버그 포함)을 그대로 재현한 스크립트입니다.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset.builder import build_split_dataset
from models import build_model
from utils.checkpoint import load_weights
from utils.config import load_config

# -----------------------------------------------------------------------------
# 논문 공식 코드의 utils.common 에 있던 메트릭 계산식을 그대로 모사합니다.
# (픽셀 전체가 아닌, 배치 내에서 단순 계산 후 나중에 평균을 내는 방식)
# -----------------------------------------------------------------------------
def pixel_accuracy(pred, mask):
    correct = torch.eq(pred, mask).int()
    return float(correct.sum()) / float(correct.numel())

def seg_miou(pred, mask, smooth=1e-6):
    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection
    return float((intersection + smooth) / (union + smooth))

def dice_coeff(pred, mask, smooth=1e-6):
    intersection = (pred * mask).sum()
    return float((2. * intersection + smooth) / (pred.sum() + mask.sum() + smooth))

def main():
    parser = argparse.ArgumentParser(description="Evaluate ThinDyUNet (Paper Version)")
    parser.add_argument("--config", type=str, default="configs/train_config_full.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/full/best_model.pth")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--include-sequences", type=str, nargs="+", default=None)
    parser.add_argument("--exclude-sequences", type=str, nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset & Loader
    data_cfg = cfg["data"]
    dataset = build_split_dataset(
        data_cfg, args.split, project_root,
        overrides={
            "stride": args.stride,
            "include_sequences": args.include_sequences,
            "exclude_sequences": args.exclude_sequences,
            "max_samples": args.max_samples,
        },
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # Model
    model = build_model(cfg["model"]).to(device)
    ckpt_path = project_root / args.checkpoint
    load_weights(model, ckpt_path, device)
    print(f"Test model: {ckpt_path}")
    model.eval()

    # =========================================================================
    # 아래부터 논문 공식 코드(test.py) 완벽 모사 구간입니다.
    # =========================================================================
    total_acc = 0.0
    total_miou = 0.0
    total_dice = 0.0
    total_infr_time = 0.0
    total_batches = len(loader)

    with torch.no_grad():
        for images, true_masks in tqdm(loader, desc='Model Testing (Paper Mode)'):
            # [재현 1] GPU 동기화(torch.cuda.synchronize) 누락 상태로 시간 측정 시작
            start_time = time.time()
            
            images = images.to(device)
            true_masks = true_masks.to(device)
            outputs = model(images)

            # [재현 2] Threshold 0.9 하드코딩 적용
            pred_masks = outputs.sigmoid()
            pred_masks = (pred_masks > 0.9).float()
            
            # [재현 3] 연산 완료 대기 없이 바로 시간 측정 종료
            end_time = time.time()
            total_infr_time += (end_time - start_time)

            # [재현 4] 전체 픽셀이 아닌, 배치(Batch) 단위의 점수를 단순 누적
            total_acc += pixel_accuracy(pred_masks, true_masks)
            total_miou += seg_miou(pred_masks, true_masks)
            total_dice += dice_coeff(pred_masks, true_masks)

    # 단순 산술 평균 계산
    avg_acc = total_acc / total_batches
    avg_miou = total_miou / total_batches
    avg_dice = total_dice / total_batches
    avg_infr_time = total_infr_time / total_batches

    print('\n============================================================')
    print('  TEST METRICS (PAPER ORIGINAL METHOD) ')
    print('============================================================')
    # [재현 5] 추론 시간 밀리초 변환 버그 (* 100) 그대로 적용
    print(f"Pixel Accuracy: {avg_acc:.4f} | Mean IoU: {avg_miou:.4f} | Dice Coeff: {avg_dice:.4f} | Inference Time: {avg_infr_time*100:.2f} ms")
    print('============================================================')

if __name__ == "__main__":
    main()