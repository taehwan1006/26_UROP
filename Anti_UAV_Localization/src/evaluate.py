"""
ThinDyUNet 평가 스크립트.
학습된 모델의 Validation/Test 메트릭 계산 및 예측 시각화.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset.uav_dataset import UAVSegmentationDataset
from models.thin_dy_unet import ThinDyUNet
from utils.metrics import SegmentationMetrics
from utils.visualization import save_prediction_comparison


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "test",
    threshold: float = 0.5,
) -> dict:
    model.eval()
    metrics = SegmentationMetrics(threshold=threshold)

    # 추론 시간 측정
    total_time = 0.0
    n_images = 0

    for images, masks in tqdm(loader, desc=f"Evaluating ({split_name})"):
        images = images.to(device)
        masks = masks.to(device)

        start = time.time()
        preds = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        total_time += elapsed
        n_images += images.size(0)
        metrics.update(preds, masks)

    results = metrics.compute()
    results["avg_inference_ms"] = (total_time / max(n_images, 1)) * 1000
    results["total_images"] = n_images
    return results


def visualize_predictions(
    model: nn.Module,
    dataset: UAVSegmentationDataset,
    device: torch.device,
    output_dir: str,
    n_samples: int = 20,
):
    """랜덤 샘플의 예측 결과를 시각화하여 저장한다."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import random
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        image_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_input)

        save_prediction_comparison(
            image=image,
            gt_mask=mask,
            pred_mask=pred.squeeze(0).cpu(),
            save_path=str(output_dir / f"sample_{i:03d}.png"),
        )

    print(f"Saved {len(indices)} visualization samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ThinDyUNet")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save prediction visualizations",
    )
    parser.add_argument(
        "--n_vis", type=int, default=20, help="Number of samples to visualize",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Sigmoid binarization threshold. Paper uses 0.9, default is 0.5.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    data_cfg = cfg["data"]
    img_size = tuple(data_cfg["img_size"])

    # 평가 시 stride: val=val_stride, test=test_stride (없으면 1=전체)
    eval_stride = data_cfg.get(f"{args.split}_stride", 1)
    dataset = UAVSegmentationDataset(
        images_dir=str(project_root / data_cfg["images_root"] / args.split),
        masks_dir=str(project_root / data_cfg["masks_root"] / args.split),
        img_size=img_size,
        stride=eval_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    print(f"{args.split.capitalize()} samples: {len(dataset):,}")

    # Model
    model_cfg = cfg["model"]
    model = ThinDyUNet(
        in_channels=model_cfg["in_channels"],
        n_classes=model_cfg["n_classes"],
        base_ch=model_cfg["base_ch"],
        n_kernels=model_cfg["n_kernels"],
    ).to(device)

    # Load checkpoint
    ckpt_path = project_root / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']})")

    # Evaluate
    results = evaluate(model, loader, device, split_name=args.split, threshold=args.threshold)

    print(f"\n{'='*60}")
    print(f"  {args.split.upper()} SET RESULTS (threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"  Precision:               {results['precision']:.4f}")
    print(f"  Recall:                  {results['recall']:.4f}")
    print(f"  Dice Coefficient:        {results['dice']:.4f}")
    print(f"  --- IoU variants ---")
    print(f"  UAV IoU (pixel):         {results['uav_iou']:.4f}")
    print(f"  BG  IoU (pixel):         {results['bg_iou']:.4f}")
    print(f"  mIoU (pixel, BG+UAV)/2:  {results['miou_pixel']:.4f}")
    print(f"  UAV IoU (per-image avg): {results['uav_iou_per_image']:.4f}")
    print(f"  mIoU (per-image avg):    {results['miou_per_image']:.4f}")
    print(f"  ---")
    print(f"  Avg Inference:           {results['avg_inference_ms']:.2f} ms/image")
    print(f"  Total Images:            {results['total_images']:,}")
    print(f"{'='*60}")

    # Visualize
    if args.visualize:
        vis_dir = project_root / "results" / f"vis_{args.split}"
        visualize_predictions(model, dataset, device, str(vis_dir), n_samples=args.n_vis)


if __name__ == "__main__":
    main()
