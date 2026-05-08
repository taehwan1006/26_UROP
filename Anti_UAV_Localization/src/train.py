"""
ThinDyUNet 학습 스크립트.
논문 Section 5.2 설정 기반.
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


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 1,
    log_interval: int = 100,
) -> float:
    """
    Gradient accumulation 학습.
    실제 micro-batch는 batch_size, optimizer step은 accumulation_steps마다 수행.
    Effective batch = batch_size × accumulation_steps.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        preds = model(images)
        # accumulation 시 loss를 accum_steps로 나눠서 평균 효과
        loss = criterion(preds, masks) / accumulation_steps
        loss.backward()

        # accumulation_steps마다 step
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps  # 원래 스케일로 복원
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    # 마지막 잔여 gradient flush
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metrics = SegmentationMetrics(threshold=0.5)

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        total_loss += loss.item()
        n_batches += 1
        metrics.update(preds, masks)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, metrics.compute()


def main():
    parser = argparse.ArgumentParser(description="Train ThinDyUNet")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Config 로드
    cfg = load_config(args.config)
    project_root = Path(__file__).resolve().parent.parent

    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Dataset 생성
    data_cfg = cfg["data"]
    img_size = tuple(data_cfg["img_size"])

    train_dataset = UAVSegmentationDataset(
        images_dir=str(project_root / data_cfg["images_root"] / "train"),
        masks_dir=str(project_root / data_cfg["masks_root"] / "train"),
        img_size=img_size,
        stride=data_cfg.get("train_stride", 1),
    )
    val_dataset = UAVSegmentationDataset(
        images_dir=str(project_root / data_cfg["images_root"] / "val"),
        masks_dir=str(project_root / data_cfg["masks_root"] / "val"),
        img_size=img_size,
        stride=data_cfg.get("val_stride", 1),
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    # DataLoader
    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # Model
    model_cfg = cfg["model"]
    model = ThinDyUNet(
        in_channels=model_cfg["in_channels"],
        n_classes=model_cfg["n_classes"],
        base_ch=model_cfg["base_ch"],
        n_kernels=model_cfg["n_kernels"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # Checkpoint 디렉토리
    save_dir = project_root / train_cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # Gradient accumulation
    accum_steps = train_cfg.get("accumulation_steps", 1)
    effective_batch = train_cfg["batch_size"] * accum_steps

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {train_cfg['max_epochs']} epochs")
    print(f"Micro batch: {train_cfg['batch_size']}, Accum steps: {accum_steps}, "
          f"Effective batch: {effective_batch}")
    print(f"LR: {train_cfg['learning_rate']}, Patience: {train_cfg['early_stopping_patience']}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, train_cfg["max_epochs"]):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            accumulation_steps=accum_steps,
            log_interval=train_cfg["log_interval"],
        )

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # 로그 출력
        print(
            f"Epoch [{epoch+1}/{train_cfg['max_epochs']}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Precision: {val_metrics['precision']:.3f} | "
            f"Recall: {val_metrics['recall']:.3f} | "
            f"Dice: {val_metrics['dice']:.3f} | "
            f"mIoU: {val_metrics['miou']:.3f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Best model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                },
                save_dir / "best_model.pth",
            )
            print(f"  -> Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{train_cfg['early_stopping_patience']})")

        # 매 에폭 체크포인트
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_metrics": val_metrics,
            },
            save_dir / "last_model.pth",
        )

        # Early stopping
        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\nTraining completed!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    main()
