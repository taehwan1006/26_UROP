"""
ThinDyUNet 학습 스크립트 (논문 베이스 + 검증된 개선 통합).

논문 적용:
  - AdamW optimizer
  - DiceLoss (또는 BCE+Dice combo)
  - ReduceLROnPlateau (factor=0.15, patience=10, cooldown=5)

우리 개선 유지:
  - ImageNet Normalize (dataset)
  - shuffle=True
  - N-fold efficient dynamic conv (model)
  - Gradient accumulation으로 effective batch 동일
  - 3가지 mIoU 정의 모두 출력
  - CSV epoch 로그, 옵션 AMP, 주기적 체크포인트

사용:
    # 처음부터 학습
    python src/train_full.py --config configs/train_config_full.yaml

    # 중단 지점부터 재개 (optimizer/scheduler 상태까지 복원)
    python src/train_full.py --resume checkpoints/full/last_model.pth

    # 사전학습 가중치로 파인튜닝 (가중치만 로드, optimizer는 새 LR로 재초기화)
    python src/train_full.py --config configs/train_config_full_dut.yaml \
        --finetune checkpoints/full/best_model.pth

    # config의 다른 split 조합으로 학습
    python src/train_full.py --train-split train_dut --val-split val_dut
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset.builder import build_split_dataset
from engine import train_one_epoch, validate
from models import build_model
from utils.checkpoint import load_weights, resume_training, save_checkpoint
from utils.config import load_config
from utils.losses import build_loss


CSV_HEADER = [
    "epoch", "train_loss", "val_loss",
    "precision", "recall", "dice",
    "uav_iou_pixel", "miou_pixel",
    "uav_iou_per_image", "miou_per_image",
    "lr", "epoch_time_sec",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train ThinDyUNet")
    parser.add_argument("--config", type=str, default="configs/train_config_full.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="학습 재개 (가중치 + optimizer/scheduler 복원)")
    parser.add_argument("--finetune", type=str, default=None,
                        help="파인튜닝 (가중치만 로드, optimizer는 재초기화)")
    parser.add_argument("--train-split", type=str, default="train",
                        help="config data.sources 의 학습 split 이름")
    parser.add_argument("--val-split", type=str, default="val",
                        help="config data.sources 의 검증 split 이름")
    args = parser.parse_args()
    if args.resume and args.finetune:
        parser.error("--resume 과 --finetune 은 함께 쓸 수 없습니다")
    return args


def main():
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(__file__).resolve().parent.parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Dataset — 소스 추가/시퀀스 선택은 config의 data.sources 에서만 하면 된다
    data_cfg = cfg["data"]
    train_dataset = build_split_dataset(data_cfg, args.train_split, project_root)
    val_dataset = build_split_dataset(data_cfg, args.val_split, project_root)

    train_cfg = cfg["training"]
    num_workers = data_cfg["num_workers"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Model
    model = build_model(cfg["model"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__} - {total_params:,} params ({total_params / 1e6:.2f}M)")

    # Loss
    criterion = build_loss(train_cfg.get("loss", {"name": "bce_dice"}))
    print(f"Loss: {type(criterion).__name__}")

    # Optimizer (AdamW per paper)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=weight_decay,
    )
    print(f"Optimizer: AdamW (lr={train_cfg['learning_rate']}, wd={weight_decay})")

    # Scheduler (ReduceLROnPlateau per paper)
    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.get("factor", 0.15),
        patience=sched_cfg.get("patience", 10),
        cooldown=sched_cfg.get("cooldown", 5),
        min_lr=sched_cfg.get("min_lr", 1e-7),
    )

    # AMP
    use_amp = bool(train_cfg.get("amp_enabled", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"AMP fp16: {use_amp}")

    save_dir = project_root / train_cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.finetune:
        # 가중치만 이어받고 optimizer/scheduler는 새 LR로 시작
        load_weights(model, Path(args.finetune), device)
        print(f"Fine-tuning from: {args.finetune}")
    elif args.resume:
        start_epoch, best_val_loss = resume_training(
            model, optimizer, Path(args.resume), device,
            scheduler=scheduler, scaler=scaler,
        )
        print(f"Resumed from: {args.resume} (epoch {start_epoch})")

    accum_steps = train_cfg.get("accumulation_steps", 1)
    effective_batch = train_cfg["batch_size"] * accum_steps

    csv_path = save_dir / "train_log.csv"
    if not args.resume or not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    ckpt_every = train_cfg.get("ckpt_every_epochs", 0)
    print(f"\n{'='*60}")
    print(f"Training {train_cfg['max_epochs']} epochs")
    print(f"Batch: {train_cfg['batch_size']} × accum {accum_steps} = effective {effective_batch}")
    print(f"Scheduler: ReduceLROnPlateau factor={sched_cfg.get('factor', 0.15)} "
          f"patience={sched_cfg.get('patience', 10)} cooldown={sched_cfg.get('cooldown', 5)}")
    print(f"Save dir: {save_dir}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, train_cfg["max_epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            accumulation_steps=accum_steps,
            log_interval=train_cfg["log_interval"],
            scaler=scaler,
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{train_cfg['max_epochs']}] "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"P {val_metrics['precision']:.3f} R {val_metrics['recall']:.3f} "
            f"Dice {val_metrics['dice']:.3f} "
            f"mIoU(px) {val_metrics['miou_pixel']:.3f} "
            f"UAV-IoU(img) {val_metrics['uav_iou_per_image']:.3f} | "
            f"LR {current_lr:.2e} | {epoch_time:.0f}s"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{val_metrics['precision']:.6f}",
                f"{val_metrics['recall']:.6f}",
                f"{val_metrics['dice']:.6f}",
                f"{val_metrics['uav_iou']:.6f}",
                f"{val_metrics['miou_pixel']:.6f}",
                f"{val_metrics['uav_iou_per_image']:.6f}",
                f"{val_metrics['miou_per_image']:.6f}",
                f"{current_lr:.3e}",
                f"{epoch_time:.1f}",
            ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1
            improved = False

        ckpt_kwargs = dict(
            epoch=epoch, model=model, optimizer=optimizer,
            best_val_loss=best_val_loss, scheduler=scheduler,
            scaler=scaler, val_metrics=val_metrics,
        )
        if improved:
            save_checkpoint(save_dir / "best_model.pth", **ckpt_kwargs)
            print(f"  -> Best (val_loss {best_val_loss:.4f})")
        else:
            print(f"  -> No improvement ({patience_counter}/{train_cfg['early_stopping_patience']})")

        save_checkpoint(save_dir / "last_model.pth", **ckpt_kwargs)
        if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
            save_checkpoint(save_dir / f"epoch_{epoch+1:03d}.pth", **ckpt_kwargs)

        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nDone. Best val loss: {best_val_loss:.4f} | Saved in: {save_dir}")


if __name__ == "__main__":
    main()
