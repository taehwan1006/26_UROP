"""
학습된 모델로 시각화만 빠르게 생성 (전체 평가 생략).
하나의 figure에 row=sample, col=[input, GT, pred, overlay]로 정리해 한눈에 비교.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.builder import build_split_dataset
from models import build_model
from utils.checkpoint import load_weights
from utils.config import load_config
from utils.image import denorm


def overlay(img: np.ndarray, mask: np.ndarray, color=(1.0, 0.2, 0.2), alpha=0.5) -> np.ndarray:
    """마스크 영역을 빨강으로 반투명 오버레이."""
    out = img.copy()
    rgb = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * rgb
    return out


def find_uav_indices(dataset: Dataset, n: int, seed: int = 0) -> list:
    """UAV가 실제로 있는 프레임 위주로 샘플 인덱스 선택."""
    rng = random.Random(seed)
    n_total = len(dataset)
    candidates = list(range(n_total))
    rng.shuffle(candidates)

    chosen = []
    for idx in candidates:
        _, mask = dataset[idx]
        if mask.sum() > 50:  # UAV 픽셀이 있는 샘플
            chosen.append(idx)
            if len(chosen) >= n:
                break
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config_stride10.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/stride10/best_model.pth")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_samples", type=int, default=12, help="시각화 샘플 수")
    parser.add_argument("--output", default="results/vis_grid_stride10.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_stride", type=int, default=200,
        help="시각화용 데이터 검색을 위한 stride (큰 값으로 빠르게)",
    )
    parser.add_argument("--include-sequences", type=str, nargs="+", default=None,
                        help="시각화할 시퀀스 glob 패턴 (예: video01 video0*)")
    parser.add_argument("--exclude-sequences", type=str, nargs="+", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(str(project_root / args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset (시각화 후보군만 빠르게 모으기 위해 stride 큰 값)
    dataset = build_split_dataset(
        cfg["data"], args.split, project_root,
        overrides={
            "stride": args.eval_stride,
            "include_sequences": args.include_sequences,
            "exclude_sequences": args.exclude_sequences,
        },
    )

    # Model
    model = build_model(cfg["model"]).to(device)
    ckpt_path = project_root / args.checkpoint
    ckpt = load_weights(model, ckpt_path, device)
    model.eval()
    print(f"Loaded: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # 샘플 선택
    indices = find_uav_indices(dataset, args.n_samples, seed=args.seed)
    print(f"Selected {len(indices)} UAV-containing samples")

    # 그리드 시각화
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Input", "Ground Truth", "Prediction", "Overlay (red=pred)"]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]
            x = image.unsqueeze(0).to(device)
            pred_logit = model(x)
            pred = (torch.sigmoid(pred_logit).squeeze().cpu().numpy() >= 0.5).astype(np.uint8)

            img_np = denorm(image)
            gt_np = mask.squeeze().cpu().numpy()

            axes[row, 0].imshow(img_np)
            axes[row, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[row, 3].imshow(overlay(img_np, pred))

            for c in range(4):
                axes[row, c].axis("off")
                if row == 0:
                    axes[row, c].set_title(col_titles[c], fontsize=14)

            # 좌측 라벨
            axes[row, 0].text(
                -50, img_np.shape[0] // 2, f"#{idx}",
                rotation=90, va="center", ha="right", fontsize=11,
            )

    plt.tight_layout()
    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # 개별 샘플도 따로 저장 (sample_XXX.png)
    indiv_dir = out_path.parent / f"vis_{args.split}_stride10"
    indiv_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            x = image.unsqueeze(0).to(device)
            pred_logit = model(x)
            pred = (torch.sigmoid(pred_logit).squeeze().cpu().numpy() >= 0.5).astype(np.uint8)

            img_np = denorm(image)
            gt_np = mask.squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img_np); axes[0].set_title("Input"); axes[0].axis("off")
            axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1); axes[1].set_title("GT"); axes[1].axis("off")
            axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Pred"); axes[2].axis("off")
            axes[3].imshow(overlay(img_np, pred)); axes[3].set_title("Overlay"); axes[3].axis("off")
            plt.tight_layout()
            plt.savefig(indiv_dir / f"sample_{i:03d}_idx{idx}.png", dpi=100, bbox_inches="tight")
            plt.close()
    print(f"Saved {len(indices)} individual samples to: {indiv_dir}")


if __name__ == "__main__":
    main()
