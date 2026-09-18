"""
세그멘테이션(ThinDyUNet) + 단일 표적 트래커 통합 평가 스크립트.

1단계 (네트워크 추론, 체크포인트·split별 1회): 시퀀스 전체 프레임의 확률맵을 캐시에 저장
2단계 (트래킹 + 평가, 반복 실행): 캐시 → 연결성분 후보 → 트래커 → 지표

지표
- [주] 픽셀 지표 (utils/metrics.py SegmentationMetrics와 같은 정의, SAM 2.1 마스크 GT, 512×512)
    · all   : 필터링 없는 V2 마스크 (확률 >= high_thresh 전체) — evaluate.py 재현용 기준
    · track : 트래커가 고른 성분만 남긴 마스크 (coast·미출력 프레임은 빈 마스크)
- [참고] 박스 SOT 지표 (Success / Precision@20px / Norm.Precision / State Accuracy)

전처리는 UAVSegmentationDataset과 동일하다: 원본 프레임 전체를 img_size(512×512)로 bilinear
리사이즈 → ToTensor → ImageNet Normalize. 박스는 원본 해상도로 되돌려 GT와 비교한다.

사용 예:
    # 세그멘테이션 단독 기준선 (매 프레임 최고 점수 성분)
    python Anti_UAV_Localization/src/track_eval.py \
        --config Anti_UAV_Localization/configs/train_config_full_dut.yaml \
        --checkpoint checkpoints/best_dut_v3/best_model.pth \
        --mode per_frame --run-name baseline_per_frame

    # Kalman + 게이팅 + 2단 연관 + coasting (캐시 재사용, 추론 생략)
    python Anti_UAV_Localization/src/track_eval.py \
        --config Anti_UAV_Localization/configs/train_config_full_dut.yaml \
        --checkpoint checkpoints/best_dut_v3/best_model.pth \
        --mode kalman --run-name kf_full
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tracking.detections import prob_to_detections, q_thresh
from tracking.pixel_metrics import PixelMetrics, build_or_load_gt_masks
from tracking.prob_cache import SparseProbReader, SparseProbWriter
from tracking.sot_metrics import aggregate, evaluate_sequence, load_gt, save_pred
from tracking.tracker import SingleTargetTracker, TrackerConfig

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------- #
# 시퀀스 탐색
# ---------------------------------------------------------------------- #
def _matches_any(rel_path: str, patterns: Sequence[str]) -> bool:
    """uav_dataset._matches_any 와 같은 규칙 (torch 없이 쓰기 위해 복제)."""
    segments = rel_path.split("/")
    for pat in patterns:
        pat = pat.replace("\\", "/").rstrip("/")
        if fnmatch(rel_path, pat) or fnmatch(rel_path, f"{pat}/*"):
            return True
        if any(fnmatch(seg, pat) for seg in segments):
            return True
    return False


def resolve_masks_dir(data_cfg: dict, split: str, project_root: Path) -> Path:
    sources = (data_cfg.get("sources") or {}).get(split)
    if sources:
        masks_dir = sources[0]["masks_dir"]
    else:
        masks_dir = f"{data_cfg.get('masks_root', 'data/raw/masks')}/{split}"
    p = Path(masks_dir)
    return p if p.is_absolute() else project_root / p


def resolve_images_dir(data_cfg: dict, split: str, project_root: Path) -> Path:
    sources = (data_cfg.get("sources") or {}).get(split)
    if sources:
        if len(sources) > 1:
            print(f"[warn] split '{split}'에 소스가 {len(sources)}개 — 트래킹 평가는 첫 소스만 사용: {sources[0]['name']}")
        images_dir = sources[0]["images_dir"]
    else:
        images_dir = f"{data_cfg.get('images_root', 'data/raw/images')}/{split}"
    p = Path(images_dir)
    return p if p.is_absolute() else project_root / p


def discover_sequences(
    images_dir: Path,
    include: Optional[Sequence[str]],
    exclude: Optional[Sequence[str]],
) -> List[Tuple[str, List[Path]]]:
    leaf_dirs = set()
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
            leaf_dirs.add(p.parent)
    seqs = []
    for leaf in sorted(leaf_dirs):
        rel = leaf.relative_to(images_dir).as_posix()
        if exclude and _matches_any(rel, exclude):
            continue
        if include and not _matches_any(rel, include):
            continue
        frames = sorted(p for p in leaf.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
        if frames:
            seqs.append((rel, frames))
    return seqs


# ---------------------------------------------------------------------- #
# 1단계: 추론 → 캐시
# ---------------------------------------------------------------------- #
class _Model:
    """모델/디바이스를 필요할 때 한 번만 로드 (캐시가 모두 있으면 torch를 import하지 않음)."""

    def __init__(self, cfg: dict, ckpt_path: Path, amp: bool):
        import torch
        from models import build_model
        from utils.checkpoint import load_weights

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(cfg["model"]).to(self.device).eval()
        ckpt = load_weights(self.model, ckpt_path, self.device)
        self.amp = amp and self.device.type == "cuda"
        print(f"Device: {self.device} | Loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')}) | AMP={self.amp}")


class FrameListDataset:
    """시퀀스 프레임 목록용 map-style 데이터셋.

    Windows에서 DataLoader(num_workers>0)는 spawn으로 데이터셋을 pickle하므로 모듈 최상위에 정의한다.
    전처리는 UAVSegmentationDataset.img_transform과 동일.
    """

    def __init__(self, frames: List[Path], img_size: Tuple[int, int]):
        from torchvision import transforms
        from utils.image import IMAGENET_MEAN, IMAGENET_STD

        self.frames = list(frames)
        self.tf = transforms.Compose([
            transforms.Resize(tuple(img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        import torch
        from PIL import Image

        im = Image.open(self.frames[i])
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        return self.tf(im), torch.tensor([w, h], dtype=torch.int32)


def infer_sequence(
    m: _Model,
    frames: List[Path],
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    writer: SparseProbWriter,
) -> np.ndarray:
    torch = m.torch
    from torch.utils.data import DataLoader

    loader = DataLoader(FrameListDataset(frames, img_size), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(m.device.type == "cuda"))
    sizes = []
    with torch.no_grad():
        for images, wh in loader:
            images = images.to(m.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=m.amp):
                logits = m.model(images)
            prob_q = torch.round(torch.sigmoid(logits.float()) * 255.0).to(torch.uint8)
            prob_q = prob_q[:, 0].cpu().numpy()
            for k in range(prob_q.shape[0]):
                writer.add(prob_q[k])
            sizes.append(wh.numpy())
    return np.concatenate(sizes, axis=0)


def ensure_cache(
    seqs, cfg, args, project_root: Path, cache_dir: Path
) -> Dict[str, Path]:
    img_size = tuple(cfg["data"]["img_size"])
    ckpt_path = Path(args.checkpoint)
    ckpt_stat = ckpt_path.stat() if ckpt_path.exists() else None
    meta = {
        "checkpoint": str(ckpt_path),
        "ckpt_size": ckpt_stat.st_size if ckpt_stat else -1,
        "ckpt_mtime": int(ckpt_stat.st_mtime) if ckpt_stat else -1,
        "amp": bool(args.amp),
        "img_size": f"{img_size[0]}x{img_size[1]}",
        "model": cfg["model"].get("name", "ThinDyUNet"),
    }

    paths, model = {}, None
    for rel, frames in seqs:
        path = cache_dir / (rel.replace("/", "__") + ".npz")
        paths[rel] = path
        if path.exists() and not args.refresh_cache:
            r = SparseProbReader(path)
            stale = (
                len(r) != len(frames)
                or r.meta.get("ckpt_size") != str(meta["ckpt_size"])
                or r.meta.get("ckpt_mtime") != str(meta["ckpt_mtime"])
                or r.min_prob > args.cache_min_prob + 1e-9
            )
            if not stale:
                continue
            print(f"[cache] {rel}: 캐시가 현재 설정과 달라 다시 추론합니다.")
        if model is None:
            if ckpt_stat is None:
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
            model = _Model(cfg, ckpt_path, args.amp)
        t0 = time.time()
        writer = SparseProbWriter(img_size, args.cache_min_prob)
        sizes = infer_sequence(model, frames, img_size, args.batch_size, args.num_workers, writer)
        writer.save(path, [f.name for f in frames], sizes, meta)
        print(f"[cache] {rel}: {len(frames)} frames inferred in {time.time() - t0:.1f}s → {path}")
    return paths


# ---------------------------------------------------------------------- #
# 2단계: 트래킹 + 평가
# ---------------------------------------------------------------------- #
EMPTY_IDX = np.zeros(0, dtype=np.uint32)


def run_tracking_on_sequence(
    reader: SparseProbReader,
    tcfg: TrackerConfig,
    args,
    gt_masks: Optional[List[Optional[np.ndarray]]] = None,
):
    """반환: pred(N,4), scores, statuses, t_post(s), pixel dict {"all": PixelMetrics, "track": PixelMetrics} 또는 None"""
    tracker = SingleTargetTracker(tcfg)
    n = len(reader)
    pred = np.full((n, 4), np.nan)
    scores = np.zeros(n)
    statuses: List[str] = []
    t_post = 0.0
    use_low = tcfg.mode == "kalman" and tcfg.use_low
    qh = q_thresh(args.high_thresh)
    pix = None
    if gt_masks is not None:
        npix = reader.in_hw[0] * reader.in_hw[1]
        pix = {"all": PixelMetrics(npix), "track": PixelMetrics(npix)}
    for i in range(n):
        prob_q = reader.get(i)
        t0 = time.perf_counter()
        dets = prob_to_detections(
            prob_q, tuple(reader.orig_sizes[i]),
            high_thresh=args.high_thresh, low_thresh=args.low_thresh,
            score_mode=args.score_mode, min_area=args.min_area, use_low=use_low,
        )
        out = tracker.step(dets)
        t_post += time.perf_counter() - t0
        if out.box is not None:
            pred[i] = out.box
        scores[i] = out.score
        statuses.append(out.status)

        if pix is not None and gt_masks[i] is not None:
            idx, val = reader.get_sparse(i)
            pix["all"].update(idx[val >= qh], gt_masks[i])
            parts = [d.pix for d in out.mask_dets if d.is_high or args.low_mask == "include"]
            track_idx = EMPTY_IDX if not parts else (parts[0] if len(parts) == 1 else np.unique(np.concatenate(parts)))
            pix["track"].update(track_idx, gt_masks[i])
    return pred, scores, statuses, t_post, pix


def build_tracker_config(args) -> TrackerConfig:
    return TrackerConfig(
        mode=args.mode, select=args.select, init_thresh=args.init_thresh,
        use_low=not args.no_low, gate=not args.no_gate, gate_chi2=args.gate_chi2,
        gate_dims=args.gate_dims, gate_min_radius=args.gate_min_radius, max_coast=args.max_coast, max_age=args.max_age,
        reinit_after=args.reinit_after, min_hits=args.min_hits, output_box=args.output_box,
        mask_mode=args.mask_mode, mask_expand=args.mask_expand, coast_output=args.coast_output,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ThinDyUNet + single-target tracker evaluation (pixel + SOT metrics)")
    g = p.add_argument_group("data / model")
    g.add_argument("--config", type=str, default="configs/train_config_full_dut.yaml")
    g.add_argument("--checkpoint", type=str, required=True, help="절대경로, 현재 폴더 기준, 또는 Anti_UAV_Localization 기준 상대경로")
    g.add_argument("--split", type=str, default="test_dut")
    g.add_argument("--gt-dir", type=str,
                   default="../DUT-Anti-UAV/Anti-UAV-Tracking-V0GT/Anti-UAV-Tracking-V0GT",
                   help="videoXX_gt.txt 폴더 (기본값은 Anti_UAV_Localization 기준 상대경로)")
    g.add_argument("--include-sequences", type=str, nargs="+", default=None)
    g.add_argument("--exclude-sequences", type=str, nargs="+", default=None)
    g.add_argument("--batch-size", type=int, default=None, help="추론 배치 (기본: config training.batch_size)")
    g.add_argument("--num-workers", type=int, default=None)
    g.add_argument("--amp", action="store_true", help="FP16 autocast 추론 (기본은 evaluate.py와 같은 FP32)")

    g = p.add_argument_group("cache")
    g.add_argument("--cache-dir", type=str, default=None, help="기본: results/tracking/_cache/<ckpt>_<split>_<fp32|amp>")
    g.add_argument("--cache-min-prob", type=float, default=0.05)
    g.add_argument("--refresh-cache", action="store_true")

    g = p.add_argument_group("detections (mask → components)")
    g.add_argument("--high-thresh", type=float, default=0.5)
    g.add_argument("--low-thresh", type=float, default=0.3)
    g.add_argument("--score-mode", choices=["max", "mean"], default="max")
    g.add_argument("--min-area", type=int, default=1, help="512×512 공간 픽셀 수")

    g = p.add_argument_group("pixel metrics")
    g.add_argument("--no-pixel-eval", action="store_true", help="픽셀 지표 계산 끄기")
    g.add_argument("--low-mask", choices=["include", "exclude"], default="include",
                   help="저신뢰(2차 연관) 성분이 선택된 프레임에서 그 성분 픽셀(low_thresh 이진화)을 마스크에 포함할지")
    g.add_argument("--refresh-gt-cache", action="store_true")

    d = TrackerConfig()
    g = p.add_argument_group("tracker")
    g.add_argument("--mode", choices=["per_frame", "kalman"], default=d.mode)
    g.add_argument("--select", choices=["score", "area"], default=d.select)
    g.add_argument("--init-thresh", type=float, default=d.init_thresh)
    g.add_argument("--no-low", action="store_true", help="저신뢰 2차 연관 끄기")
    g.add_argument("--no-gate", action="store_true", help="게이트 끄기 (최근접 후보와 무조건 연관)")
    g.add_argument("--gate-chi2", type=float, default=d.gate_chi2,
                   help="Mahalanobis² 게이트 임계값 (자유도 2: 9.21, 자유도 4: 13.28 권장)")
    g.add_argument("--gate-dims", type=int, choices=[2, 4], default=d.gate_dims,
                   help="연관 기준 차원: 2=중심만, 4=중심+크기")
    g.add_argument("--gate-min-radius", type=float, default=d.gate_min_radius)
    g.add_argument("--max-coast", type=int, default=d.max_coast)
    g.add_argument("--max-age", type=int, default=d.max_age)
    g.add_argument("--reinit-after", type=int, default=d.reinit_after)
    g.add_argument("--min-hits", type=int, default=d.min_hits)
    g.add_argument("--output-box", choices=["det", "kf"], default=d.output_box)
    g.add_argument("--mask-mode", choices=["selected", "expand"], default=d.mask_mode,
                   help="마스크 출력 범위: selected=고른 성분 1개, expand=출력 박스를 확대한 영역과 겹치는 성분 전부")
    g.add_argument("--mask-expand", type=float, default=d.mask_expand, help="mask_mode=expand의 박스 확대 배율")
    g.add_argument("--coast-output", choices=["box", "none", "mask"], default=d.coast_output,
                   help="가림 구간 출력: box=박스만(기존), none=출력 없음, mask=박스+예측 영역 성분 마스크")

    g = p.add_argument_group("output")
    g.add_argument("--run-name", type=str, default=None)
    g.add_argument("--out-root", type=str, default="results/tracking")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent

    def _abs(s: str) -> Path:
        """절대경로 → 그대로 / 현재 작업 폴더 기준으로 존재 → 그 경로 / 그 외 → project_root 기준."""
        q = Path(s)
        if q.is_absolute():
            return q
        return q.resolve() if q.exists() else project_root / q

    cfg_path = _abs(args.config)
    args.checkpoint = str(_abs(args.checkpoint))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.low_thresh < args.cache_min_prob:
        raise ValueError(f"--low-thresh({args.low_thresh}) must be >= --cache-min-prob({args.cache_min_prob})")
    if args.batch_size is None:
        args.batch_size = cfg.get("training", {}).get("batch_size", 8)
    if args.num_workers is None:
        args.num_workers = cfg["data"].get("num_workers", 4)

    images_dir = resolve_images_dir(cfg["data"], args.split, project_root)
    gt_dir = _abs(args.gt_dir)
    seqs = discover_sequences(images_dir, args.include_sequences, args.exclude_sequences)
    if not seqs:
        raise FileNotFoundError(f"no sequences under {images_dir}")

    # GT 로드 + 프레임 수 검증 (maskconv.py와 같이 정렬 순서 i번째 프레임 ↔ i번째 GT 줄)
    gts = {}
    for rel, frames in seqs:
        gt_path = gt_dir / f"{Path(rel).name}_gt.txt"
        if not gt_path.exists():
            raise FileNotFoundError(f"GT not found for {rel}: {gt_path}")
        gt = load_gt(gt_path)
        if len(gt) != len(frames):
            raise ValueError(f"{rel}: frames({len(frames)}) != GT lines({len(gt)}) — 정렬/누락 확인 필요")
        gts[rel] = gt
    print(f"[data] {images_dir} | {len(seqs)} sequences | {sum(len(f) for _, f in seqs):,} frames")

    ckpt_p = Path(args.checkpoint)
    ckpt_tag = f"{ckpt_p.parent.name}_{ckpt_p.stem}" if ckpt_p.parent.name else ckpt_p.stem
    out_root = project_root / args.out_root if not Path(args.out_root).is_absolute() else Path(args.out_root)
    cache_dir = _abs(args.cache_dir) if args.cache_dir else \
        out_root / "_cache" / f"{ckpt_tag}_{args.split}_{'amp' if args.amp else 'fp32'}"
    cache_paths = ensure_cache(seqs, cfg, args, project_root, cache_dir)

    # 픽셀 GT 마스크 (SAM 2.1) — UAVSegmentationDataset과 같은 경로 규칙·전처리
    img_hw = tuple(cfg["data"]["img_size"])
    masks_dir = resolve_masks_dir(cfg["data"], args.split, project_root)
    do_pixel = not args.no_pixel_eval
    if do_pixel and not masks_dir.exists():
        print(f"[warn] masks_dir 없음 → 픽셀 지표 생략: {masks_dir}")
        do_pixel = False
    gt_cache_dir = out_root / "_cache" / f"gt_masks_{args.split}_{img_hw[0]}x{img_hw[1]}"

    tcfg = build_tracker_config(args)
    run_name = args.run_name or f"{tcfg.mode}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = out_root / run_name
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)

    seq_results, status_total, t_post_total, n_total = {}, Counter(), 0.0, 0
    pix_total = None
    seq_pixel: Dict[str, Dict[str, dict]] = {}
    n_missing_masks = 0
    for rel, frames in seqs:
        reader = SparseProbReader(cache_paths[rel])
        gt_masks = None
        if do_pixel:
            t0 = time.time()
            gt_masks, n_miss = build_or_load_gt_masks(
                gt_cache_dir / (rel.replace("/", "__") + ".npz"), masks_dir, rel, frames, img_hw,
                num_threads=args.num_workers, refresh=args.refresh_gt_cache)
            n_missing_masks += n_miss
            if time.time() - t0 > 2:
                print(f"  [gt-mask cache] {rel}: {time.time() - t0:.1f}s")
        pred, scores, statuses, t_post, pix = run_tracking_on_sequence(reader, tcfg, args, gt_masks)
        if pix is not None:
            seq_pixel[rel] = {k: v.compute() for k, v in pix.items()}
            if pix_total is None:
                npix = img_hw[0] * img_hw[1]
                pix_total = {"all": PixelMetrics(npix), "track": PixelMetrics(npix)}
            for k in pix:
                pix_total[k].merge(pix[k])
        res = evaluate_sequence(gts[rel], pred)
        res["summary"]["status_counts"] = dict(Counter(statuses))
        seq_results[rel] = res
        status_total.update(statuses)
        t_post_total += t_post
        n_total += len(frames)
        save_pred(out_dir / "preds" / f"{Path(rel).name}.txt", pred, scores, statuses)
        s = res["summary"]
        line = (f"  {rel:>10s} | AUC {s['success_auc']:.4f} | P@20 {s['precision_20px']:.4f} "
                f"| SA {s['state_accuracy']:.4f} | absentFP {s['absent_frame_fp_rate']:.3f}")
        if rel in seq_pixel:
            a, t = seq_pixel[rel]["all"], seq_pixel[rel]["track"]
            line += (f" || pixel all R {a['recall']:.3f} P {a['precision']:.3f} IoU {a['uav_iou']:.3f}"
                     f" | track R {t['recall']:.3f} P {t['precision']:.3f} IoU {t['uav_iou']:.3f}")
        print(line + f" | {len(frames)} fr")

    agg = aggregate(seq_results)
    agg["postprocess_ms_per_frame"] = 1000.0 * t_post_total / max(n_total, 1)
    agg["status_counts"] = dict(status_total)
    if pix_total is not None:
        agg["pixel"] = {k: v.compute() for k, v in pix_total.items()}
        agg["pixel"]["low_mask"] = args.low_mask
        agg["pixel"]["frames_without_mask"] = n_missing_masks

    sm, fp = agg["sequence_mean"], agg["frame_pooled"]
    print(f"\n{'=' * 72}")
    print(f"  {run_name}  ({args.split}, {len(seqs)} seqs, {n_total:,} frames, mode={tcfg.mode})")
    print(f"{'=' * 72}")
    if "pixel" in agg:
        print(f"  [PIXEL, threshold {args.high_thresh}, micro]   Recall  Precision  UAV IoU   Dice   | UAV IoU(img) mIoU(img) | absent frames w/ pred")
        for k, label in (("all", "all   (V2 raw mask)"), ("track", "track (tracker-kept)")):
            m = agg["pixel"][k]
            print(f"    {label:22s} {m['recall']:.4f}   {m['precision']:.4f}   {m['uav_iou']:.4f}  {m['dice']:.4f} "
                  f"|   {m['uav_iou_per_image']:.4f}      {m['miou_per_image']:.4f}  | "
                  f"{m['absent_frames_with_pred']:,}/{m['absent_frames']:,}")
        if agg["pixel"]["frames_without_mask"]:
            print(f"    (마스크 파일 없는 프레임 {agg['pixel']['frames_without_mask']:,}장은 픽셀 지표에서 제외 — evaluate.py와 동일)")
        print(f"  [BOX, 참고]")
    print(f"  [sequence mean]  Success AUC {sm['success_auc']:.4f} | Precision@20px {sm['precision_20px']:.4f} "
          f"| NormP@0.20 {sm['norm_precision_0.20']:.4f} | NormP AUC {sm['norm_precision_auc']:.4f} "
          f"| SA {sm['state_accuracy']:.4f}")
    print(f"  [frame pooled ]  Success AUC {fp['success_auc']:.4f} | Precision@20px {fp['precision_20px']:.4f} "
          f"| NormP@0.20 {fp['norm_precision_0.20']:.4f} | NormP AUC {fp['norm_precision_auc']:.4f} "
          f"| SA {fp['state_accuracy']:.4f}")
    print(f"  present frames output rate {fp['present_frame_output_rate']:.4f} "
          f"| absent frames FP rate {fp['absent_frame_fp_rate']:.4f} "
          f"({fp['n_present']:,} present / {fp['n_absent']:,} absent)")
    print(f"  post-process (components + tracker, CPU): {agg['postprocess_ms_per_frame']:.3f} ms/frame")
    print(f"  status: {dict(status_total)}")
    print(f"  → {out_dir}")

    run_info = {
        "args": vars(args),
        "tracker_config": tcfg.to_dict(),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir) if do_pixel else None,
        "gt_dir": str(gt_dir),
        "cache_dir": str(cache_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    per_seq_out = {k: dict(v["summary"]) for k, v in seq_results.items()}
    for k in per_seq_out:
        if k in seq_pixel:
            per_seq_out[k]["pixel"] = seq_pixel[k]
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_sequence": per_seq_out}, f, indent=2, ensure_ascii=False)

    if seq_pixel:
        pcols = ["recall", "precision", "uav_iou", "dice", "uav_iou_per_image", "miou_per_image",
                 "absent_frames", "absent_frames_with_pred", "n_images"]
        with open(out_dir / "pixel_per_sequence.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sequence"] + [f"all_{c}" for c in pcols] + [f"track_{c}" for c in pcols])
            for rel, d in seq_pixel.items():
                w.writerow([rel] + [d["all"][c] for c in pcols] + [d["track"][c] for c in pcols])

    cols = ["success_auc", "precision_20px", "norm_precision_0.20", "norm_precision_auc",
            "state_accuracy", "mean_iou_present", "present_frame_output_rate", "absent_frame_fp_rate",
            "n_frames", "n_present", "n_absent"]
    with open(out_dir / "per_sequence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence"] + cols)
        for rel, r in seq_results.items():
            w.writerow([rel] + [r["summary"][c] for c in cols])
    return agg


if __name__ == "__main__":
    main()
