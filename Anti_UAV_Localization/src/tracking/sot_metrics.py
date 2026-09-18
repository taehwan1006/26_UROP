"""
단일 객체 추적(SOT) 평가지표.

박스 형식: [x, y, w, h] (원본 해상도). 드론 부재 프레임 GT는 -100 -100 -100 -100.
예측 없음(트래커가 "드론 없음"으로 판단)은 NaN 행으로 표현한다.

[1] OPE 지표 (OTB/LaSOT/pysot 계열 정의) — GT가 존재하는 프레임만 분모로 사용
    - Success      : IoU > t 인 프레임 비율, t ∈ linspace(0, 1, 21). AUC = 곡선 평균
    - Precision    : 중심 오차 ≤ t px 비율, t ∈ 0..50. 대표값 P@20px
    - Norm. Prec.  : GT 크기로 정규화한 중심 오차 ≤ t 비율, t ∈ linspace(0, 0.5, 51).
                     대표값 두 가지를 모두 기록: P_norm@0.20 (pysot 구현) / AUC over [0, 0.5] (LaSOT 논문)
    - GT 존재 & 예측 없음 → IoU = 0, 중심 오차 = ∞ (실패)
    ※ 부재 프레임(-100)을 어떻게 처리하는지는 DUT-Anti-UAV 공식 평가 코드로 아직 확인하지 못했다.
      여기서는 "GT 존재 프레임만 평가"로 명시적으로 정의하고, 부재 판단 능력은 [2]로 따로 본다.

[2] 존재 판단 포함 지표
    - State Accuracy (SA) : 프레임 평균 of { GT 존재: IoU,  GT 부재: 예측 없음이면 1 아니면 0 }
      (Anti-UAV Challenge의 state accuracy 기본 항. 챌린지 버전별 penalty 항은 포함하지 않음)
    - 부재 프레임 오탐률 : GT 부재 프레임 중 박스를 출력한 비율
    - 존재 프레임 출력률 : GT 존재 프레임 중 박스를 출력한 비율

집계: 시퀀스별 곡선을 계산한 뒤 시퀀스 평균(pysot 방식, 주 지표)과
      전 프레임 풀링(보조)을 모두 보고한다.
"""

from typing import Dict, List

import numpy as np

SUCCESS_T = np.linspace(0.0, 1.0, 21)
PRECISION_T = np.arange(0, 51, dtype=np.float64)
NORM_PRECISION_T = np.linspace(0.0, 0.5, 51)


def valid_gt_mask(gt: np.ndarray) -> np.ndarray:
    return (gt[:, 0] != -100) & (gt[:, 2] > 0) & (gt[:, 3] > 0)


def has_pred_mask(pred: np.ndarray) -> np.ndarray:
    return ~np.isnan(pred).any(axis=1)


def box_iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """행 단위 IoU. a, b: (N, 4)."""
    ax1, ay1 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def per_frame_errors(gt: np.ndarray, pred: np.ndarray) -> Dict[str, np.ndarray]:
    """GT 존재 프레임에 대한 IoU / 중심 오차 / 정규화 중심 오차, 그리고 마스크들."""
    assert gt.shape == pred.shape, (gt.shape, pred.shape)
    v = valid_gt_mask(gt)
    p = has_pred_mask(pred)
    n = len(gt)

    iou = np.zeros(n)
    cerr = np.full(n, np.inf)
    nerr = np.full(n, np.inf)

    both = v & p
    if np.any(both):
        g, q = gt[both], pred[both]
        iou[both] = box_iou_xywh(g, q)
        gc = g[:, :2] + g[:, 2:] / 2.0
        qc = q[:, :2] + q[:, 2:] / 2.0
        cerr[both] = np.linalg.norm(gc - qc, axis=1)
        nerr[both] = np.linalg.norm((gc - qc) / g[:, 2:], axis=1)
    return {"valid": v, "has_pred": p, "iou": iou, "cerr": cerr, "nerr": nerr}


def curves(iou: np.ndarray, cerr: np.ndarray, nerr: np.ndarray) -> Dict[str, np.ndarray]:
    """GT 존재 프레임 배열로부터 곡선 계산."""
    if len(iou) == 0:
        nan = lambda t: np.full(len(t), np.nan)
        return {"success": nan(SUCCESS_T), "precision": nan(PRECISION_T), "norm_precision": nan(NORM_PRECISION_T)}
    return {
        "success": (iou[None, :] > SUCCESS_T[:, None]).mean(axis=1),
        "precision": (cerr[None, :] <= PRECISION_T[:, None]).mean(axis=1),
        "norm_precision": (nerr[None, :] <= NORM_PRECISION_T[:, None]).mean(axis=1),
    }


def summarize_curves(c: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "success_auc": float(np.mean(c["success"])),
        "precision_20px": float(c["precision"][20]),
        "norm_precision_0.20": float(c["norm_precision"][20]),
        "norm_precision_auc": float(np.mean(c["norm_precision"])),
    }


def presence_metrics(e: Dict[str, np.ndarray]) -> Dict[str, float]:
    v, p, iou = e["valid"], e["has_pred"], e["iou"]
    sa_terms = np.where(v, iou, (~p).astype(np.float64))
    n_valid, n_absent = int(v.sum()), int((~v).sum())
    return {
        "state_accuracy": float(sa_terms.mean()) if len(v) else float("nan"),
        "absent_frame_fp_rate": float((p & ~v).sum() / n_absent) if n_absent else float("nan"),
        "present_frame_output_rate": float((p & v).sum() / n_valid) if n_valid else float("nan"),
        "n_frames": int(len(v)),
        "n_present": n_valid,
        "n_absent": n_absent,
    }


def evaluate_sequence(gt: np.ndarray, pred: np.ndarray) -> Dict:
    e = per_frame_errors(gt, pred)
    v = e["valid"]
    c = curves(e["iou"][v], e["cerr"][v], e["nerr"][v])
    out = summarize_curves(c)
    out.update(presence_metrics(e))
    out["mean_iou_present"] = float(e["iou"][v].mean()) if v.any() else float("nan")
    return {"summary": out, "curves": c, "errors": e}


def aggregate(seq_results: Dict[str, Dict]) -> Dict:
    """시퀀스 평균(주) + 프레임 풀링(보조)."""
    names = [k for k, r in seq_results.items() if r["summary"]["n_present"] > 0]
    mean_curves = {
        k: np.mean([seq_results[n]["curves"][k] for n in names], axis=0)
        for k in ("success", "precision", "norm_precision")
    }
    seq_mean = summarize_curves(mean_curves)
    seq_mean["state_accuracy"] = float(np.mean([r["summary"]["state_accuracy"] for r in seq_results.values()]))

    E = {k: np.concatenate([r["errors"][k] for r in seq_results.values()])
         for k in ("valid", "has_pred", "iou", "cerr", "nerr")}
    v = E["valid"]
    pooled = summarize_curves(curves(E["iou"][v], E["cerr"][v], E["nerr"][v]))
    pooled.update(presence_metrics(E))

    return {
        "sequence_mean": seq_mean,
        "frame_pooled": pooled,
        "curves_sequence_mean": {k: val.tolist() for k, val in mean_curves.items()},
        "n_sequences": len(seq_results),
    }


def load_gt(path) -> np.ndarray:
    """DUT-Anti-UAV `videoXX_gt.txt` (공백 구분 x y w h, CRLF 가능)."""
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if not parts:
                continue
            rows.append([float(x) for x in parts[:4]])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def save_pred(path, pred: np.ndarray, scores: np.ndarray, statuses: List[str]) -> None:
    """GT와 같은 x y w h 형식 + score + status. 예측 없음은 -100."""
    with open(path, "w", encoding="utf-8") as f:
        for box, s, st in zip(pred, scores, statuses):
            if np.isnan(box).any():
                f.write(f"-100 -100 -100 -100 {s:.4f} {st}\n")
            else:
                f.write(f"{box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f} {s:.4f} {st}\n")
