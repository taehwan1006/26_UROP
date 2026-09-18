"""세그멘테이션 결과 기반 단일 표적 추적 모듈 (torch 비의존: numpy + cv2)."""

from .detections import Detection, prob_to_detections, q_thresh
from .kalman import BoxKalmanFilter
from .pixel_metrics import PixelMetrics, build_or_load_gt_masks
from .prob_cache import SparseProbReader, SparseProbWriter
from .sot_metrics import aggregate, evaluate_sequence, load_gt, save_pred
from .tracker import FrameOutput, SingleTargetTracker, TrackerConfig
