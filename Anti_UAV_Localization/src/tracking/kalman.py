"""
등속(constant velocity) Kalman Filter — 박스 [cx, cy, w, h] 추적용.

상태 x = [cx, cy, w, h, vcx, vcy, vw, vh]  (원본 해상도 픽셀 좌표)
관측 z = [cx, cy, w, h]

잡음 크기는 ByteTrack/DeepSORT와 같이 객체 크기에 비례시키되,
DeepSORT가 높이 h를 쓰는 것과 달리 드론은 가로로 긴 경우가 많아 s = max(w, h)를 쓴다.
아주 작은 드론에서 공분산이 0에 가까워지지 않도록 s에 하한(min_size)을 둔다.
"""

import numpy as np


class BoxKalmanFilter:
    ndim = 4

    def __init__(
        self,
        std_weight_position: float = 1.0 / 20,
        std_weight_velocity: float = 1.0 / 160,
        min_size: float = 8.0,
    ):
        self.wp = std_weight_position
        self.wv = std_weight_velocity
        self.min_size = min_size

        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)

    def _size(self, box_cxcywh: np.ndarray) -> float:
        return max(float(box_cxcywh[2]), float(box_cxcywh[3]), self.min_size)

    def initiate(self, z: np.ndarray):
        """z: [cx, cy, w, h] → (mean(8,), cov(8,8))"""
        mean = np.r_[z, np.zeros(4)]
        s = self._size(z)
        std = np.r_[[2 * self.wp * s] * 4, [10 * self.wv * s] * 4]
        return mean, np.diag(std ** 2)

    def predict(self, mean: np.ndarray, cov: np.ndarray):
        s = self._size(mean[:4])
        std = np.r_[[self.wp * s] * 4, [self.wv * s] * 4]
        Q = np.diag(std ** 2)
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + Q
        # 박스 크기가 음수로 발산하지 않도록 하한
        mean[2] = max(mean[2], 1.0)
        mean[3] = max(mean[3], 1.0)
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray):
        s = self._size(mean[:4])
        R = np.diag(np.array([self.wp * s] * 4) ** 2)
        return self.H @ mean, self.H @ cov @ self.H.T + R

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray):
        proj_mean, proj_cov = self.project(mean, cov)
        K = np.linalg.solve(proj_cov, (cov @ self.H.T).T).T  # cov H^T S^-1
        mean = mean + K @ (z - proj_mean)
        cov = cov - K @ proj_cov @ K.T
        return mean, cov

    def gating_distance_center(self, mean: np.ndarray, cov: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """중심 좌표 (N,2)에 대한 제곱 Mahalanobis 거리 (자유도 2)."""
        proj_mean, proj_cov = self.project(mean, cov)
        S = proj_cov[:2, :2]
        d = centers - proj_mean[:2]
        return np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d)

    def gating_distance_box(self, mean: np.ndarray, cov: np.ndarray, boxes_cxcywh: np.ndarray) -> np.ndarray:
        """[cx, cy, w, h] (N,4)에 대한 제곱 Mahalanobis 거리 (자유도 4).

        위치뿐 아니라 크기 연속성까지 보므로, 궤적이 같고 크기만 다른 물체
        (예: 낙하산에 매달린 드론에서 낙하산)를 구분하는 데 쓸 수 있다.
        """
        proj_mean, proj_cov = self.project(mean, cov)
        d = boxes_cxcywh - proj_mean
        return np.einsum("ni,ij,nj->n", d, np.linalg.inv(proj_cov), d)


def xywh_to_cxcywh(b: np.ndarray) -> np.ndarray:
    return np.array([b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, b[2], b[3]], dtype=np.float64)


def cxcywh_to_xywh(b: np.ndarray) -> np.ndarray:
    return np.array([b[0] - b[2] / 2.0, b[1] - b[3] / 2.0, b[2], b[3]], dtype=np.float64)
