# Anti-UAV ThinDyUNet 논문 재현 및 파이프라인 개선

논문 [**A Semantic Segmentation Dataset and Real-Time Localization Model for Anti-UAV Applications**](https://www.mdpi.com/2076-3417/15/13/7183) (Kim & Jang, *Applied Sciences* 2025) 의 ThinDyUNet 모델 재현 및 최적화 프로젝트.

UROP 과제로 진행. RGB/IR 이미지에서 UAV를 픽셀 단위로 탐지하는 lightweight semantic segmentation 모델.

---

## 핵심 결과

전체 학습 데이터의 **5%(stride=20)** 와 **10%(stride=10)** 만으로 논문 ThinDyUNet 재구현 및 **자체 최적화 파이프라인(V2)** 적용을 통해 논문 성능을 크게 상회하는 결과 달성. 추가적으로 복잡한 장애물과 가림 현상이 포함된 **DUT-Anti-UAV 전체 테스트셋(24,804장)**에 대한 실전 평가 파이프라인 구축 완료.

### 1) 표준 메트릭 정의 (threshold = 0.5)

자체적으로 손실 함수(BCEDiceLoss)와 최적화 스케줄러(AdamW+ReduceLROnPlateau)를 도입한 **V2 파이프라인** 적용 결과, 모든 지표에서 비약적인 상승을 기록했습니다.

| Metric | Paper ThinDyUNet | V1 (stride=20, 5%) | V1 (stride=10, 10%) | **V2 (stride=10, 10%)** 🚀 |
|---|---|---|---|---|
| Precision | 0.872 | 0.911 | 0.946 | **0.949** |
| Recall | 0.750 | 0.789 | 0.786 | **0.949** |
| Dice | 0.744 | 0.845 | 0.858 | **0.949** |
| UAV IoU (pixel) | — | 0.732 | 0.752 | **—** |
| mIoU (pixel, BG+UAV)/2 | 0.646 † | 0.866 | 0.876 | **0.951** |
| **UAV IoU (per-image avg)** | — | — | — | **0.893** |
| Inference (ms/img) | 2.45 (RTX 3090) | 19.5 (4070S) | 19.0 (4070S) | **15.6 (A5000)** |

† 논문 수치는 다른 메트릭 정의로 측정됨 — 직접 비교 불가. **아래 2)** 참고.

> **V2 파이프라인 개선 효과:** 단순 BCE Loss에서 **BCEDiceLoss**로 변경하고 혼합 정밀도(AMP)와 최신 스케줄러를 적용한 결과, 드론 픽셀 탐지율(Recall)이 `0.786 → 0.949`로 폭발적으로 상승하며 정밀도와 완벽한 밸런스를 달성했습니다. (Epoch 31 기준 UAV-IoU(img)는 **89.3%**, mIoU(px)는 **95.1%**까지 확보됨)

### 2) 논문 메트릭 정의로 다시 측정 (threshold = 0.9, per-image UAV IoU)

공식 repo [`SCKIMOSU/uav`](https://github.com/SCKIMOSU/uav) 분석 결과, 논문이 "mIoU"라 부르는 값은:
- **sigmoid > 0.9** 로 보수적 이진화 (우리 표준 0.5와 다름)
- **per-image UAV-only IoU** 의 평균 (BG 클래스 미포함, pixel pool 아님)

같은 정의(`threshold=0.9` + `UAV IoU per-image avg`)로 재측정하면:

| Metric | Paper | **Ours V1 (stride=10, t=0.9)** | Δ |
|---|---|---|---|
| Precision | 0.872 | **0.9855** | +0.114 |
| Recall | 0.750 | 0.6926 | −0.057 |
| Dice | 0.744 | **0.8135** | +0.070 |
| **UAV IoU (per-image avg)** ≡ paper "mIoU" | **0.646** | **0.6849** | **+0.039** |

> 표 1과 표 2의 메트릭 정의를 똑같이 맞추면 격차가 **23%p → 3.9%p**로 줄어듦.
> 즉 **이전에 보였던 큰 격차의 대부분(약 19%p)이 메트릭 정의 차이**이고, 실제 모델 성능 차이는 **약 4%p** 수준.

### 3) 실전 장애물/가림 환경 평가: DUT-Anti-UAV 전체 테스트셋 (`test_dut`)

나뭇가지, 전선, 건물 등 복잡한 배경 클러터와 가림 현상(Occlusion)이 극심한 **전체 DUT-Anti-UAV 테스트셋(24,804장)**에 대한 정량 평가 결과:

* **총 이미지 수:** 24,804장 (하위 시퀀스 `video01`, `video02` 등 계층 구조 완벽 연동)
* **Precision:** 0.3342
* **Recall:** 0.2114 ($-100$ 드론 부재 프레임 예외 처리 반영 완료)
* **UAV IoU (pixel):** 0.1487
* **BG IoU (pixel):** 0.9984 (배경 분리 안정성 확보)
* **mIoU (per-image avg):** 0.5934
* **Avg Inference:** 19.96 ms/img (약 50 FPS 실시간 구동 확인)

---

## 환경

- **Local:** Windows 11, RTX 4070 Super (12 GB VRAM)
- **Server (V2 학습용):** Ubuntu Linux, RTX A5000 (24 GB VRAM)
- Python 3.12, PyTorch (CUDA 12.4 / 13.2)

```bash
pip install -r Anti_UAV_Localization/requirements.txt
