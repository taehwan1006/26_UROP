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

```

---

## 데이터셋 및 SAM 2.1 마스크 변환 파이프라인

논문 저자 공개 데이터셋 및 DUT-Anti-UAV 전체 시퀀스(나무/전선 등 장애물 환경) 처리:

* `UAVSemanticSegmentationInput.tar.gz` — 이미지 (RGB + IR), 605,045장
* `UAVSemanticSegmentationLabels.tar.gz` — 바이너리 마스크
* **DUT-Anti-UAV 시퀀스 데이터 (`Anti-UAV-Tracking-V0` 및 `_gt.txt`)**

압축 해제 및 마스크 생성:

```bash
python Anti_UAV_Localization/scripts/extract_data.py

```

### 💡 DUT-Anti-UAV 마스크 변환 스크립트 (`maskconv.py`) 특징

* **계층 구조 유지:** `video01`, `video02` 등 하위 시퀀스 폴더 구조를 `data/raw/masks/test_dut/` 아래에 그대로 보존.
* **$-100$ 드론 부재 프레임 예외 처리:** `_gt.txt` 내에 `-100 -100 -100 -100`으로 기록된 프레임(드론이 화면 밖으로 나갔거나 가려져 추적 불가능한 구간)은 SAM 2.1 추론을 생략하고 깨끗한 빈 검은색 마스크(`0`)로 자동 저장하여 데이터 오염 방지.
* **표준 접미사 적용:** `0001_mask.png` 형태로 저장하여 `uav_dataset.py`와 완벽 1:1 매칭.

---

## 모델: ThinDyUNet (1.37M params)

논문 Section 4 / Figure 6 기반 구현.

* **U-Net 기반** encoder-decoder
* **고정 64채널** (모든 layer)
* **Dynamic Convolution** (encoder)
* N개 커널 후보 + SE-style attention으로 입력 의존적 가중합
* `K_dyn = Σ αᵢ·Kᵢ`, `Y = LeakyReLU(GroupNorm(K_dyn * X))`
* **N-fold 효율 구현**
* `conv(x, Σαᵢ·Kᵢ) = Σαᵢ·conv(x, Kᵢ)` 항등식 활용
* 단일 conv (output channels = N×out_ch) 후 attention 가중합
* cuDNN이 잘 최적화하는 일반 conv 한 번으로 처리

```python
out = F.conv2d(x, weight, padding=padding)         # (B, N*C_out, H, W)
out = out.view(B, N, C_out, H, W)
out = (alpha.view(B, N, 1, 1, 1) * out).sum(dim=1) # (B, C_out, H, W)

```

---

## 학습 (V2 파이프라인)

논문 Section 5.2 설정을 기반으로 하되, **안정성과 속도를 극대화한 V2 세팅** 적용 (서버 기준):

| 항목 | 값 | 비고 |
| --- | --- | --- |
| Input size | 512×512 |  |
| Optimizer | **AdamW** | (개선) weight_decay=1e-4 |
| Learning rate | 1e-4 |  |
| Scheduler | **ReduceLROnPlateau** | (개선) factor=0.15, patience=4 |
| Batch Size | 8 | (개선) A5000 24GB 활용 |
| Gradient accumulation | 3 | effective batch **24** 유지 |
| Loss | **BCEDiceLoss** | (개선) 작은 드론 객체 탐지율 대폭 상향 |
| AMP (FP16) | **True** | (개선) 학습 속도 30% 향상 |

```bash
# V2 파이프라인 (추천)
python Anti_UAV_Localization/src/train_full.py \
    --config Anti_UAV_Localization/configs/train_config_full.yaml

```

**학습 시간 비교:**

* V1 (RTX 4070S, stride=10, FP32): ~1,900 s/epoch
* **V2 (RTX A5000, stride=10, AMP FP16): ~1,240 s/epoch (약 35% 가속)**

---

## 평가 및 실전 데이터셋 연동

```bash
# 표준 평가 (threshold=0.5)
python Anti_UAV_Localization/src/evaluate.py \
    --config Anti_UAV_Localization/configs/train_config_full.yaml \
    --checkpoint Anti_UAV_Localization/checkpoints/full/best_model.pth \
    --split test

# 전체 DUT-Anti-UAV 테스트셋 평가 (비디오 계층 구조 및 시각화 지원)
python Anti_UAV_Localization/src/evaluate.py \
    --config Anti_UAV_Localization/configs/train_config_full.yaml \
    --checkpoint Anti_UAV_Localization/checkpoints/last_model.pth \
    --split test_dut --visualize --n_vis 50

```

---

## 실험 노트 (트러블슈팅 및 개선 사항)

| 시도 | 결과 및 원인 |
| --- | --- |
| **$-100$ 좌표 마스크 예외 처리** | **데이터 정제 및 Recall 정상화** — 초기 전체 데이터셋 평가 시 Recall이 0에 수렴했던 원인이 $-100$ 값을 SAM 2가 받아 생성이 왜곡되었기 때문임을 규명. 빈 마스크 예외 처리를 도입하여 전체 24,804장 데이터셋의 정량 평가 수치(Recall 21.1%)를 정상 복구함. |
| **재귀적 데이터셋 로더 (`rglob`)** | **하위 폴더 계층 구조 지원** — `uav_dataset.py`에 `rglob("*")` 및 시퀀스 폴더 순회 로직을 적용하여 `test_dut/video01/` 형태의 복잡한 디렉토리 구조를 코드 수정 없이 완벽하게 연동. |
| **AMP fp16 적용** | **성능 폭발 (RTX A5000)** — 기존 4070S(V1)에서는 효과가 미미했으나, A5000에서 AMP 적용 시 VRAM 절약은 물론 에포크당 소요 시간이 1,900초에서 1,240초로 약 **35% 가속**됨. |
| **Loss 함수 변경** | **Recall 수직 상승** — 순수 BCE Loss에서 `BCEDiceLoss`로 변경 후, 클래스 불균형(배경 99%) 문제가 해결되며 정밀도(P)와 재현율(R)이 모두 94% 후반대로 수렴. |

---

## 프로젝트 구조

```
Anti_UAV_Localization/
├── configs/
│   ├── train_config.yaml               # stride=20 (5%)
│   ├── train_config_stride10.yaml      # stride=10 (10%) V1
│   └── train_config_full.yaml          # stride=10 (10%) V2 파이프라인 메인
├── scripts/
│   ├── extract_data.py                 # tar.gz → data/raw/
│   ├── maskconv.py                     # SAM 2.1 기반 전체 DUT-Anti-UAV 마스크 변환 (-100 예외처리 포함)
│   ├── benchmark_batch.py              # batch size별 속도/메모리 측정
│   └── visualize_compare.py            # 두 모델 비교 시각화
├── src/
│   ├── dataset/uav_dataset.py          # rglob 기반 계층형 시퀀스 stride 샘플링
│   ├── models/
│   │   ├── thin_dy_unet.py             # ThinDyUNet (1.37M)
│   │   └── thin_unet.py                # ablation: regular conv (14.78M)
│   ├── utils/
│   │   ├── metrics.py                  # IoU 다중 정의
│   │   └── visualization.py
│   ├── train_full.py                   # V2: AMP + AdamW + BCEDice + 스케줄러 메인 학습 스크립트
│   └── evaluate.py                     # 테스트셋 및 DUT-Anti-UAV 평가/시각화
├── results/                            # 시각화 PNG
├── data/                               # gitignore (raw/)
└── checkpoints/                        # gitignore

```

---

## References

* Kim, S.; Jang, K. *A Semantic Segmentation Dataset and Real-Time Localization Model for Anti-UAV Applications.* Applied Sciences 2025, 15, 7183.
* Wang, L. et al. *Temporal Segment Networks for Action Recognition in Videos.* ECCV 2016 — sequence-aware sparse sampling 관련.
* Chen, Y. et al. *Dynamic Convolution: Attention over Convolution Kernels.* CVPR 2020 — dynamic convolution 원조.

```

```
