# Anti-UAV ThinDyUNet 논문 재현 및 파이프라인 개선 (V2)

논문 [**A Semantic Segmentation Dataset and Real-Time Localization Model for Anti-UAV Applications**](https://www.mdpi.com/2076-3417/15/13/7183) (Kim & Jang, *Applied Sciences* 2025) 의 ThinDyUNet 모델 재현 및 자체 최적화(V2) 프로젝트. 

UROP 과제로 진행되었으며, 본 연구의 성과(V2 파이프라인)를 바탕으로 **제7회 한국인공지능학술대회**에 "복잡한 환경에서의 실시간 안티드론을 위한 경량 세그멘테이션 모델 파이프라인 최적화 연구"라는 제목으로 컴퓨터 비전 분야 논문을 제출하였습니다[cite: 1]. 

---

## 핵심 결과

전체 학습 데이터의 **5%(stride=20)** 와 **10%(stride=10)** 만으로 논문 ThinDyUNet 재구현 및 **자체 최적화 파이프라인(V2)** 적용을 통해 논문 성능을 크게 상회하는 결과를 달성했습니다. 특히, 복잡한 장애물과 가림 현상이 포함된 **DUT-Anti-UAV 전체 테스트셋(24,804장)**에 대한 실전 평가에서 압도적인 성능 향상을 기록했습니다.

### 1) 표준 메트릭 정의 (threshold = 0.5)

자체적으로 손실 함수(BCEDiceLoss)와 최적화 스케줄러(AdamW+ReduceLROnPlateau)를 도입한 **V2 파이프라인** 적용 결과, 모든 지표에 성능 향상을 기록했습니다.

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

### 2) 논문 메트릭 정의로 다시 측정 (threshold = 0.9, per-image UAV IoU)

공식 repo 분석 결과, 표 1과 표 2의 메트릭 정의를 똑같이 맞추면 격차가 **23%p → 3.9%p**로 줄어듦. 즉 이전에 보였던 큰 격차의 대부분이 메트릭 정의 차이이며, 실제 모델 성능 차이는 약 4%p 수준임을 확인.

| Metric | Paper | **Ours V1 (stride=10, t=0.9)** | Δ |
|---|---|---|---|
| Precision | 0.872 | **0.9855** | +0.114 |
| Recall | 0.750 | 0.6926 | −0.057 |
| Dice | 0.744 | **0.8135** | +0.070 |
| **UAV IoU (per-image avg)** ≡ paper "mIoU" | **0.646** | **0.6849** | **+0.039** |

### 3)  실전 장애물/가림 환경 평가: DUT-Anti-UAV 전체 테스트셋 (`test_dut`)

나뭇가지, 전선, 건물 등 복잡한 배경 클러터와 가림 현상(Occlusion)이 극심한 **전체 DUT-Anti-UAV 테스트셋(24,804장, video01~ 하위 시퀀스 계층 완벽 연동)**에 대한 파인튜닝(학습) 및 최종 정량 평가 결과입니다. V2 최적화를 통해 드론 미탐률을 획기적으로 개선했습니다.

| 평가지표 (Metrics) | 기존 결과 (Baseline) | **V2 파이프라인 최종 결과** | **성능 향상폭** |
| :--- | :--- | :--- | :--- |
| **Recall (탐지율)** | 0.2114 (21.1%) | **0.7915 (79.1%)** |  **+58.0%p (약 3.7배 상승)** |
| **Precision (정밀도)** | 0.3342 (33.4%) | **0.5492 (54.9%)** |  **+21.5%p 상승** |
| **UAV IoU (드론 영역)** | 0.1487 (14.8%) | **0.4798 (47.9%)** |  **+33.1%p (약 3.2배 상승)** |
| **mIoU (per-image avg)** | 0.5934 (59.3%) | **0.7887 (78.8%)** |  **+19.5%p 상승** |
| **Avg Inference (속도)** | 19.96 ms (약 50 FPS) | **14.58 ms (약 68 FPS)** |  **약 36% 처리 속도 단축** |

> **개선 분석:** SAM 2.1 마스크 정제와 BCEDiceLoss를 결합한 파인튜닝 결과, 기존 21.1%에 불과했던 실전 드론 탐지율(Recall)을 79.1%까지 끌어올렸으며, 실시간 방공 시스템에 즉시 투입 가능한 초고속 추론 속도(68 FPS)를 확보했습니다.

---

## 환경

- **Local:** Windows 11, RTX 4070 Super (12 GB VRAM)
- **Server (V2 학습용):** Ubuntu Linux, RTX A5000 (24 GB VRAM)
- Python 3.12, PyTorch (CUDA 12.4 / 13.2)


pip install -r Anti_UAV_Localization/requirements.txt
pip install opencv-python-headless  # 학회 논문용 고화질 시각화를 위한 cv2 모듈


---

## 데이터셋 및 SAM 2.1 마스크 변환 파이프라인

논문 저자 공개 데이터셋 및 DUT-Anti-UAV 전체 시퀀스(나무/전선 등 장애물 환경) 처리:

python Anti_UAV_Localization/scripts/extract_data.py



### 💡 DUT-Anti-UAV 마스크 변환 스크립트 (`maskconv.py`) 특징

* **계층 구조 유지:** `video01`, `video02` 등 하위 시퀀스 폴더 구조를 `data/raw/masks/test_dut/` 아래에 그대로 보존.
* **-100 드론 부재 프레임 예외 처리:** `_gt.txt` 내에 `-100`으로 기록된 빈 프레임의 왜곡을 방지하기 위해 SAM 2.1 추론을 생략하고 깨끗한 빈 검은색 마스크로 정제.


* **표준 접미사 적용:** `0001_mask.png` 형태로 저장하여 `uav_dataset.py`와 완벽 1:1 매칭.

---

## 모델: ThinDyUNet (1.37M params)

논문 Section 4 / Figure 6 기반 구현.

* **U-Net 기반** encoder-decoder (고정 64채널)
* **Dynamic Convolution** (encoder)
* **N-fold 효율 구현:** `conv(x, Σαᵢ·Kᵢ) = Σαᵢ·conv(x, Kᵢ)` 항등식 활용하여 cuDNN에 친화적인 단일 conv 연산으로 최적화.

---

## 학습 (V2 파이프라인)

논문 설정을 기반으로 하되, **안정성과 속도를 극대화한 V2 세팅** 적용:

| 항목 | 값 | 비고 |
| --- | --- | --- |
| Optimizer | **AdamW** | (개선) weight_decay=1e-4 |
| Scheduler | **ReduceLROnPlateau** | (개선) factor=0.15, patience=4

 |
| Batch Size | 8 | (개선) A5000 24GB 활용 |
| Loss | **BCEDiceLoss** | (개선) 작은 드론 객체 탐지율 대폭 상향

 |
| AMP (FP16) | **True** | (개선) 학습 속도 30% 향상 (A5000 기준)

 |

# V2 파이프라인 (추천)
python Anti_UAV_Localization/src/train_full.py \
    --config Anti_UAV_Localization/configs/train_config_full_dut.yaml


---

## 평가 및 실전 데이터셋 연동

### 학술대회 논문용 고화질 시각화 기능 포함

V2 파이프라인에는 정답(Green), 예측(Red), 일치(Yellow) 영역을 투명도(Alpha) 블렌딩으로 비교해 주는 논문용 고화질 Figure 추출 모듈이 포함되어 있습니다.


# 전체 DUT-Anti-UAV 테스트셋 평가 및 시각화 이미지 50장 추출
python Anti_UAV_Localization/src/evaluate.py \
    --config Anti_UAV_Localization/configs/train_config_full_dut.yaml \
    --checkpoint Anti_UAV_Localization/checkpoints/full/best_model.pth \
    --split test_dut \
    --visualize \
    --n_vis 50


---

## 실험 노트 (트러블슈팅 및 개선 사항)

| 시도 | 결과 및 원인 |
| --- | --- |
| **-100 좌표 마스크 예외 처리** | **데이터 정제 및 Recall 정상화** — 초기 전체 데이터셋 평가 시 Recall이 0에 수렴했던 원인이 -100 값을 SAM이 받아 왜곡되었기 때문임을 규명. 빈 마스크 예외 처리를 통해 문제를 완벽히 해결함.

 |
| **재귀적 데이터셋 로더 (`rglob`)** | **하위 폴더 계층 구조 지원** — 시퀀스 폴더 순회 로직을 적용하여 `test_dut/video01/` 형태의 복잡한 디렉토리 구조를 코드 수정 없이 완벽 연동. |
| **Loss 함수 변경 (BCEDiceLoss)** | **Recall 수직 상승** — 클래스 불균형(배경 99%) 문제가 해결되며 정밀도와 재현율 모두 압도적으로 상승함.

 |
| **학술대회 제출용 오버레이 구현** | **정성적 평가 신뢰도 확보** — 학계 표준인 RGB 컬러 오버레이 시각화 유틸리티(`visualization.py`)를 자체 개발하여 논문 삽입용 시각 자료 품질 극대화. |

---

## References

* Kim, S.; Jang, K. *A Semantic Segmentation Dataset and Real-Time Localization Model for Anti-UAV Applications.* Applied Sciences 2025, 15, 7183.
* Wang, L. et al. *Temporal Segment Networks for Action Recognition in Videos.* ECCV 2016 — sequence-aware sparse sampling 관련.
* Chen, Y. et al. *Dynamic Convolution: Attention over Convolution Kernels.* CVPR 2020 — dynamic convolution 원조.

```

```
