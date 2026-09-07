# Anti-UAV ThinDyUNet 논문 재현 및 파이프라인 개선 (V2)

논문 [**A Semantic Segmentation Dataset and Real-Time Localization Model for Anti-UAV Applications**](https://www.mdpi.com/2076-3417/15/13/7183) (Kim & Jang, *Applied Sciences* 2025) 의 ThinDyUNet 모델 재현 및 자체 최적화(V2) 프로젝트. 

UROP 과제로 진행되었으며, 본 연구의 성과(V2 파이프라인)를 바탕으로 **제7회 한국인공지능학술대회**에 "복잡한 환경에서의 실시간 안티드론을 위한 경량 세그멘테이션 모델 파이프라인 최적화 연구"라는 제목으로 컴퓨터 비전 분야 논문을 제출하였습니다. 

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

### 3) 🚀 실전 장애물/가림 환경 평가: DUT-Anti-UAV 전체 테스트셋 (`test_dut`)

나뭇가지, 전선, 건물 등 복잡한 배경 클러터와 가림 현상(Occlusion)이 극심한 **전체 DUT-Anti-UAV 테스트셋(24,804장, video01~ 하위 시퀀스 계층 완벽 연동)**에 대한 파인튜닝(학습) 및 최종 정량 평가 결과입니다. V2 최적화를 통해 드론 미탐률을 획기적으로 개선했습니다.

| 평가지표 (Metrics) | 사전학습 베이스라인 (파인튜닝 전) | **V2 파인튜닝 최종 결과** | **성능 향상폭** |
| :--- | :--- | :--- | :--- |
| **Recall (탐지율)** | 0.2114 (21.1%) | **0.7915 (79.1%)** |  **+58.0%p (약 3.7배 상승)** |
| **Precision (정밀도)** | 0.3342 (33.4%) | **0.5492 (54.9%)** |  **+21.5%p 상승** |
| **UAV IoU (드론 영역)** | 0.1487 (14.8%) | **0.4798 (47.9%)** |  **+33.1%p (약 3.2배 상승)** |
| **mIoU (per-image avg)** | 0.5934 (59.3%) | **0.7887 (78.8%)** |  **+19.5%p 상승** |
| **Avg Inference (속도)** | 19.96 ms (약 50 FPS) | **14.58 ms (약 68 FPS)** |  **지연시간 약 27% 단축 (처리량 36% 향상)** |

> **개선 분석 및 한계점:** 
> * **UAV IoU 대표 지표 승격:** 빈 프레임(-100) 예외 처리로 인해 Background IoU가 1.0에 수렴하여 전체 mIoU가 과대 계상(78.8%)되는 통계적 착시를 방지하고자, 순수 객체 분할 품질을 나타내는 **UAV-class IoU (47.9%)**를 핵심 평가 지표로 채택했습니다.
> * **탐지율 vs 정밀도 병목 (Trade-off):** SAM 2.1 마스크 정제와 BCEDiceLoss를 결합한 파인튜닝으로 극한 환경에서의 드론 미탐률을 획기적으로 개선했습니다(Recall 79.1%). 다만 클러터 환경 특성상 Precision은 54.9% (F1 Score ≈ 0.648)로 오탐(False Positive) 한계가 여전히 존재하나, 방공 시스템에서 가장 치명적인 미탐(False Negative)을 우선적으로 방어해 냈습니다.

---

## 환경

- **Local (Test용):** Windows 11, RTX 4070 Super (12 GB VRAM)
- **Server (V2 학습 및 Inference 측정용):** Ubuntu Linux, RTX A5000 (24 GB VRAM)
- Python 3.12, PyTorch (CUDA 12.4 / 13.2)

> **추론 속도(Inference) 측정 조건:** Input 해상도 512x512, Batch Size 1, AMP (FP16) 적용, Warm-up 100 iter 제외 후 24,804장 반복 측정 기준 (RTX A5000).

```bash
pip install -r Anti_UAV_Localization/requirements.txt
pip install opencv-python-headless  # 학회 논문용 고화질 시각화를 위한 cv2 모듈
```


---

## 데이터셋 및 SAM 2.1 마스크 변환 파이프라인

논문 저자 공개 데이터셋 및 DUT-Anti-UAV 전체 시퀀스(나무/전선 등 장애물 환경) 처리:

```bash
python Anti_UAV_Localization/scripts/extract_data.py
```


### 💡 DUT-Anti-UAV 마스크 변환 스크립트 (`maskconv.py`) 특징

* **계층 구조 유지:** `video01`, `video02` 등 하위 시퀀스 폴더 구조를 `data/raw/masks/test_dut/` 아래에 그대로 보존.
  
* **-100 드론 부재 프레임 예외 처리:** `_gt.txt` 내에 `-100`으로 기록된 빈 프레임의 왜곡을 방지하기 위해 SAM 2.1 추론을 생략하고 깨끗한 빈 검은색 마스크로 정제.

* **표준 접미사 적용:** `0001_mask.png` 형태로 저장하여 `uav_dataset.py`와 완벽 1:1 매칭.

---

## 데이터셋 구성 (`data.sources`)

학습/평가에 쓸 데이터셋은 **config에서만 선언**하며, 스크립트는 수정하지 않는다. `sources` 아래에 split 이름별로 소스를 나열하면 여러 개일 때 `ConcatDataset`으로 자동 결합된다.

```yaml
data:
  images_root: "data/raw/images"   # sources 미정의 split의 폴백 루트
  masks_root: "data/raw/masks"
  img_size: [512, 512]
  num_workers: 4

  sources:
    train:                                     # 소스 2개 → 자동 결합
      - name: dut_tracking
        images_dir: "data/raw/images/train_dut"
        masks_dir: "data/raw/masks/train_dut"
        stride: 1
      - name: dut_detection
        images_dir: "data/detection_raw/images/train"
        masks_dir: "data/detection_raw/masks/train"

    test_dut:
      - name: dut_hard_seqs
        images_dir: "data/raw/images/test_dut"
        masks_dir: "data/raw/masks/test_dut"
        include_sequences: ["video0*"]         # 시퀀스 일부만
        exclude_sequences: ["video07"]
        max_samples: 2000                      # 균등 간격 추출
```

| 소스 옵션 | 설명 |
| --- | --- |
| `stride` | 시퀀스별 프레임 샘플링 간격 (1=전체, 10=10%) |
| `include_sequences` | 사용할 시퀀스 glob 패턴. 예: `video01`, `video0*`, `*/visible` |
| `exclude_sequences` | 제외할 시퀀스 패턴 (include보다 우선) |
| `max_samples` | 최종 샘플 수 상한 (전체에서 균등 간격으로 추림) |

**새 데이터셋 추가**는 `images_dir` / `masks_dir` 블록 하나만 넣으면 되고, **기존 config는 `sources` 없이도 폴백**(`images_root` + split 폴더명)으로 그대로 동작한다.

시퀀스 선택은 CLI로도 덮어쓸 수 있어, config를 고치지 않고 부분 평가를 돌릴 수 있다:

```bash
# 특정 시퀀스만 평가
python Anti_UAV_Localization/src/evaluate.py --split test_dut \
    --include-sequences video01 "video0[2-5]"

# 전선/철탑 시퀀스 제외하고 2000장만
python Anti_UAV_Localization/src/evaluate.py --split test_dut \
    --exclude-sequences video07 --max-samples 2000
```

---

## 모델: ThinDyUNet (1.37M params)

논문 Section 4 / Figure 6 기반 구현.

* **U-Net 기반** encoder-decoder (고정 64채널)
* **Dynamic Convolution** (encoder)
* **N-fold 효율 구현:** `conv(x, Σαᵢ·Kᵢ) = Σαᵢ·conv(x, Kᵢ)` 항등식 활용하여 cuDNN에 친화적인 단일 conv 연산으로 최적화.

모델도 config의 `model.name`으로 선택하므로, **ablation은 코드 수정 없이 config 한 줄**로 전환된다.

```yaml
model:
  name: "ThinDyUNet"   # 1.37M — dynamic conv
  # name: "ThinUNet"   # 14.78M — regular conv (ablation)
  in_channels: 3
  n_classes: 1
  base_ch: 64
  n_kernels: 3         # ThinDyUNet 전용
```

---

## 학습 (V2 파이프라인)

논문 설정을 기반으로 하되, **안정성과 속도를 향상한 Ver2 세팅** 적용:

| 항목 | 값 | 비고 |
| --- | --- | --- |
| Optimizer | **AdamW** | (개선) weight_decay=1e-4 |
| Scheduler | **ReduceLROnPlateau** | (개선) factor=0.15, patience=4 |
| Batch Size | 8 | (개선) A5000 24GB 활용 |
| Loss | **BCEDiceLoss** | (개선) 작은 드론 객체 탐지율 대폭 상향 |
| AMP (FP16) | **True** | (개선) 학습 속도 30% 향상 (A5000 기준) |

학습·재개·파인튜닝이 **`train_full.py` 하나로 통합**되어 있다.

```bash
# 처음부터 학습
python Anti_UAV_Localization/src/train_full.py \
    --config Anti_UAV_Localization/configs/train_config_full_dut.yaml

# 중단 지점부터 재개 (optimizer/scheduler 상태까지 복원)
python Anti_UAV_Localization/src/train_full.py \
    --resume Anti_UAV_Localization/checkpoints/full/last_model.pth

# 사전학습 가중치로 파인튜닝 (가중치만 로드, optimizer는 새 LR로 재초기화)
python Anti_UAV_Localization/src/train_full.py \
    --config Anti_UAV_Localization/configs/train_config_full_dut.yaml \
    --finetune Anti_UAV_Localization/checkpoints/full/best_model.pth

# config에 정의한 다른 split 조합으로 학습
python Anti_UAV_Localization/src/train_full.py \
    --train-split train_dut --val-split val_dut
```

| 인자 | 설명 |
| --- | --- |
| `--resume` | 가중치 + optimizer/scheduler 복원 후 이어서 학습 |
| `--finetune` | 가중치만 로드하고 optimizer는 재초기화 (`--resume`과 상호배타) |
| `--train-split` / `--val-split` | `data.sources`의 어떤 split을 쓸지 선택 (기본 `train` / `val`) |

---

## 평가 및 실전 데이터셋 연동

### 학술대회 논문용 고화질 시각화 기능 포함

V2 파이프라인에는 정답(Green), 예측(Red), 일치(Yellow) 영역을 투명도(Alpha) 블렌딩으로 비교해 주는 논문용 고화질 Figure 추출 모듈이 포함되어 있습니다.


```bash
# 전체 DUT-Anti-UAV 테스트셋 평가 및 시각화 이미지 50장 추출
python Anti_UAV_Localization/src/evaluate.py \
    --config Anti_UAV_Localization/configs/train_config_full_dut.yaml \
    --checkpoint Anti_UAV_Localization/checkpoints/full/best_model.pth \
    --split test_dut \
    --visualize \
    --n_vis 50

# 특정 시퀀스만 빠르게 평가 (오탐 사례 분석용)
python Anti_UAV_Localization/src/evaluate.py \
    --checkpoint Anti_UAV_Localization/checkpoints/full/best_model.pth \
    --split test_dut \
    --include-sequences video01 video02 \
    --stride 10
```

`--stride` / `--include-sequences` / `--exclude-sequences` / `--max-samples`는 `evaluate.py`, `evaluate-paper.py`, `scripts/visualize_only.py`, `scripts/visualize_compare.py`에서 공통으로 쓸 수 있다.

---

## 실험 노트 (트러블슈팅 및 개선 사항)

| 시도 | 결과 및 원인 |
| --- | --- |
| **-100 좌표 마스크 예외 처리** | **데이터 정제 및 Recall 정상화** — 초기 전체 데이터셋 평가 시 Recall이 0에 수렴했던 원인이 -100 값을 SAM이 받아 왜곡되었기 때문임을 규명. 빈 마스크 예외 처리를 통해 문제를 완벽히 해결함. |
| **재귀적 데이터셋 로더 (`rglob`)** | **하위 폴더 계층 구조 지원** — 시퀀스 폴더 순회 로직을 적용하여 `test_dut/video01/` 형태의 복잡한 디렉토리 구조를 코드 수정 없이 완벽 연동. |
| **Loss 함수 변경 (BCEDiceLoss)** | **Recall 수직 상승** — 클래스 불균형(배경 99%) 문제가 해결되며 정밀도와 재현율 모두 상승함. |
| **학술대회 제출용 오버레이 구현** | **정성적 평가 신뢰도** — 학계 표준인 RGB 컬러 오버레이 시각화 유틸리티(`visualization.py`)를 자체 개발하여 논문 삽입용 시각 자료 품질 극대화. |
| **데이터셋 config화 (`data.sources`)** | **실험마다 스크립트를 고치던 문제 해소** — 데이터셋 경로가 학습 스크립트에 하드코딩돼 있어 소스 추가나 시퀀스 일부 선택에 코드 수정이 필요했음. config 선언 방식으로 바꾸고 `train_full_finetuning.py`를 `--finetune` 플래그로 통합. |

---

## 프로젝트 구조

```
Anti_UAV_Localization/
├── configs/                    # 학습/평가 설정 (data.sources 스키마)
├── scripts/
│   ├── extract_data.py         # tar.gz → data/raw/
│   ├── visualize_only.py       # 예측 그리드 시각화
│   └── visualize_compare.py    # 두 모델 예측 비교
└── src/
    ├── dataset/
    │   ├── uav_dataset.py      # 이미지/마스크 로딩 + 시퀀스 필터
    │   └── builder.py          # config sources → (Concat)Dataset
    ├── models/
    │   ├── __init__.py         # build_model — model.name으로 선택
    │   ├── thin_dy_unet.py     # ThinDyUNet (1.37M, dynamic conv)
    │   └── thin_unet.py        # ablation: regular conv (14.78M)
    ├── utils/
    │   ├── losses.py           # DiceLoss / BCEDiceLoss / build_loss
    │   ├── metrics.py          # IoU 다중 정의
    │   ├── checkpoint.py       # 저장 / 가중치 로드 / resume
    │   ├── config.py           # YAML 로딩
    │   ├── image.py            # ImageNet 정규화 상수 + denorm
    │   └── visualization.py    # 논문용 오버레이 Figure
    ├── engine.py               # 학습/검증 루프 (AMP·accumulation)
    ├── train_full.py           # V2 메인 학습 (--resume / --finetune)
    ├── evaluate.py             # 테스트셋 평가/시각화
    ├── evaluate-paper.py       # 논문 공식 메트릭 정의 재현
    ├── train.py                # V1 baseline (원고 수치 재현용, 보존)
    └── train_amp.py            # V1 AMP 버전 (원고 수치 재현용, 보존)
```

---

## References

* Kim, S.; Jang, K. *A Semantic Segmentation Dataset and Real-Time Localization Model for Anti-UAV Applications.* Applied Sciences 2025, 15, 7183.
* Wang, L. et al. *Temporal Segment Networks for Action Recognition in Videos.* ECCV 2016 — sequence-aware sparse sampling 관련.
* Chen, Y. et al. *Dynamic Convolution: Attention over Convolution Kernels.* CVPR 2020 — dynamic convolution 원조.
