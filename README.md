# LightCLR

LightCLR은 실내 자율주행 환경에서 발생하는 빛 반사, 빛 산란, 조도 변화에 강한 차선 인식 모델을 만들기 위한 contrastive learning 기반 실험 프로젝트입니다. DeepRacer 대회에서 실내 조명과 반사광 때문에 차량이 차선을 안정적으로 인식하지 못했던 문제를 해결하기 위해 시작했습니다.

## 프로젝트 개요

차선 인식 모델은 조명 조건이 바뀌면 같은 차선이라도 다른 이미지처럼 인식할 수 있습니다. LightCLR은 같은 원본 이미지에서 만든 clean/noisy image pair를 positive pair로 두고, 서로 다른 원본 이미지에서 나온 pair는 negative pair로 두어 encoder가 조명 노이즈에 덜 민감한 표현을 학습하도록 구성했습니다.

Contrastive pretraining에는 TUSimple과 BDD100K 이미지 데이터를 사용할 수 있도록 구성했습니다. 이후 TuSimple lane label을 이용해 lane detection head를 학습하고, SimCLR로 사전학습한 encoder를 fine-tuning 단계에서 함께 업데이트합니다.

## 학습 원리

1. TUSimple, BDD100K 이미지에서 원본 이미지를 읽습니다.
2. 원본 이미지를 clean view로 변환합니다.
3. 같은 원본 이미지에 brightness, contrast, blur, gaussian noise를 적용해 noisy view를 만듭니다.
4. 같은 원본에서 나온 `(clean, noisy)` pair는 positive pair로 학습합니다.
5. 배치 안의 다른 이미지에서 나온 view들은 negative pair로 학습합니다.
6. NT-Xent loss로 encoder가 조명 변화보다 차선/도로 구조에 집중하도록 사전학습합니다.
7. 사전학습된 encoder를 lane detection model에 로드합니다.
8. fine-tuning 단계에서는 encoder를 freeze하지 않고 낮은 learning rate로 함께 업데이트합니다.
9. lane head는 TuSimple lane point의 normalized x 좌표를 예측합니다.

## 코드 구조

```text
LightCLR/
├── README.md
├── configs/
│   └── lane_det_config.py
├── data/
│   ├── contrastive_dataset.py
│   └── tusimple_dataset.py
├── experiments/
│   └── eval_lane_detection.py
├── models/
│   ├── encoder.py
│   ├── lane_att.py
│   └── lane_head.py
├── training/
│   ├── losses.py
│   ├── train_baseline.py
│   ├── train_contrastive.py
│   └── train_finetune.py
└── utils/
    ├── logger.py
    └── metrics.py
```

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `data/contrastive_dataset.py` | 같은 이미지에서 clean/noisy positive pair 생성 |
| `data/tusimple_dataset.py` | TuSimple annotation 로드 및 lane point target 생성 |
| `training/train_contrastive.py` | noise augmentation 기반 SimCLR pretraining |
| `training/train_baseline.py` | contrastive pretraining 없는 lane detector 학습 |
| `training/train_finetune.py` | SimCLR encoder를 로드해 lane detector fine-tuning |
| `training/losses.py` | NT-Xent loss와 lane point loss |
| `models/encoder.py` | ResNet34 기반 feature encoder |
| `models/lane_head.py` | normalized lane x 좌표 예측 head |

## 실행 방법

의존성 설치:

```bash
pip install -r requirements.txt
```

데이터 경로는 `configs/lane_det_config.py`에서 설정합니다.

```python
CFG = {
    "data_root": "dataset/tusimple",
    "contrastive_roots": [
        "dataset/tusimple/clips",
        "dataset/bdd100k/images/100k/train",
    ],
}
```

Contrastive pretraining:

```bash
python training/train_contrastive.py
```

Baseline lane detector 학습:

```bash
python training/train_baseline.py
```

Contrastive encoder 기반 fine-tuning:

```bash
python training/train_finetune.py
```

평가:

```bash
python experiments/eval_lane_detection.py
```

## 결과

- 빛 반사와 산란으로 생긴 입력 변화에 대해 encoder feature가 덜 흔들립니다.
- noisy image와 clean image가 가까운 embedding으로 정렬됩니다.
- fine-tuning 단계에서 lane label을 이용해 contrastive feature가 실제 차선 검출 task에 맞게 조정됩니다.
- baseline 모델과 SimCLR fine-tuned 모델을 같은 TuSimple validation/test split에서 비교할 수 있습니다.
