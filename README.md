# MiniRec: Lightweight Recommendation Framework

경량·Config-first 추천 시스템 실험 프레임워크. 닫힌해(Closed-form) 모델과 SGD 모델을 동일한 인터페이스로 실험하고, 멀티시드 HPO 결과를 논문 수준으로 리포트합니다.

---

## Requirements

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) (권장) 또는 pip

---

## Installation

### uv (권장)

```bash
git clone https://github.com/CREE1116/minirec.git
cd minirec

# 의존성 설치 및 가상환경 자동 생성
uv sync

# 가상환경 활성화
source .venv/bin/activate
```

### pip

```bash
git clone https://github.com/CREE1116/minirec.git
cd minirec

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # 또는 pip install -e .
```

---

## Data Preparation

`data/` 디렉토리에 원시 데이터를 위치시킵니다. `data/` 디렉토리 자체는 gitignore 대상이며, `src/data/`의 소스코드는 별도로 추적됩니다.

```
data/
├── ml100k/
│   └── u.data          # tab-separated: user_id item_id rating timestamp
├── ml1m/
│   └── ratings.dat
├── steam/
│   └── steam.csv
└── yahooR3/
    └── ...
```

각 데이터셋의 경로·구분자·컬럼 정보는 `configs/datasets/` YAML에 명시합니다.

> **주의**: 실험 설정(split_method, valid_ratio 등)을 변경한 경우, 반드시 `data_cache/` 디렉토리를 삭제하고 재실행하세요. 변경 전 캐시가 그대로 사용될 수 있습니다.

---

## Quick Start

### 단일 실험

```python
import minirec

metrics = minirec.run(
    dataset_cfg='configs/datasets/ml-100k.yaml',
    model_cfg='configs/models/ease.yaml',
    output_path='output/ease_ml100k'
)
print(metrics)
# {'NDCG@10': 0.412, 'Recall@10': 0.334, 'Coverage@10': 0.183, ...}
```

결과는 `output_path/` 아래 `metrics.json`과 `config.yaml`로 자동 저장됩니다.

### HPO (멀티시드 베이지안 최적화)

```python
import minirec

hpo_cfg = {
    'direction': 'max',      # HPO objective: evaluation.yaml의 main_metric 자동 사용
    'n_seeds': 3,            # 독립 시드 수
    'patience': 20,          # Optuna 레벨 early stopping
    'params': [
        {'name': 'reg_lambda', 'type': 'float', 'range': '1 1000', 'log': True}
    ]
}

summary = minirec.hporun(
    dataset_cfg='configs/datasets/ml-100k.yaml',
    model_cfg='configs/models/ease.yaml',
    hpo_cfg=hpo_cfg,
    n_trials=30
)
# summary: {'NDCG@20_mean': 0.423, 'NDCG@20_std': 0.008, ...}
```

> HPO objective metric은 `configs/evaluation.yaml`의 `main_metric` / `main_metric_k`를 자동으로 사용합니다. 별도로 지정하지 않습니다.

### 배치 실험 (test_hpo.py)

여러 데이터셋 × 모델 조합을 순차 실행하고 CSV 리포트를 자동 생성합니다.

```bash
python test_hpo.py
```

---

## Project Structure

```
minirec/
├── __init__.py              # 공개 API: run(), hporun()
├── pyproject.toml           # 의존성 (uv)
├── test_hpo.py              # 배치 HPO 실험 스크립트
│
├── configs/
│   ├── evaluation.yaml      # 전역 평가 설정 (metrics, top_k, main_metric)
│   ├── datasets/            # 데이터셋별 전처리·분할 설정
│   │   ├── ml-100k.yaml
│   │   ├── ml-1m.yaml
│   │   ├── steam.yaml
│   │   └── yahoo-r3.yaml
│   └── models/              # 모델별 하이퍼파라미터 기본값
│       ├── ease.yaml
│       ├── mf.yaml
│       └── ...
│
├── src/
│   ├── data/
│   │   └── loader.py        # 데이터 로딩, 분할, 캐싱, 그래프 생성
│   ├── models/
│   │   ├── base.py          # BaseModel (nn.Module)
│   │   ├── ease.py          # EASE
│   │   ├── lira.py          # LIRA
│   │   ├── dlae.py          # DLAE
│   │   ├── puresvd.py       # PureSVD
│   │   ├── ials.py          # iALS
│   │   ├── gf_cf.py         # GF-CF
│   │   ├── ips_lae.py       # IPS-LAE
│   │   ├── ipsdlae.py       # IPS-DLAE
│   │   ├── ipswiener.py     # IPS-Wiener
│   │   ├── pop_ips_wiener.py # Pop-IPS-Wiener
│   │   ├── energy_wiener.py # Energy-Wiener
│   │   ├── hybrid_wiener.py # Hybrid-Wiener
│   │   ├── energy_dlae.py   # Energy-DLAE
│   │   ├── mf.py            # BPR-MF
│   │   └── lightgcn.py      # LightGCN
│   ├── eval/
│   │   └── metrics.py       # 전체 평가 메트릭 계산
│   ├── hpo/
│   │   └── optimizer.py     # BayesianOptimizer (Optuna)
│   ├── trainer.py           # 학습·평가 루프 (Early Stopping 포함)
│   └── utils/
│       ├── config.py        # YAML 로드 및 deep merge
│       ├── seed.py          # 전역 시드 고정
│       └── svd.py           # SVD 캐싱 유틸리티
│
├── data/                    # 원시 데이터 (gitignore)
├── data_cache/              # 전처리 캐시 (gitignore)
└── output/                  # 실험 결과 (gitignore)
```

---

## Configuration

설정은 세 레이어로 구성되며, 우선순위는 `model > dataset > evaluation` 순입니다.

### evaluation.yaml (전역 기본값)

```yaml
method: "full"
top_k: [10, 20, 50]
metrics:
  ["Recall", "NDCG", "HitRate", "Precision",
   "LongTailRecall", "LongTailNDCG", "HeadRecall", "HeadNDCG",
   "Coverage", "LongTailCoverage", "Novelty", "GiniIndex", "PopRatio"]
main_metric: "NDCG"    # HPO objective 및 early stopping 기준
main_metric_k: 20
long_tail_percent: 0.8  # 상위 80% interaction volume을 head로 정의
```

### dataset yaml

```yaml
dataset_name: "ml-100k"
data_path: "data/ml100k/u.data"
separator: "\t"
columns: ["user_id", "item_id", "rating", "timestamp"]
has_header: false

min_user_interactions: 5
min_item_interactions: 0
rating_threshold: 0      # 0이면 필터링 안 함
dedup: true

split_method: "temporal_rs"  # loo | rs | temporal_rs
train_ratio: 0.8
valid_ratio: 0.1
seed: 42
```

### model yaml (Closed-form 예시)

```yaml
model:
  model_name: 'EASE'
  reg_lambda: 500.0
  device: 'auto'         # auto | cpu | cuda
```

### model yaml (SGD 예시)

```yaml
model:
  model_name: 'MF'
  embedding_dim: 64
  device: 'auto'

  train:
    epochs: 100
    batch_size: 1024
    lr: 0.001
    weight_decay: 0.0001
    patience: 10
    num_negatives: 1
    negative_sampling_strategy: 'uniform'  # uniform | popularity
```

---

## Evaluation Protocol

논문 수준의 공정성을 위해 다음 프로토콜을 따릅니다.

| 항목 | 내용 |
|------|------|
| 평가 방식 | Full-ranking (전체 아이템 대상, 샘플링 없음) |
| 학습 아이템 마스킹 | 평가 시 유저의 학습 아이템을 -1e10으로 마스킹 |
| HPO | Validation set 기준으로만 하이퍼파라미터 선택 |
| 최종 평가 | Test set 단 1회 |
| item_popularity | Train set만 기준으로 계산 (test 정보 미사용) |
| 멀티시드 | 각 시드 독립적으로 HPO → test 평가 → 평균/표준편차 리포트 |

---

## Adding a New Model

1. `src/models/your_model.py` 작성 (`BaseModel` 상속)
2. `src/models/__init__.py`의 `MODEL_REGISTRY`에 등록
3. `configs/models/your_model.yaml` 작성

자세한 내용은 [docs/USAGE.md](docs/USAGE.md)를 참조하세요.
