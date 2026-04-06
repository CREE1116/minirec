# MiniRec Framework Usage Guide

`minirec`은 가볍고(Lightweight), 설정 기반(Config-first)이며, 닫힌 해(Closed-form) 모델과 SGD 모델을 모두 지원하는 통합 추천 시스템 실험 프레임워크입니다.

## 1. 핵심 개념

- **`run(dataset_cfg, model_cfg)`**: 단일 실험을 수행합니다. 모델의 구현 방식(`fit` 또는 `calc_loss`)에 따라 실행 경로를 자동으로 결정합니다.
- **`hporun(dataset_cfg, model_cfg, hpo_cfg)`**: Optuna를 이용한 Bayesian HPO를 수행합니다. HPO objective metric은 항상 `configs/evaluation.yaml`의 `main_metric` / `main_metric_k`를 사용합니다.
- **자동 캐싱**: 동일한 데이터 설정으로 다시 실행 시, 전처리된 데이터(`.pkl`)를 즉시 불러옵니다.

> **캐시 주의**: `split_method`, `valid_ratio`, `train_ratio`, `seed` 등 전처리 관련 설정을 변경한 경우, 반드시 `data_cache/` 디렉토리를 삭제하고 재실행하세요. 캐시 파일명에 주요 설정값이 포함되어 있지만, 변경 전 캐시가 남아있을 수 있습니다.

---

## 2. 단일 실험 (`run`)

```python
import minirec

metrics = minirec.run(
    dataset_cfg='configs/datasets/ml-100k.yaml',
    model_cfg='configs/models/ease.yaml',
    output_path='output/ease_ml100k'
)
print(metrics)
# {'NDCG@10': 0.412, 'Recall@10': 0.334, ...}
```

결과는 `output_path/` 아래 `metrics.json`과 `config.yaml`로 자동 저장됩니다.

---

## 3. 하이퍼파라미터 최적화 (`hporun`)

`minirec`의 HPO는 **멀티 시드(Multi-Seed)** 평가를 기본으로 지원하여 실험 신뢰성을 높입니다.

### HPO 설정 상세 (`hpo_cfg`)

| 필드 | 설명 | 기본값 |
|------|------|--------|
| `direction` | `max` 또는 `min` | `'max'` |
| `n_seeds` | 독립 시드 수 | `3` |
| `seeds` | 직접 시드 리스트 지정 (예: `[42, 43, 44]`) | `[42, 43, 44]` |
| `patience` | Optuna 레벨 Early Stopping (개선 없는 trial 수) | `20` |
| `params` | 탐색할 파라미터 리스트 (`name`, `type`, `range`, `log`) | 필수 |

> **HPO objective metric**: `hpo_cfg`에 별도로 지정하지 않습니다. `configs/evaluation.yaml`의 `main_metric` / `main_metric_k`가 자동으로 사용됩니다.

### 파라미터 타입

| `type` | 설명 | 필요 필드 |
|--------|------|-----------|
| `float` | 연속형 실수 | `range` (`"low high"`), `log` |
| `int` | 정수 | `range` (`"low high"`), `log` |
| `int_for_k` | SVD rank k — 자동으로 `min(n_users, n_items)-1`로 상한 설정 | 없음 |
| `categorical` | 범주형 | `range` (공백 구분 문자열 또는 리스트) |

### HPO 실행 흐름

1. **Seed-wise Bayesian Search**: 각 시드별로 TPE Sampler를 사용하여 `n_trials`만큼 탐색합니다.
   - **각 trial은 Validation set만 사용** (test set 미노출). 닫힌 해 모델과 SGD 모델 모두 동일한 원칙이 적용됩니다.
2. **Seed-wise Best Test**: 시드별 최적 파라미터로 **Test set** 전체 메트릭을 계산합니다. 이 평가는 시드당 딱 1회만 수행됩니다.
3. **Statistical Aggregation**: 모든 시드의 Test 결과를 취합하여 평균(Mean)과 표준편차(Std)를 계산하고 `final_summary.json`에 저장합니다.
4. **Visualization**: 각 시드별 최적화 경로(`optimization_history.png`), 파라미터 중요도(`param_importance.png`), Slice plot을 저장합니다.

```python
import minirec

hpo_cfg = {
    'direction': 'max',
    'n_seeds': 3,
    'patience': 20,
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
# {'NDCG@20_mean': 0.423, 'NDCG@20_std': 0.008, ...}
```

파라미터 이름이 `model.` 접두사 없이 지정된 경우(예: `reg_lambda`), 자동으로 `model.reg_lambda`로 매핑됩니다. 중첩 경로도 지원합니다(예: `model.train.lr`).

---

## 4. 모델 구현하기

새로운 모델을 추가하려면 `src/models/` 내에 클래스를 만들고 `BaseModel`을 상속받으세요.

### Step 1: 모델 파일 작성

#### 닫힌 해(Closed-form) 모델 예시

```python
# src/models/my_model.py
from .base import BaseModel
import numpy as np

class MyModel(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.weight_matrix = None

    def fit(self, data_loader):
        # 전체 train 데이터를 사용하여 파라미터 계산
        # self.weight_matrix = ...
        pass

    def forward(self, user_indices):
        # 추천 점수 반환 (shape: [batch_size, n_items])
        u_ids = user_indices.cpu().numpy()
        user_vec = ...
        return user_vec @ self.weight_matrix
```

#### SGD 모델 예시

```python
# src/models/my_deep_model.py
from .base import BaseModel
import torch
import torch.nn as nn

class MyDeepModel(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        emb_dim = config['model'].get('embedding_dim', 64)
        self.user_emb = nn.Embedding(self.n_users, emb_dim)
        self.item_emb = nn.Embedding(self.n_items, emb_dim)

    def forward(self, user_indices):
        u = self.user_emb(user_indices)
        return u @ self.item_emb.weight.T

    def calc_loss(self, batch_data):
        # batch_data: {'user': ..., 'pos_item': ..., 'neg_item': ...}
        u = self.user_emb(batch_data['user'])
        pos = self.item_emb(batch_data['pos_item'])
        neg = self.item_emb(batch_data['neg_item'])
        loss = -torch.log(torch.sigmoid((u * pos).sum(-1) - (u * neg).sum(-1))).mean()
        return (loss,), None
```

### Step 2: 모델 레지스트리 등록

`src/models/__init__.py`의 `MODEL_REGISTRY`에 추가합니다:

```python
from .my_model import MyModel
MODEL_REGISTRY = {
    ...
    'mymodel': MyModel,
}
```

### Step 3: 설정 파일 작성

`configs/models/my_model.yaml`:

```yaml
model:
  model_name: 'MyModel'
  reg_lambda: 100.0
  device: 'auto'
```

SGD 모델이라면 `train` 섹션도 추가합니다:

```yaml
model:
  model_name: 'MyDeepModel'
  embedding_dim: 64
  device: 'auto'

  train:
    epochs: 100
    batch_size: 1024
    lr: 0.001
    weight_decay: 0.0001
    patience: 10
    num_negatives: 1
    negative_sampling_strategy: 'uniform'   # uniform | popularity
```

---

## 5. 데이터 분할 방식

설정 파일의 `split_method`로 지정합니다.

| 방법 | 설명 |
|------|------|
| `loo` | Leave-One-Out: 유저당 마지막 1개를 test, 그 이전 1개를 validation으로 사용 |
| `rs` | Random Split: 전체 interaction을 `train_ratio` / `valid_ratio` / (나머지 test) 비율로 무작위 분할 |
| `temporal_rs` | Temporal Random Split: 시간순 정렬 후 `train_ratio` / `valid_ratio` 비율로 분할 |

### 분할 후 데이터 흐름

```
전체 interactions
    └─ remap_ids()            # user_id, item_id 0-based 재매핑
    └─ split_xxx()            # train / valid / test 분리
    └─ item_popularity 계산   # train_df 기반으로만 계산 (test 정보 미사용)
    └─ 캐싱 (.pkl)
```

> `item_popularity`는 반드시 **train set 기준**으로 계산됩니다. Novelty, GiniIndex, PopRatio, LongTailCoverage 등 popularity 기반 메트릭의 공정성이 보장됩니다.

---

## 6. 평가 프로토콜

| 항목 | 내용 |
|------|------|
| 평가 방식 | Full-ranking (전체 아이템 대상, 샘플링 없음) |
| 학습 아이템 마스킹 | 평가 시 유저의 학습 아이템을 `-1e10`으로 마스킹 |
| HPO | Validation set 기준으로만 하이퍼파라미터 선택 |
| 최종 평가 | Test set 단 1회 (시드당) |
| `item_popularity` | Train set만 기준으로 계산 |
| 멀티시드 | 각 시드 독립적으로 HPO → test 평가 → 평균/표준편차 리포트 |
