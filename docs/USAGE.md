# MiniRec Framework Usage Guide

`minirec`은 가볍고(Lightweight), 설정 기반(Config-first)이며, 닫힌 해(Closed-form) 모델과 SGD 모델을 모두 지원하는 통합 추천 시스템 실험 프레임워크입니다.

## 1. 핵심 개념
- **run(dataset_cfg, model_cfg)**: 단일 실험을 수행합니다. 모델의 구현 방식(`fit` 또는 `calc_loss`)에 따라 실행 경로를 자동으로 결정합니다.
- **hporun(dataset_cfg, model_cfg, hpo_cfg)**: Optuna를 이용한 하이퍼파라미터 탐색을 수행합니다.
- **자동 캐싱**: 동일한 데이터 설정으로 다시 실행 시, 처리된 데이터(`pickle`)를 즉시 불러옵니다.

## 2. 모델 구현하기
새로운 모델을 추가하려면 `minirec/models/` 내에 클래스를 만들고 `BaseModel`을 상속받으세요.

### 닫힌 해(Closed-form) 모델 예시
```python
from .base import BaseModel
import numpy as np

class MyModel(BaseModel):
    def fit(self, data_loader):
        # 전체 데이터를 사용한 파라미터 계산 (예: EASE)
        pass
    
    def forward(self, user_indices):
        # 추천 점수 반환
        pass
```

### SGD(Deep Learning) 모델 예시
```python
from .base import BaseModel
import torch

class MyDeepModel(BaseModel):
    def forward(self, user_indices):
        # 모델의 예측값 반환
        pass
        
    def calc_loss(self, batch_data):
        # 손실 함수 계산 로직 (예: BPR Loss)
        return (loss_val,), {}
```

## 3. 실험 실행하기
설정 파일 경로를 인자로 전달하여 실험을 시작합니다.

```python
import minirec

# 단일 실험
metrics = minirec.run(
    dataset_cfg='configs/minirec/dataset_ml100k.yaml',
    model_cfg='configs/minirec/model_ease.yaml',
    output_path='output/my_experiment'
)

# 하이퍼파라미터 최적화 (HPO)
hpo_cfg = {
    'metric': 'NDCG@10',
    'direction': 'max',
    'params': [
        {'name': 'model.reg_lambda', 'type': 'float', 'range': '100 1000', 'log': True}
    ]
}
best_params = minirec.hporun(
    dataset_cfg='configs/minirec/dataset_ml100k.yaml',
    model_cfg='configs/minirec/model_ease.yaml',
    hpo_cfg=hpo_cfg,
    n_trials=20
)
```

## 4. 데이터 분할 방식
설정 파일의 `split_method`를 통해 지정할 수 있습니다.
- `loo`: Leave-One-Out (유저당 마지막 1개 테스트)
- `rs`: Random Split (무작위 비율 분할)
- `temporal`: Temporal Random Split (시간순 비율 분할)
