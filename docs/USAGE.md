# MiniRec Framework Usage Guide

`minirec`은 가볍고(Lightweight), 설정 기반(Config-first)이며, 닫힌 해(Closed-form) 모델과 SGD 모델을 모두 지원하는 통합 추천 시스템 실험 프레임워크입니다.

## 1. 핵심 개념
- **run(dataset_cfg, model_cfg)**: 단일 실험을 수행합니다. 모델의 구현 방식(`fit` 또는 `calc_loss`)에 따라 실행 경로를 자동으로 결정합니다.
- **hporun(dataset_cfg, model_cfg, hpo_cfg)**: Optuna를 이용한 하이퍼파라미터 탐색을 수행합니다.
- **자동 캐싱**: 동일한 데이터 설정으로 다시 실행 시, 처리된 데이터(`pickle`)를 즉시 불러옵니다.

## 2. 하이퍼파라미터 최적화 (Smart HPO)
`minirec`의 HPO는 단순한 탐색을 넘어, 실험의 신뢰성을 높이기 위한 **멀티 시드(Multi-Seed)** 평가를 기본으로 지원합니다.

### HPO 설정 상세 (`hpo_cfg`)
- `metric`: 최적화 기준이 될 메인 메트릭 (예: `NDCG@10`).
- `direction`: `max` 또는 `min`.
- `n_seeds`: 독립적으로 최적화 및 평가를 수행할 시드 개수 (기본값: 3).
- `seeds`: 직접 시드 리스트를 지정할 수도 있습니다 (예: `[42, 43, 44]`).
- `patience`: Optuna 레벨의 Early Stopping. 지정된 횟수만큼 개선이 없으면 탐색 중단.
- `params`: 탐색할 파라미터 리스트. `name`, `type`, `range`, `log` 속성 포함.

### HPO 실행 흐름
1. **Memory-only Trials**: 각 시드별로 `n_trials`만큼 탐색을 진행합니다. 이때 디스크에 불필요한 파일을 남기지 않고 메모리에서 검증 메트릭만 계산합니다.
2. **Seed-wise Best Test**: 각 시드별 최적 파라미터를 찾으면, 해당 파라미터로 **Test 데이터셋**에 대해 전체 메트릭을 계산합니다.
3. **Statistical Aggregation**: 모든 시드의 Test 결과를 취합하여 평균(Mean)과 표준편차(Std)를 계산하고 `final_summary.json`에 저장합니다.
4. **Optimization History**: 각 시드별로 최적화 경로를 시각화하여 `optimization_history.png`로 저장합니다.

```python
hpo_cfg = {
    'metric': 'NDCG@10',
    'direction': 'max',
    'n_seeds': 3,
    'patience': 20,
    'params': [
        {'name': 'model.reg_lambda', 'type': 'float', 'range': '10 1000', 'log': True},
        {'name': 'train.lr', 'type': 'float', 'range': '0.0001 0.01', 'log': True}
    ]
}
```

## 3. 모델 구현하기
새로운 모델을 추가하려면 `src/models/` 내에 클래스를 만들고 `BaseModel`을 상속받으세요.

### 닫힌 해(Closed-form) 모델 예시
```python
from .base import BaseModel
import scipy.sparse as sp

class MyModel(BaseModel):
    def fit(self, data_loader):
        # 전체 데이터를 사용한 파라미터 계산 (예: EASE)
        # self.weight_matrix = ...
        pass
    
    def forward(self, user_indices):
        # 추천 점수 반환
        return scores
```

### SGD(Deep Learning) 모델 예시
```python
from .base import BaseModel
import torch

class MyDeepModel(BaseModel):
    def forward(self, user_indices):
        # 모델의 예측값 반환
        return scores
        
    def calc_loss(self, batch_data):
        # 손실 함수 계산 로직 (예: BPR Loss)
        # return (loss_tuple,), extra_info
        return (loss,), None
```

## 4. 데이터 분할 방식
설정 파일의 `split_method`를 통해 지정할 수 있습니다.
- `loo`: Leave-One-Out (유저당 마지막 1개 테스트, 1개 검증)
- `rs`: Random Split (무작위 비율 분할)
- `temporal`: Temporal Random Split (시간순 비율 분할)
