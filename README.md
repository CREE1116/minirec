# 🚀 MiniRec: Lightweight & Config-First Recommendation Framework

`minirec`은 복잡하고 파편화된 추천 시스템 실험 과정을 하나로 묶어, **가장 핵심적인 로직만으로 구성한 초경량 프레임워크**입니다. 닫힌 해(Closed-form) 모델과 반복 학습(SGD) 모델을 하나의 인터페이스로 지원합니다.

## ✨ Key Features

- **Unified Interface**: `minirec.run()`과 `minirec.hporun()`으로 모든 실험 제어.
- **Dual-Path Execution**: 모델 구조에 따라 최적의 학습 경로(Closed-form vs SGD) 자동 선택.
- **Config-First Design**: 모든 실험 파라미터를 YAML 또는 Dictionary로 관리.
- **Support List**: [Implemented Models & Metrics](docs/MODELS_AND_METRICS.md) 상세 리스트 제공.
- **Smart Caching**: 전처리된 데이터를 피클 파일로 자동 저장하여 반복 로딩 시간 단축.
- **Flexible Splitting**: `LOO`, `Random Split`, `Temporal Split` 등 필수 분할 방식 완벽 지원.

## 📦 Installation

```bash
# 레포지토리 클론 후 해당 환경에서 바로 import 가능
import minirec
```

## 🛠 Quick Start

### 1. 단일 실험 실행 (run)
```python
import minirec

# 설정 파일 로드 및 실행
metrics = minirec.run(
    dataset_cfg='configs/minirec/dataset_ml100k.yaml',
    model_cfg='configs/minirec/model_ease.yaml',
    output_path='output/test_run'
)
print(f"Results: {metrics}")
```

### 2. 하이퍼파라미터 최적화 (hporun)
```python
hpo_cfg = {
    'metric': 'NDCG@10',
    'direction': 'max',
    'params': [
        {'name': 'model.reg_lambda', 'type': 'float', 'range': '10 1000', 'log': True}
    ]
}

best_params = minirec.hporun(
    dataset_cfg='configs/minirec/dataset_ml100k.yaml',
    model_cfg='configs/minirec/model_ease.yaml',
    hpo_cfg=hpo_cfg,
    n_trials=20
)
```

## 📂 Configuration Structure

`minirec`은 세 단계의 설정을 병합하여 사용합니다:
1. **Evaluation Config**: 메트릭, Top-K 등 평가 기준 (기본값 제공)
2. **Dataset Config**: 데이터 경로, 필터링, 분할 방식 등
3. **Model Config**: 모델 파라미터, 학습 방식 등

## 💡 Implementing New Models

`minirec/models/base.py`의 `BaseModel`을 상속받아 구현합니다.

- **Closed-form 모델**: `fit(self, data_loader)` 메서드를 구현하세요.
- **SGD 모델**: `calc_loss(self, batch_data)` 메서드를 구현하세요.

상세한 구현 예시는 `minirec/docs/USAGE.md`를 참조하세요.

---
유지보수가 쉽고 가벼운 실험 환경을 지향합니다.
# minirec
