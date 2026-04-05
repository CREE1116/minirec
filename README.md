# 🚀 MiniRec: Lightweight & Config-First Recommendation Framework

`minirec`은 복잡하고 파편화된 추천 시스템 실험 과정을 하나로 묶어, **가장 핵심적인 로직만으로 구성한 초경량 프레임워크**입니다.

## ✨ Key Features

- **Unified Interface**: `minirec.run()`과 `minirec.hporun()`으로 모든 실험 제어.
- **Dual-Path Execution**: 모델 구조에 따라 최적의 학습 경로(Closed-form vs SGD) 자동 선택.
- **Smart HPO (Multi-Seed)**: Optuna 기반의 베이지안 최적화를 지원하며, 여러 시드의 평균 성능과 표준편차를 자동으로 계산.
- **Disk-Efficient Trials**: HPO 과정 중 불필요한 체크포인트 생성을 억제하고 메모리 상에서 최적화 수행.
- **Smart Caching**: 전처리된 데이터를 피클 파일로 자동 저장하여 반복 로딩 시간 단축.
- **Flexible Splitting**: `LOO`, `Random Split`, `Temporal Split` 등 필수 분할 방식 완벽 지원.

## 📦 Installation & Setup

```bash
# uv를 이용한 의존성 설치 (추천)
uv sync

# 가상환경 활성화
source .venv/bin/activate
```

## 🛠 Quick Start

### 1. 단일 실험 실행 (run)
```python
import minirec

metrics = minirec.run(
    dataset_cfg='configs/datasets/ml-100k.yaml',
    model_cfg='configs/models/ease.yaml',
    output_path='output/test_run'
)
print(f"Results: {metrics}")
```

### 2. 하이퍼파라미터 최적화 (hporun)
`hporun`은 지정된 시드들(기본 3개)에 대해 독립적인 최적화를 수행한 후, 각 시드별 Best 파라미터의 Test 성능 평균을 리포트합니다.

```python
hpo_cfg = {
    'metric': 'NDCG@10',
    'direction': 'max',
    'n_seeds': 3,
    'params': [
        {'name': 'model.reg_lambda', 'type': 'float', 'range': '10 1000', 'log': True}
    ]
}

minirec.hporun(
    dataset_cfg='configs/datasets/ml-100k.yaml',
    model_cfg='configs/models/ease.yaml',
    hpo_cfg=hpo_cfg,
    n_trials=20
)
```

## 📂 Project Structure

- `src/models/`: 추천 알고리즘 구현체 (BaseModel 상속)
- `src/data/`: 데이터 로딩 및 전처리 (Caching 지원)
- `src/trainer.py`: 학습 및 평가 루프 제어 (Early Stopping 지원)
- `src/hpo/`: 멀티 시드 베이지안 최적화 로직
- `configs/`: YAML 기반 설정 파일 관리

---
유지보수가 쉽고 가벼운 실험 환경을 지향합니다. 상세 설명은 [docs/USAGE.md](docs/USAGE.md)를 참조하세요.
