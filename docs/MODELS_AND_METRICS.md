# 📊 Supported Models & Metrics

`minirec`은 다양한 추천 알고리즘과 평가 지표를 지원합니다.

## 🧠 Implemented Models

모든 모델은 `BaseModel`을 상속받아 구현되어 있으며, 모델의 특성에 따라 최적의 학습 경로(Closed-form 또는 SGD)를 따릅니다.

### 1. Closed-form 모델 (고정된 해를 계산)
반복적인 학습 없이 수학적 연산을 통해 최적의 가중치를 계산합니다.
- **EASE**: Embarrassingly Shallow Autoencoders
- **GF-CF**: Graph Filter based Collaborative Filtering
- **iALS**: Implicit Alternating Least Squares (전용 `fit` 구현)
- **IPS-LAE**: Inverse Propensity Scoring - Linear AutoEncoder
- **LIRA**: Linear Item-Item Recommendation Algorithm
- **PureSVD**: Pure Singular Value Decomposition

### 2. SGD 모델 (반복 학습)
오차 역전파와 그래디언트 하강법을 사용하여 학습합니다.
- **MF (BPR-MF)**: Matrix Factorization with Bayesian Personalized Ranking loss
- **LightGCN**: Graph Convolutional Networks for Recommendation

---

## 📈 Evaluation Metrics

모델의 성능을 측정하기 위해 다음과 같은 메트릭을 지원합니다. 모든 메트릭은 Top-K(예: @10, @20) 기반으로 계산됩니다.

- **Recall@K**: 실제 선호한 아이템 중 상위 K개 내에 포함된 비율
- **NDCG@K**: 순위 가중치를 반영하여 상위 K개 내의 선호 아이템 점수 합산
- **HitRate@K**: 상위 K개 내에 선호 아이템이 최소 하나 이상 포함될 확률

---

## 📂 Data Directory

기본 데이터 경로는 다음과 같습니다:
- **Root**: `/Users/leejongmin/code/minirec/data/`
- **Example**: `data/ml-100k/u.data`

각 데이터셋에 대한 상세 설정은 `configs/datasets/`에서 관리합니다.
