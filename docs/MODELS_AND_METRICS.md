# Supported Models & Metrics

`minirec`은 다양한 추천 알고리즘과 평가 지표를 지원합니다.

---

## Implemented Models

모든 모델은 `BaseModel`을 상속받아 구현되어 있으며, 모델의 특성에 따라 최적의 학습 경로를 자동으로 따릅니다.

### Closed-form 모델 (닫힌 해)

반복적인 학습 없이 수학적 연산으로 최적 가중치를 계산합니다. `fit()` 메서드를 구현합니다.

| 모델 | 클래스 | 핵심 아이디어 |
|------|--------|--------------|
| EASE | `EASE` | Embarrassingly Shallow Autoencoders. Ridge 회귀를 통한 Item-Item 가중치 행렬 학습 |
| LIRA | `LIRA` | Linear Item-Item Recommendation Algorithm |
| DLAE | `DLAE` | Denoising Linear AutoEncoder. Dropout 마스킹 기반 EASE 변형 |
| PureSVD | `PureSVD` | Pure Singular Value Decomposition. 상위 k 성분으로 Item-Item 유사도 근사 |
| iALS | `IALS` | Implicit Alternating Least Squares. Confidence 가중 행렬로 교대 최적화 |
| GF-CF | `GFCF` | Graph Filter based Collaborative Filtering. Symmetric normalization 기반 스펙트럼 필터 |
| IPS-LAE | `IPSLAE` | Inverse Propensity Scoring Linear AutoEncoder. 인기도 보정 가중치 적용 |
| IPS-DLAE | `IPSDLAE` | IPS + Dropout 마스킹 결합 |
| IPS-Wiener | `IPSWiener` | IPS + Wiener Filter (Pure Ridge, Diagonal zeroing 없음) |
| Pop-IPS-Wiener | `PopIPSWiener` | Popularity-aware IPS + Wiener Filter |
| Energy-Wiener | `EnergyWiener` | Energy-normalized Wiener Filter |
| Hybrid-Wiener | `HybridWiener` | IPS + Energy 혼합 정규화 Wiener Filter |
| Energy-DLAE | `EnergyDLAE` | Energy-normalized DLAE |

### SGD 모델 (반복 학습)

오차 역전파와 그래디언트 하강법으로 학습합니다. `calc_loss()` 메서드를 구현합니다.

| 모델 | 클래스 | 핵심 아이디어 |
|------|--------|--------------|
| MF (BPR-MF) | `MF` | Matrix Factorization with Bayesian Personalized Ranking loss |
| LightGCN | `LightGCN` | Graph Convolutional Networks for Recommendation |

---

## Evaluation Metrics

모든 메트릭은 Top-K 기반으로 계산됩니다 (예: `@10`, `@20`, `@50`). `configs/evaluation.yaml`에서 계산할 메트릭과 K 값을 설정합니다.

### Accuracy 메트릭

| 메트릭 | 설명 |
|--------|------|
| `Recall@K` | 정답 아이템 중 상위 K개에 포함된 비율 |
| `NDCG@K` | 순위 가중치를 반영한 DCG 정규화 점수 |
| `HitRate@K` | 상위 K개 내에 정답 아이템이 하나 이상 있는 유저 비율 |
| `Precision@K` | 상위 K개 중 정답 아이템 비율 |

### Long-tail / Head 메트릭

`long_tail_percent`(기본값: `0.8`)로 정의됩니다. 전체 interaction volume의 상위 80%를 차지하는 아이템을 Head로, 나머지를 Long-tail로 분류합니다.

| 메트릭 | 설명 |
|--------|------|
| `LongTailRecall@K` | Long-tail 정답 아이템 기준 Recall |
| `LongTailNDCG@K` | Long-tail 정답 아이템 기준 NDCG |
| `HeadRecall@K` | Head 정답 아이템 기준 Recall |
| `HeadNDCG@K` | Head 정답 아이템 기준 NDCG |

### Beyond-accuracy 메트릭

| 메트릭 | 설명 |
|--------|------|
| `Coverage@K` | 전체 아이템 중 적어도 한 유저에게 추천된 아이템 비율 |
| `LongTailCoverage@K` | Long-tail 아이템 중 추천된 비율 |
| `Novelty@K` | 추천 아이템의 평균 self-information (낮은 인기도 = 높은 Novelty) |
| `GiniIndex@K` | 추천 분포의 불평등 지수 (0에 가까울수록 균등) |
| `PopRatio@K` | 추천 아이템의 평균 인기도 비율 (전체 추천 아이템 기준) |

---

## 설정 예시

### evaluation.yaml

```yaml
method: "full"
top_k: [10, 20, 50]
metrics:
  ["Recall", "NDCG", "HitRate", "Precision",
   "LongTailRecall", "LongTailNDCG", "HeadRecall", "HeadNDCG",
   "Coverage", "LongTailCoverage", "Novelty", "GiniIndex", "PopRatio"]
main_metric: "NDCG"
main_metric_k: 20
long_tail_percent: 0.8
```

- `main_metric` / `main_metric_k`: SGD Early Stopping 및 HPO objective로 자동 사용됩니다.
- `long_tail_percent`: 이 값 미만의 cumulative popularity를 Long-tail로 정의합니다.

---

## 데이터 경로

기본 데이터 경로는 프로젝트 루트 기준 `data/` 디렉토리입니다.

```
data/
├── ml100k/u.data
├── ml1m/ratings.dat
├── steam/steam.csv
└── yahooR3/...
```

각 데이터셋의 경로 및 설정은 `configs/datasets/`에서 관리합니다.
