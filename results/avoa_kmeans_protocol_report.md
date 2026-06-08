# AVOA-K-MEANS Protocol Report

- Dataset: ML-100K official 5-fold
- K=10, WNMF L=50
- Baseline: PLAIN_KMEANS (B0_KMEANS)
- Challenger: AVOA_KMEANS (B_AVOA + Lloyd K-means)

## CV5 Improvement vs Plain K-means (%)

| Metric | Improvement % |
|--------|---------------|
| MAE | 0.05% |
| RMSE | 0.13% |
| Precision@10 | 0.08% |
| Recall@10 | 0.24% |
| NDCG@10 | 0.29% |

## Tez cümlesi (şablon)

> AVOA-K-MEANS yönteminde, WNMF ile elde edilen kullanıcı özellik uzayında African Vulture Optimization Algorithm (AVOA) ile centroid konumları optimize edilmiş; elde edilen centroidler başlangıç değeri olarak Lloyd K-means algoritmasına verilmiştir. Düz K-means baseline'ına kıyasla ML-100K official 5-fold protokolünde CalcAvgRating tahmininde MAE/0.05%, RMSE/0.13%, Precision@10/0.08%, Recall@10/0.24%, NDCG@10/0.29% iyileşme gözlemlenmiştir.

Detay: `results\avoa_kmeans_protocol_cv5.csv`