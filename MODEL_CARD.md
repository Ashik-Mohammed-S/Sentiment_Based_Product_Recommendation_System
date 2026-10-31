# Recommendation model and evaluation

The previous item-rating cosine average ranked rare products with a handful of
high ratings ahead of useful recommendations. It also hid most cold-start cases
in its warm-user ranking evaluation. The replacement treats ranking and rating
prediction as separate objectives.

Ranking combines item popularity, smoothed observed positive-review fractions and
regularized item co-occurrence (EASE). Unknown users receive the community ranking.
Known preferences contribute through the item-to-item matrix. Chosen items and
explicit exclusions are always removed from the output. Product metadata similarity
was considered on validation but not selected (weight zero).

Repeated user/product observations are aggregated before splitting. The fixed
outer 80/20 pair split uses seed 42. An inner 80/20 split of the outer training
pairs (seed 43) selects ranking configuration by NDCG@5 and rating shrinkage by
RMSE. All co-occurrence, popularity, positive fractions, user biases and rating
means are fitted using only the relevant training partition. Metadata is static;
held-out review text, ratings and sentiment are never used in model fitting.
The final test set is evaluated after selection, not used to choose weights.

Selected ranking: EASE regularization 1000, collaborative weight 2, sentiment
power 1, content weight 0. Ratings use item-mean shrinkage 20 and regularized user
bias. The browser displays the item estimate; it does not know reviewer identities.

| Final holdout measure | New model | Baseline |
| --- | --- | --- |
| NDCG@5 | 0.46965 | Popularity 0.45644 |
| Precision@5 | 0.11767 | Popularity 0.11615 |
| Recall@5 | 0.57521 | Popularity 0.56903 |
| Rating RMSE | 0.79820 | Global mean 0.93269 |
| Rating MAE | 0.54396 | — |

Final holdout: 5,518 pairs; ranking evaluates 4,866 users with held-out positive
ratings, including 4,205 cold users. Unknown relevant items remain in metric
denominators. Ranking catalog coverage is 15.23% versus popularity 3.91%.
The ranking improvement is modest; statistical significance was not established.
The former 0.0061 precision used a warm-only cohort, so it is not directly
comparable to the all-user precision above.

Limitations: random pair holdout is not a temporal forecast; relevant means an
observed rating >=4, while unobserved items are unjudged. Sparse user histories and
a highly concentrated catalog limit personalization. Historical products may be
outdated; no current prices, availability or purchasing integration is provided.
The original source sentiment labels and source product names may contain noise.
No online engagement lift, causal effect or live-commerce quality is claimed.

The deployment model is refitted on all available pairs after evaluation. Its
JSON artifact includes only product metadata and aggregate parameters. The Flask
and browser scorers use the same artifact and have automated ranking parity checks.
The source dataset and its provenance remain in dataset/ and dataset-sources.json.
