# Results (auto-generated)

Trained: 2026-11-04T05:45:38+00:00 UTC | scikit-learn 1.8.0

* Rows: 48,895 raw -> 48,645 after cleaning (price kept between $10 and $1000)
* Final model: **Hist Gradient Boosting** (tuned with randomised search)

## Cross-validated model comparison (training set)

| model | cv_rmsle | cv_mae | cv_r2 | train_r2 | overfit_gap_rmsle |
|---|---|---|---|---|---|
| Hist Gradient Boosting | 0.399 | 45.395 | 0.439 | 0.52 | 0.04 |
| Random Forest | 0.401 | 45.442 | 0.424 | 0.55 | 0.072 |
| Ridge | 0.422 | 48.193 | 0.369 | 0.372 | 0.002 |
| Linear Regression | 0.422 | 48.198 | 0.369 | 0.372 | 0.002 |

## Final model: train vs. hold-out test

| metric | train | test |
|---|---|---|
| RMSE ($) | 81.863 | 86.418 |
| MAE ($) | 41.898 | 44.647 |
| R^2 | 0.505 | 0.465 |
| RMSLE | 0.366 | 0.389 |
| R^2 (log price) | 0.693 | 0.661 |
| Median abs. % error | 21.481 | 22.550 |
