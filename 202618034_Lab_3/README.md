# Lab 3 — Scikit-learn: Data Preprocessing and Model Performance Evaluation

**Name:** Rutuja Polojwar
**ID:** 202618034
**Dataset:** [Hotel Booking Demand — Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

---

## Preprocessing Choices

- Dropped `company` (94.3% missing — too sparse to impute reliably).
- Numerical columns imputed with `KNNImputer(n_neighbors=5)`; categorical with `SimpleImputer(strategy="most_frequent")`.
- Dropped `reservation_status` and `reservation_status_date` (data leakage — reveal the outcome directly).
- Removed 408 rows with invalid values (negative `adr`, `adults`=0, negative `children`/`babies`).
- Categorical features encoded with `OneHotEncoder(handle_unknown="ignore")`.
- Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- **Pipeline A:** KNNImputer + StandardScaler
- **Pipeline B:** KNNImputer + MinMaxScaler
- Models: `LogisticRegression(max_iter=1000)`, `DecisionTreeClassifier(random_state=42)` — each run on both pipelines.

---

## Final Comparison Table

| Experiment                         | Train Acc | Test Acc | Precision | Recall | F1     |
| ---------------------------------- | --------- | -------- | --------- | ------ | ------ |
| Logistic Regression – A (Standard) | 0.8189    | 0.8181   | 0.8119    | 0.6630 | 0.7299 |
| Logistic Regression – B (MinMax)   | 0.8131    | 0.8133   | 0.8092    | 0.6495 | 0.7206 |
| Decision Tree – A (Standard)       | 0.9963    | 0.8614   | 0.8129    | 0.8134 | 0.8131 |
| Decision Tree – B (MinMax)         | 0.9963    | 0.8613   | 0.8124    | 0.8138 | 0.8131 |

Confusion matrices for the best model of each type are in `/figures`.

---

## Final Observations

1. **Best combination:** Decision Tree – Pipeline A has the highest test accuracy/F1, but it overfits badly (train 0.9963 vs test 0.8614). Logistic Regression – Pipeline A generalizes best (train ≈ test).
2. **Scaler effect on Logistic Regression:** StandardScaler slightly outperforms MinMaxScaler (F1 0.7299 vs 0.7206) — Logistic Regression is sensitive to feature scale.
3. **Scaler effect on Decision Tree:** Negligible (F1 0.8131 for both) — trees split on thresholds, unaffected by monotonic scaling.
4. Logistic Regression misses more actual cancellations (2973 FN) than the Decision Tree (1646 FN), giving it much lower recall (0.66 vs 0.81).
5. Precision is similar for both models (~0.81); the real gap is in recall and generalization, not precision.
6. The Decision Tree's near-perfect training accuracy (99.6%) is a clear overfitting signal — pruning (`max_depth`, `min_samples_leaf`) would likely help.
