# 🏠 NYC Airbnb Price Prediction – End-to-End Machine Learning Project

DS605 – Fundamentals of Machine Learning · Lab Assignment 4

A complete ML workflow on the Kaggle **New York City Airbnb Open Data (`AB_NYC_2019.csv`)**:
data cleaning → EDA → feature engineering/selection → model comparison → hyper-parameter tuning →
evaluation (incl. over-fitting checks) → saved pipeline → **Streamlit web app** for nightly-price estimates.

**Deployed app:** _paste your Streamlit Community Cloud link here_
**Repository:** _paste your GitHub link here_

---

## 1. Repository structure

```
├── airbnb_price_prediction.ipynb   # full analysis notebook (Tasks 1, 2, 4)
├── airbnb_utils.py                 # shared cleaning / feature-engineering / modelling code
├── train.py                        # reproducible command-line training (alternative to the notebook)
├── app.py                          # Streamlit application (Task 3)
├── requirements.txt                # dependencies (Python 3.11+)
├── requirements-dev.txt            # + jupyter and pytest
├── data/AB_NYC_2019.csv            # <- you download this (see step 3)
├── models/
│   ├── airbnb_price_pipeline.joblib  # saved FULL pipeline (features + encoding + model)
│   └── metadata.json                 # dropdown values, metrics, cleaning info for the app
├── images/                         # plots from the notebook + app screenshots
├── reports/                        # results.md, model_comparison.csv (auto-generated)
└── tests/test_pipeline.py          # pytest checks
```

## 2. How to operate this project (step by step)

### 2.1 Prerequisites

- **Python 3.11 or newer** (`python --version`) and **git**.
- A free GitHub account (and a free [Streamlit Community Cloud](https://streamlit.io/cloud) account for deployment).

### 2.2 Set up the environment

```bash
cd <this-repository-folder>
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
```

### 2.3 Download the dataset

1. Open <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data> (free Kaggle login required) → **Download** → unzip.
2. Put the file at **`data/AB_NYC_2019.csv`**.

Or with the Kaggle CLI: `kaggle datasets download -d dgomonov/new-york-city-airbnb-open-data -p data --unzip`
(then make sure the file is named `AB_NYC_2019.csv`).

### 2.4 Train the model – choose ONE option

**Option A – Notebook (recommended for the submission, shows all analysis & plots)**

```bash
jupyter lab
```

Open `airbnb_price_prediction.ipynb` → **Kernel ▸ Restart Kernel and Run All Cells**.
Takes roughly 5–20 minutes depending on your CPU (the model comparison and tuning dominate).
For a quick trial, set `FAST = True` in the second code cell (fewer tuning iterations).

**Option B – Script**

```bash
python train.py            # full run
python train.py --quick    # fast smoke test
```

Both options create `models/airbnb_price_pipeline.joblib`, `models/metadata.json`, `reports/results.md`
and plots in `images/`. **Do not skip this step – the app needs these files.**

### 2.5 Run the web app

```bash
streamlit run app.py
```

A browser tab opens at <http://localhost:8501>. (Always run it from the repository root.)

**Using the app**

1. Optionally pick a **Quick example** (Manhattan apartment, Brooklyn private room, Queens shared room).
2. Choose **Borough → Neighbourhood**; latitude/longitude fill in automatically (edit them for an exact location).
3. Choose the **room type**, enter a **listing title** (optional – words such as _luxury, modern, loft_ or _budget, shared_ influence the estimate), minimum nights, availability, number of reviews, reviews per month and the host's total listings.
4. Click **Estimate nightly price**. You get the price, a typical range (±1 model error), and the neighbourhood's median price for context. Warnings appear for unrealistic inputs (e.g. coordinates outside NYC).
5. The **Model performance** tab shows metrics, the model comparison and plots; **About & limitations** documents the caveats.

### 2.6 Run the tests (optional)

```bash
pytest -q
```

## 3. Publish on GitHub and deploy (Streamlit Community Cloud)

```bash
git init
git add .
git commit -m "Airbnb price prediction: notebook, model, Streamlit app"
git branch -M main
# create an EMPTY public repo on github.com first, then:
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Checklist before pushing – these must be in the repo: the notebook (with outputs saved), `app.py`, `airbnb_utils.py`,
`requirements.txt`, `README.md`, **`models/airbnb_price_pipeline.joblib`**, **`models/metadata.json`**, `images/`.
(GitHub rejects single files > 100 MB; `train.py` warns if the model is that large.)

**Deploy:** go to <https://share.streamlit.io> → _New app_ → select your repository, branch `main`, main file `app.py` → _Deploy_.
Copy the resulting URL into the top of this README.

## 4. Results

> # Results (auto-generated)

Trained: 2026-11-04T05:45:38+00:00 UTC | scikit-learn 1.8.0

- Rows: 48,895 raw -> 48,645 after cleaning (price kept between $10 and $1000)
- Final model: **Hist Gradient Boosting** (tuned with randomised search)

## Cross-validated model comparison (training set)

| model                  | cv_rmsle | cv_mae | cv_r2 | train_r2 | overfit_gap_rmsle |
| ---------------------- | -------- | ------ | ----- | -------- | ----------------- |
| Hist Gradient Boosting | 0.399    | 45.395 | 0.439 | 0.52     | 0.04              |
| Random Forest          | 0.401    | 45.442 | 0.424 | 0.55     | 0.072             |
| Ridge                  | 0.422    | 48.193 | 0.369 | 0.372    | 0.002             |
| Linear Regression      | 0.422    | 48.198 | 0.369 | 0.372    | 0.002             |

## Final model: train vs. hold-out test

| metric              | train  | test   |
| ------------------- | ------ | ------ |
| RMSE ($)            | 81.863 | 86.418 |
| MAE ($)             | 41.898 | 44.647 |
| R^2                 | 0.505  | 0.465  |
| RMSLE               | 0.366  | 0.389  |
| R^2 (log price)     | 0.693  | 0.661  |
| Median abs. % error | 21.481 | 22.550 |

| Item                 | Value             |
| -------------------- | ----------------- |
| Final model          | _from results.md_ |
| Test R² / MAE / RMSE | _from results.md_ |
| Train–test gap       | _from results.md_ |

**Plots** (created by the notebook in `images/`):

|                                       |                                                 |
| ------------------------------------- | ----------------------------------------------- |
| ![](images/01_price_distribution.png) | ![](images/04_geography_and_neighbourhoods.png) |
| ![](images/07_model_comparison.png)   | ![](images/09_error_analysis.png)               |
| ![](images/08_learning_curve.png)     | ![](images/10_permutation_importance.png)       |

**Application screenshots** – run the app, take screenshots and save them as
`images/app_screenshot_estimate.png` and `images/app_screenshot_performance.png`:

![](images/app_screenshot_estimate.png)
![](images/app_screenshot_performance.png)

## 5. Method summary

| Stage      | What was done                                                                                                                                                                                                         |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cleaning   | duplicates removed; price < $10 (incl. $0 placeholders) and above the 99.5th percentile removed; `reviews_per_month` NaN → 0 (structural: no reviews); `name` NaN → ""; identifiers and `last_review` dropped         |
| Target     | trained on `log1p(price)` (right-skewed target); predictions converted back to dollars automatically                                                                                                                  |
| Features   | 3 categorical (borough, neighbourhood, room type; rare categories grouped) + 17 engineered numeric (distance to Times Sq./Wall St./Central Park, log-transforms, flags, title keywords); mutual-information selection |
| Models     | Linear Regression, Ridge, Random Forest, Hist Gradient Boosting – 3-fold CV, metric RMSLE (+ MAE, R²)                                                                                                                 |
| Tuning     | `RandomizedSearchCV` (20 iterations, 3-fold) on the best family                                                                                                                                                       |
| Evaluation | 80/20 hold-out test set; train-vs-test gap, learning curve, residual & per-price-bucket analysis, permutation importance                                                                                              |
| Deployment | single saved scikit-learn pipeline (`joblib`) + Streamlit app                                                                                                                                                         |

## 6. Limitations

- 2019 data – prices today are different; retrain on a recent snapshot before real use.
- No bedrooms, amenities, photos, ratings or seasonality in the dataset → accuracy has a ceiling.
- Asking prices, not realised booking prices.
- Listings outside the cleaned price range (< $10 or above the 99.5th percentile) are not modelled; the most expensive listings are under-estimated.
- The ±1-error range is a population-level approximation, not a true per-listing prediction interval.

## 7. Troubleshooting

| Problem                                                 | Fix                                                                                                   |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| App says _Model files not found_                        | run the notebook or `python train.py` first, run the app from the repo root                           |
| `ModuleNotFoundError: airbnb_utils`                     | run from the repository root; keep `airbnb_utils.py` next to `app.py`                                 |
| Warning/error when loading the model (version mismatch) | train and deploy with the same `requirements.txt` (scikit-learn is pinned)                            |
| `FileNotFoundError: data/AB_NYC_2019.csv`               | see step 2.3                                                                                          |
| Streamlit Cloud build fails                             | check Python 3.11+ is selected in _Advanced settings_ and that `requirements.txt` is in the repo root |
| `git push` rejected (file > 100 MB)                     | use Hist Gradient Boosting or shrink Random Forest (`n_estimators`, `min_samples_leaf`) and retrain   |
