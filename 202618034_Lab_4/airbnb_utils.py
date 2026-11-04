"""Shared helpers for the NYC Airbnb price-prediction project.

Everything that the notebook, ``train.py`` and the Streamlit app have in common
lives here, so the *exact same* preprocessing is used for training and for
predicting on new listings.  The saved model is a single scikit-learn object
(feature engineering + encoding + scaling + regressor + log-target handling),
which means it must be loaded in an environment where this module is
importable (i.e. run the app from the repository root).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Column definitions
# --------------------------------------------------------------------------
TARGET = "price"
CAT_FEATURES = ["neighbourhood_group", "neighbourhood", "room_type"]
# Raw columns a user must supply for a prediction (identifiers, host_name and
# last_review are deliberately excluded - see the notebook for the reasoning).
RAW_FEATURES = CAT_FEATURES + [
    "latitude", "longitude", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count",
    "availability_365", "name",
]
REQUIRED_COLUMNS = RAW_FEATURES + [TARGET]

# Engineered + numeric features produced by ``add_features``
NUM_FEATURES = [
    "latitude", "longitude",
    "dist_times_sq_km", "dist_wall_st_km", "dist_central_park_km",
    "log_min_nights", "log_reviews", "reviews_per_month",
    "log_host_listings", "host_multi",
    "availability_365", "avail_zero", "has_reviews",
    "name_len", "name_words", "has_premium_word", "has_budget_word",
]

LANDMARKS = {  # (lat, lon) of major NYC demand centres
    "times_sq": (40.7580, -73.9855),
    "wall_st": (40.7074, -74.0113),
    "central_park": (40.7829, -73.9654),
}
PREMIUM_PATTERN = (r"luxur|penthouse|modern|spacious|view|loft|designer|stunning|"
                   r"beautiful|gorgeous|elegant|renovated|huge|large|bright|brand new")
BUDGET_PATTERN = r"budget|cheap|small|tiny|shared|bunk|basement|hostel|affordable|bed in|cozy"

NYC_BOUNDS = {"lat": (40.49, 40.92), "lon": (-74.27, -73.68)}


# --------------------------------------------------------------------------
# Data loading / cleaning
# --------------------------------------------------------------------------
def load_and_clean(path: str | Path, upper_quantile: float = 0.995,
                   min_price: float = 10.0):
    """Load the Kaggle CSV and apply the cleaning decisions.

    * exact duplicate rows are dropped;
    * listings priced below ``min_price`` (includes $0 placeholders) are dropped;
    * listings above the ``upper_quantile`` price (extreme outliers, up to
      $10,000 in the raw data) are dropped so they do not dominate the loss;
    * missing ``name`` -> empty string (handled again inside the pipeline).

    Returns ``(clean_df, info)`` where ``info`` documents what was removed.
    """
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = raw.drop_duplicates().copy()
    cap = float(df.loc[df[TARGET] > 0, TARGET].quantile(upper_quantile))
    keep = (df[TARGET] >= min_price) & (df[TARGET] <= cap)
    info = {
        "rows_raw": int(len(raw)),
        "duplicates_dropped": int(len(raw) - len(df)),
        "price_zero_or_below": int((df[TARGET] <= 0).sum()),
        "price_below_min": int((df[TARGET] < min_price).sum()),
        "price_above_cap": int((df[TARGET] > cap).sum()),
        "min_price": float(min_price),
        "price_cap": cap,
        "upper_quantile": upper_quantile,
    }
    df = df.loc[keep].copy()
    df["name"] = df["name"].fillna("")
    info["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), info


# --------------------------------------------------------------------------
# Feature engineering (runs inside the saved pipeline)
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised)."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Turn raw listing columns into the model's input features."""
    X = pd.DataFrame(X).copy()
    out = pd.DataFrame(index=X.index)
    for c in CAT_FEATURES:
        out[c] = X[c].astype(str)

    lat = pd.to_numeric(X["latitude"]).to_numpy(float)
    lon = pd.to_numeric(X["longitude"]).to_numpy(float)
    out["latitude"], out["longitude"] = lat, lon
    for key, (la, lo) in LANDMARKS.items():
        out[f"dist_{key}_km"] = haversine_km(lat, lon, la, lo)

    min_nights = pd.to_numeric(X["minimum_nights"]).clip(1, 365)
    n_rev = pd.to_numeric(X["number_of_reviews"]).clip(lower=0)
    host_n = pd.to_numeric(X["calculated_host_listings_count"]).clip(lower=1)
    avail = pd.to_numeric(X["availability_365"]).clip(0, 365)

    out["log_min_nights"] = np.log1p(min_nights)
    out["log_reviews"] = np.log1p(n_rev)
    out["reviews_per_month"] = pd.to_numeric(X["reviews_per_month"]).fillna(0).clip(lower=0)
    out["log_host_listings"] = np.log1p(host_n)
    out["host_multi"] = (host_n > 1).astype(int)
    out["availability_365"] = avail
    out["avail_zero"] = (avail == 0).astype(int)
    out["has_reviews"] = (n_rev > 0).astype(int)

    title = X["name"].fillna("").astype(str).str.lower()
    out["name_len"] = title.str.len()
    out["name_words"] = title.str.split().str.len().fillna(0)
    out["has_premium_word"] = title.str.contains(PREMIUM_PATTERN, regex=True, na=False).astype(int)
    out["has_budget_word"] = title.str.contains(BUDGET_PATTERN, regex=True, na=False).astype(int)
    return out


def select_features_by_mi(X_raw: pd.DataFrame, y_log, threshold: float = 0.001,
                          sample: int = 20000):
    """Mutual information between each engineered numeric feature and log-price.

    Returns ``(selected_features, mi_series)``; features below ``threshold`` are
    dropped as carrying (almost) no signal.
    """
    from sklearn.feature_selection import mutual_info_regression
    feats = add_features(X_raw)[NUM_FEATURES]
    y_log = np.asarray(y_log)
    if len(feats) > sample:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(feats), sample, replace=False)
        feats, y_log = feats.iloc[idx], y_log[idx]
    mi = pd.Series(mutual_info_regression(feats, y_log, random_state=RANDOM_STATE),
                   index=NUM_FEATURES).sort_values(ascending=False)
    selected = [f for f in NUM_FEATURES if mi[f] >= threshold]
    return selected, mi


# --------------------------------------------------------------------------
# Pipeline construction
# --------------------------------------------------------------------------
def build_preprocessor(num_features: Optional[List[str]] = None) -> Pipeline:
    """Feature engineering -> one-hot (categoricals) + scaling (numerics)."""
    num_features = list(num_features or NUM_FEATURES)
    encode = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=20, sparse_output=False), CAT_FEATURES),
            ("num", StandardScaler(), num_features),
        ],
        remainder="drop",
    )
    return Pipeline([("features", FunctionTransformer(add_features, validate=False)),
                     ("encode", encode)])


def make_model_pipeline(regressor, num_features: Optional[List[str]] = None):
    """Full model: preprocessing + regressor, trained on log1p(price).

    ``TransformedTargetRegressor`` makes ``predict`` return **dollars**.
    """
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", build_preprocessor(num_features)),
                            ("model", regressor)]),
        func=np.log1p, inverse_func=np.expm1,
    )


def get_candidates(random_state: int = RANDOM_STATE, n_jobs: int = -1) -> Dict[str, object]:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, max_features=0.4, max_samples=0.5,
            n_jobs=n_jobs, random_state=random_state),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=random_state),
    }


def get_param_space(name: str) -> dict:
    """Randomised-search space (keys are relative to the regressor)."""
    from scipy.stats import loguniform, randint
    if name == "Ridge":
        return {"alpha": loguniform(1e-2, 1e2)}
    if name == "Random Forest":
        return {"n_estimators": randint(80, 200), "max_depth": [12, 18, 25, None],
                "min_samples_leaf": randint(3, 12), "max_features": [0.3, 0.5],
                "max_samples": [0.4, 0.6, 0.8]}
    if name == "Hist Gradient Boosting":
        return {"learning_rate": loguniform(0.03, 0.2), "max_iter": randint(200, 700),
                "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": randint(20, 100),
                "l2_regularization": loguniform(1e-3, 10), "max_depth": [None, 6, 10]}
    return {}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _rmsle(y_true, y_pred) -> float:
    y_pred = np.maximum(np.asarray(y_pred, float), 0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(np.asarray(y_true, float))) ** 2)))


def _get_rmsle_scorer():
    from sklearn.metrics import make_scorer
    return make_scorer(_rmsle, greater_is_better=False)


_CV_SCORING = None


def get_cv_scoring():
    global _CV_SCORING
    if _CV_SCORING is None:
        _CV_SCORING = {"rmsle": _get_rmsle_scorer(), "mae": "neg_mean_absolute_error", "r2": "r2"}
    return _CV_SCORING


def __getattr__(name: str):
    if name == "RMSLE_SCORER":
        return _get_rmsle_scorer()
    if name == "CV_SCORING":
        return get_cv_scoring()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def regression_report(y_true, y_pred) -> Dict[str, float]:
    """RMSE / MAE / R^2 in dollars, plus log-scale and relative errors."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(y_true, float)
    y_pred = np.maximum(np.asarray(y_pred, float), 0)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmsle": _rmsle(y_true, y_pred),
        "r2_log": float(r2_score(np.log1p(y_true), np.log1p(y_pred))),
        "median_ape_pct": float(np.median(np.abs(y_pred - y_true) / y_true) * 100),
    }


def compare_models(X, y, num_features=None, cv: int = 3, n_jobs: int = -1,
                   candidates: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """Cross-validate every candidate; train scores expose over-fitting."""
    from sklearn.model_selection import cross_validate
    rows = []
    scoring = get_cv_scoring()
    for name, reg in (candidates or get_candidates()).items():
        res = cross_validate(make_model_pipeline(reg, num_features), X, y, cv=cv,
                             scoring=scoring, return_train_score=True,
                             n_jobs=1 if name == "Random Forest" else n_jobs)
        rows.append({
            "model": name,
            "cv_rmsle": -res["test_rmsle"].mean(), "train_rmsle": -res["train_rmsle"].mean(),
            "cv_mae": -res["test_mae"].mean(), "train_mae": -res["train_mae"].mean(),
            "cv_r2": res["test_r2"].mean(), "train_r2": res["train_r2"].mean(),
            "fit_time_s": res["fit_time"].mean(),
        })
    out = pd.DataFrame(rows).sort_values("cv_rmsle").reset_index(drop=True)
    out["overfit_gap_rmsle"] = out["cv_rmsle"] - out["train_rmsle"]
    return out


def tune_model(name: str, X, y, num_features=None, n_iter: int = 20, cv: int = 3,
               n_jobs: int = -1):
    """Randomised hyper-parameter search optimising cross-validated RMSLE."""
    from sklearn.model_selection import RandomizedSearchCV
    reg = get_candidates()[name]
    if name == "Random Forest":          # forests are slow to fit: cap the search
        n_iter = min(n_iter, 10)
    space = {f"regressor__model__{k}": v for k, v in get_param_space(name).items()}
    search = RandomizedSearchCV(
        make_model_pipeline(reg, num_features), space, n_iter=n_iter, cv=cv,
        scoring=_get_rmsle_scorer(), n_jobs=1 if name == "Random Forest" else n_jobs,
        random_state=RANDOM_STATE, refit=True, verbose=1)
    return search.fit(X, y)


def raw_feature_importance(model, X, y, n_rows: int = 3000, n_repeats: int = 5) -> pd.Series:
    """Permutation importance on the *raw* input columns (increase in RMSLE)."""
    from sklearn.inspection import permutation_importance
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X), min(n_rows, len(X)), replace=False)
    res = permutation_importance(model, X.iloc[idx][RAW_FEATURES], np.asarray(y)[idx],
                                 scoring=_get_rmsle_scorer(), n_repeats=n_repeats,
                                 random_state=RANDOM_STATE, n_jobs=1)
    return pd.Series(res.importances_mean, index=RAW_FEATURES).sort_values(ascending=False)


# --------------------------------------------------------------------------
# Inference helpers (used by the app and the tests)
# --------------------------------------------------------------------------
DEFAULT_INPUT = {
    "neighbourhood_group": "Manhattan", "neighbourhood": "Midtown",
    "room_type": "Entire home/apt", "latitude": 40.7549, "longitude": -73.984,
    "minimum_nights": 2, "number_of_reviews": 10, "reviews_per_month": 0.5,
    "calculated_host_listings_count": 1, "availability_365": 180, "name": "",
}


def predict_price(model, inputs: dict) -> float:
    """Estimate the nightly price (USD) for one listing given as a dict."""
    row = {**DEFAULT_INPUT, **inputs}
    return float(model.predict(pd.DataFrame([row])[RAW_FEATURES])[0])


def price_band(pred: float, rmsle: float):
    """Rough +/-1 RMSLE band (multiplicative, ~2/3 of listings fall inside)."""
    return pred * float(np.exp(-rmsle)), pred * float(np.exp(rmsle))


def validate_inputs(inputs: dict) -> List[str]:
    """Human-readable warnings for inputs outside the training distribution."""
    w = []
    (la0, la1), (lo0, lo1) = NYC_BOUNDS["lat"], NYC_BOUNDS["lon"]
    if not (la0 <= inputs["latitude"] <= la1 and lo0 <= inputs["longitude"] <= lo1):
        w.append("The coordinates are outside New York City - the estimate is unreliable.")
    if inputs["minimum_nights"] > 365:
        w.append("Minimum nights above 365 is treated as 365.")
    if inputs["reviews_per_month"] > 30:
        w.append("More than 30 reviews per month is unrealistic.")
    if inputs["number_of_reviews"] == 0 and inputs["reviews_per_month"] > 0:
        w.append("A listing with 0 reviews should have 0 reviews per month.")
    return w


# --------------------------------------------------------------------------
# Metadata for the app / reports
# --------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def build_metadata(df: pd.DataFrame, num_features, best_name, best_params,
                   train_metrics, test_metrics, comparison: pd.DataFrame, info: dict) -> dict:
    import sklearn
    grp = df.groupby(["neighbourhood_group", "neighbourhood"]).agg(
        lat=("latitude", "median"), lon=("longitude", "median"),
        median_price=(TARGET, "median"), n=(TARGET, "size")).reset_index()
    neighbourhoods: Dict[str, dict] = {}
    for r in grp.itertuples():
        neighbourhoods.setdefault(r.neighbourhood_group, {})[r.neighbourhood] = {
            "lat": round(float(r.lat), 5), "lon": round(float(r.lon), 5),
            "median_price": float(r.median_price), "n": int(r.n)}
    return {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "best_model": best_name,
        "best_params": json.loads(json.dumps(best_params, default=_jsonable)),
        "selected_numeric_features": list(num_features),
        "train_metrics": train_metrics, "test_metrics": test_metrics,
        "model_comparison": json.loads(comparison.to_json(orient="records")),
        "cleaning": info,
        "neighbourhoods": neighbourhoods,
        "room_types": sorted(df["room_type"].unique().tolist()),
        "defaults": {
            "minimum_nights": int(df["minimum_nights"].median()),
            "number_of_reviews": int(df["number_of_reviews"].median()),
            "reviews_per_month": round(float(df["reviews_per_month"].fillna(0).median()), 2),
            "calculated_host_listings_count": int(df["calculated_host_listings_count"].median()),
            "availability_365": int(df["availability_365"].median()),
        },
    }


def _df_to_md(df: pd.DataFrame) -> str:
    """Markdown table without requiring the optional ``tabulate`` package."""
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def write_results_md(path: str | Path, meta: dict) -> None:
    """Write a ready-to-paste Markdown results summary for the README."""
    t, tr, c = meta["test_metrics"], meta["train_metrics"], meta["cleaning"]
    comp = pd.DataFrame(meta["model_comparison"])[
        ["model", "cv_rmsle", "cv_mae", "cv_r2", "train_r2", "overfit_gap_rmsle"]].round(3)
    lines = [
        "# Results (auto-generated)", "",
        f"Trained: {meta['trained_at_utc']} UTC | scikit-learn {meta['sklearn_version']}", "",
        f"* Rows: {c['rows_raw']:,} raw -> {c['rows_clean']:,} after cleaning "
        f"(price kept between ${c['min_price']:.0f} and ${c['price_cap']:.0f})",
        f"* Final model: **{meta['best_model']}** (tuned with randomised search)", "",
        "## Cross-validated model comparison (training set)", "",
        _df_to_md(comp),
        "", "## Final model: train vs. hold-out test", "",
        "| metric | train | test |", "|---|---|---|",
    ]
    for k, label in [("rmse", "RMSE ($)"), ("mae", "MAE ($)"), ("r2", "R^2"),
                     ("rmsle", "RMSLE"), ("r2_log", "R^2 (log price)"),
                     ("median_ape_pct", "Median abs. % error")]:
        lines.append(f"| {label} | {tr[k]:.3f} | {t[k]:.3f} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
