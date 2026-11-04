"""Reproducible command-line training script.

Usage:
    python train.py                       # full run (tuning included)
    python train.py --quick               # faster smoke run
    python train.py --data path/to/AB_NYC_2019.csv

Creates models/airbnb_price_pipeline.joblib, models/metadata.json,
reports/results.md, reports/model_comparison.csv and plots in images/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import airbnb_utils as U


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/AB_NYC_2019.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=20, help="random-search iterations")
    ap.add_argument("--quick", action="store_true", help="n_iter=5 (fast smoke test)")
    args = ap.parse_args()
    n_iter = 5 if args.quick else args.n_iter
    for d in (args.models_dir, args.images_dir, args.reports_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. data ---------------------------------------------------------------
    df, info = U.load_and_clean(args.data)
    print(f"Loaded {info['rows_raw']:,} rows -> {info['rows_clean']:,} after cleaning "
          f"(price in [{info['min_price']:.0f}, {info['price_cap']:.0f}])")
    X, y = df[U.RAW_FEATURES], df[U.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=U.RANDOM_STATE)

    # 2. feature selection ---------------------------------------------------
    num_features, mi = U.select_features_by_mi(X_train, np.log1p(y_train))
    dropped = sorted(set(U.NUM_FEATURES) - set(num_features))
    print(f"Selected {len(num_features)}/{len(U.NUM_FEATURES)} numeric features; dropped: {dropped or 'none'}")

    # 3. compare models ------------------------------------------------------
    print("\nComparing models (cross-validation)...")
    comparison = U.compare_models(X_train, y_train, num_features, cv=args.cv)
    print(comparison.round(3).to_string(index=False))
    comparison.to_csv(Path(args.reports_dir) / "model_comparison.csv", index=False)

    # 4. tune the best family --------------------------------------------------
    best_name = comparison.iloc[0]["model"]
    print(f"\nTuning '{best_name}' ({n_iter} random-search iterations)...")
    if U.get_param_space(best_name):
        search = U.tune_model(best_name, X_train, y_train, num_features, n_iter=n_iter, cv=args.cv)
        model = search.best_estimator_
        best_params = {k.replace("regressor__model__", ""): v for k, v in search.best_params_.items()}
        print("Best params:", best_params, f"| CV RMSLE: {-search.best_score_:.4f}")
    else:  # plain linear regression has nothing to tune
        model = U.make_model_pipeline(U.get_candidates()[best_name], num_features).fit(X_train, y_train)
        best_params = {}

    # 5. evaluate -----------------------------------------------------------
    train_m = U.regression_report(y_train, model.predict(X_train))
    pred = model.predict(X_test)
    test_m = U.regression_report(y_test, pred)
    print("\nTrain:", {k: round(v, 3) for k, v in train_m.items()})
    print("Test :", {k: round(v, 3) for k, v in test_m.items()})
    gap = train_m["r2_log"] - test_m["r2_log"]
    print(f"Train-test R2(log) gap = {gap:.3f} -> "
          + ("possible over-fitting, consider stronger regularisation" if gap > 0.10
             else "no serious over-fitting"))

    # 6. plots ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(y_test, pred, s=4, alpha=.3)
    lim = [0, max(y_test.max(), pred.max())]
    ax[0].plot(lim, lim, "r--"); ax[0].set(xlabel="Actual price ($)", ylabel="Predicted ($)", title="Predicted vs actual (test)")
    ax[1].hist(np.log1p(pred) - np.log1p(y_test), bins=60)
    ax[1].set(xlabel="log-price residual", title="Residuals (test)")
    fig.tight_layout(); fig.savefig(Path(args.images_dir) / "pred_vs_actual.png", dpi=130); plt.close(fig)
    imp = U.raw_feature_importance(model, X_test, y_test)
    fig, ax = plt.subplots(figsize=(7, 4)); imp[::-1].plot.barh(ax=ax)
    ax.set(title="Permutation importance (increase in RMSLE)"); fig.tight_layout()
    fig.savefig(Path(args.images_dir) / "feature_importance.png", dpi=130); plt.close(fig)

    # 7. save -------------------------------------------------------------------
    model_path = Path(args.models_dir) / "airbnb_price_pipeline.joblib"
    joblib.dump(model, model_path, compress=3)
    meta = U.build_metadata(df, num_features, best_name, best_params, train_m, test_m, comparison, info)
    (Path(args.models_dir) / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    U.write_results_md(Path(args.reports_dir) / "results.md", meta)
    size_mb = model_path.stat().st_size / 1e6
    print(f"\nSaved {model_path} ({size_mb:.1f} MB) in {time.time() - t0:.0f}s")
    if size_mb > 90:
        print("WARNING: file is close to GitHub's 100 MB limit - use Hist Gradient Boosting, "
              "or reduce Random Forest n_estimators / raise min_samples_leaf.")


if __name__ == "__main__":
    main()
