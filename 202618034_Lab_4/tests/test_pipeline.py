"""Basic tests. Run with:  pytest -q   (model tests are skipped until a model is trained)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import airbnb_utils as U

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "airbnb_price_pipeline.joblib"
META = ROOT / "models" / "metadata.json"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="train the model first")


def _row(**kw):
    return pd.DataFrame([{**U.DEFAULT_INPUT, **kw}])[U.RAW_FEATURES]


def test_add_features_columns_and_handles_missing():
    out = U.add_features(_row(reviews_per_month=np.nan, name=None, minimum_nights=1250))
    assert set(U.NUM_FEATURES) <= set(out.columns)
    assert out.loc[0, "reviews_per_month"] == 0            # NaN -> 0
    assert out.loc[0, "log_min_nights"] == pytest.approx(np.log1p(365))  # clipped
    assert out.isna().sum().sum() == 0


def test_distance_to_times_square_is_small_for_midtown():
    assert U.add_features(_row(latitude=40.7580, longitude=-73.9855)).loc[0, "dist_times_sq_km"] < 0.01


def test_price_band_and_validation():
    lo, hi = U.price_band(100, 0.4)
    assert lo < 100 < hi
    assert U.validate_inputs({**U.DEFAULT_INPUT, "latitude": 10.0})        # outside NYC
    assert not U.validate_inputs(U.DEFAULT_INPUT)


@needs_model
def test_model_predicts_positive_dollars_and_is_deterministic():
    model = joblib.load(MODEL)
    a = U.predict_price(model, {})
    assert 10 <= a <= 1500 and a == U.predict_price(model, {})


@needs_model
def test_model_handles_unseen_neighbourhood():
    assert U.predict_price(joblib.load(MODEL), {"neighbourhood": "Nowhereville"}) > 0


@needs_model
def test_sanity_entire_home_costs_more_than_shared_room():
    model = joblib.load(MODEL)
    assert U.predict_price(model, {"room_type": "Entire home/apt"}) > U.predict_price(model, {"room_type": "Shared room"})


@needs_model
def test_metadata_is_consistent():
    meta = json.loads(META.read_text())
    assert set(meta["neighbourhoods"]) <= {"Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"}
    assert 0 < meta["test_metrics"]["r2"] <= 1
