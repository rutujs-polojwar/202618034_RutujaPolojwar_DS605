"""Streamlit app: NYC Airbnb nightly-price estimator.

Run locally:   streamlit run app.py
(Train first with ``python train.py`` or by running the notebook so that
``models/airbnb_price_pipeline.joblib`` and ``models/metadata.json`` exist.)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from airbnb_utils import predict_price, price_band, validate_inputs  # noqa: F401 (module needed to unpickle)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "airbnb_price_pipeline.joblib"
META_PATH = ROOT / "models" / "metadata.json"
IMAGES = ROOT / "images"

st.set_page_config(page_title="NYC Airbnb Price Estimator", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta():
    return json.loads(META_PATH.read_text(encoding="utf-8"))


if not (MODEL_PATH.exists() and META_PATH.exists()):
    st.error("Model files not found. Run `python train.py` (or the notebook) first - "
             "it creates `models/airbnb_price_pipeline.joblib` and `models/metadata.json`.")
    st.stop()

model, meta = load_model(), load_meta()
NBHDS = meta["neighbourhoods"]
BOROUGHS = sorted(NBHDS)
ROOMS = meta["room_types"]
D = meta["defaults"]

# Realistic quick-start examples (a neighbourhood missing from the data falls back gracefully)
EXAMPLES = {
    "Manhattan - entire apartment": dict(borough="Manhattan", neighbourhood="Midtown", room_type="Entire home/apt",
                                         minimum_nights=3, number_of_reviews=45, reviews_per_month=1.2,
                                         calculated_host_listings_count=1, availability_365=200,
                                         title="Modern spacious 1BR near Times Square"),
    "Brooklyn - private room": dict(borough="Brooklyn", neighbourhood="Williamsburg", room_type="Private room",
                                    minimum_nights=2, number_of_reviews=20, reviews_per_month=0.8,
                                    calculated_host_listings_count=1, availability_365=120,
                                    title="Cozy private room in Williamsburg"),
    "Queens - shared room": dict(borough="Queens", neighbourhood="Astoria", room_type="Shared room",
                                 minimum_nights=1, number_of_reviews=5, reviews_per_month=0.3,
                                 calculated_host_listings_count=3, availability_365=300,
                                 title="Budget bunk bed near the subway"),
}


def _nb_list(borough):
    return sorted(NBHDS[borough])


def _default_nb(borough):
    """The borough's neighbourhood with the most listings."""
    return max(NBHDS[borough], key=lambda n: NBHDS[borough][n]["n"])


def _set_coords():
    info = NBHDS[st.session_state.borough][st.session_state.neighbourhood]
    st.session_state.lat, st.session_state.lon = info["lat"], info["lon"]


def on_borough_change():
    st.session_state.neighbourhood = _default_nb(st.session_state.borough)
    _set_coords()


def apply_example():
    ex = EXAMPLES.get(st.session_state.example)
    if not ex:
        return
    borough = ex["borough"] if ex["borough"] in NBHDS else BOROUGHS[0]
    nb = ex["neighbourhood"] if ex["neighbourhood"] in NBHDS[borough] else _default_nb(borough)
    st.session_state.update(
        borough=borough, neighbourhood=nb,
        room_type=ex["room_type"] if ex["room_type"] in ROOMS else ROOMS[0],
        minimum_nights=ex["minimum_nights"], number_of_reviews=ex["number_of_reviews"],
        reviews_per_month=float(ex["reviews_per_month"]),
        host_listings=ex["calculated_host_listings_count"],
        availability_365=ex["availability_365"], title=ex["title"])
    _set_coords()


if "borough" not in st.session_state:   # first run: sensible defaults
    st.session_state.update(
        borough="Manhattan" if "Manhattan" in NBHDS else BOROUGHS[0], room_type=ROOMS[0],
        minimum_nights=D["minimum_nights"], number_of_reviews=D["number_of_reviews"],
        reviews_per_month=float(D["reviews_per_month"]),
        host_listings=D["calculated_host_listings_count"],
        availability_365=D["availability_365"], title="", example="-")
    st.session_state.neighbourhood = _default_nb(st.session_state.borough)
    _set_coords()

st.title("🏠 NYC Airbnb Nightly Price Estimator")
st.caption("Machine-learning estimate trained on the 2019 New York City Airbnb Open Data.")

tab_pred, tab_perf, tab_about = st.tabs(["Estimate price", "Model performance", "About & limitations"])

# ------------------------------------------------------------------ predict
with tab_pred:
    st.selectbox("Quick example (optional)", ["-"] + list(EXAMPLES), key="example", on_change=apply_example)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Listing details")
        st.selectbox("Borough", BOROUGHS, key="borough", on_change=on_borough_change)
        st.selectbox("Neighbourhood", _nb_list(st.session_state.borough), key="neighbourhood",
                     on_change=_set_coords)
        st.selectbox("Room type", ROOMS, key="room_type")
        st.text_input("Listing title (optional)", key="title",
                      help="Words like 'luxury', 'modern', 'spacious' or 'budget', 'shared' are used by the model.")
        c1, c2 = st.columns(2)
        c1.number_input("Latitude", 40.4, 41.0, key="lat", step=0.001, format="%.5f")
        c2.number_input("Longitude", -74.5, -73.5, key="lon", step=0.001, format="%.5f")
        c3, c4 = st.columns(2)
        c3.number_input("Minimum nights", 1, 365, key="minimum_nights", step=1)
        c4.number_input("Availability (days/year)", 0, 365, key="availability_365", step=5)
        c5, c6 = st.columns(2)
        c5.number_input("Number of reviews", 0, 1000, key="number_of_reviews", step=1)
        c6.number_input("Reviews per month", 0.0, 30.0, key="reviews_per_month", step=0.1)
        st.number_input("Host's total listings", 1, 500, key="host_listings", step=1)
        go = st.button("Estimate nightly price", type="primary")

    with right:
        st.subheader("Estimate")
        inputs = dict(
            neighbourhood_group=st.session_state.borough, neighbourhood=st.session_state.neighbourhood,
            room_type=st.session_state.room_type, latitude=st.session_state.lat,
            longitude=st.session_state.lon, minimum_nights=st.session_state.minimum_nights,
            number_of_reviews=st.session_state.number_of_reviews,
            reviews_per_month=st.session_state.reviews_per_month,
            calculated_host_listings_count=st.session_state.host_listings,
            availability_365=st.session_state.availability_365, name=st.session_state.title)
        if go:
            for w in validate_inputs(inputs):
                st.warning(w)
            price = predict_price(model, inputs)
            lo, hi = price_band(price, meta["test_metrics"]["rmsle"])
            st.metric("Estimated price per night", f"${price:,.0f}")
            st.write(f"Typical range: **${lo:,.0f} - ${hi:,.0f}** (about +/-1 model error)")
            info = NBHDS[inputs["neighbourhood_group"]][inputs["neighbourhood"]]
            st.write(f"For context, the median listing in **{inputs['neighbourhood']}** is "
                     f"${info['median_price']:,.0f} per night ({info['n']:,} listings).")
        else:
            st.info("Fill in the listing details and press **Estimate nightly price**.")
        st.map(pd.DataFrame({"lat": [st.session_state.lat], "lon": [st.session_state.lon]}), zoom=11)

# -------------------------------------------------------------- performance
with tab_perf:
    st.subheader(f"Final model: {meta['best_model']}")
    t, tr = meta["test_metrics"], meta["train_metrics"]
    cols = st.columns(4)
    cols[0].metric("Test R²", f"{t['r2']:.3f}")
    cols[1].metric("Test MAE", f"${t['mae']:.1f}")
    cols[2].metric("Test RMSE", f"${t['rmse']:.1f}")
    cols[3].metric("Median % error", f"{t['median_ape_pct']:.1f}%")
    st.dataframe(pd.DataFrame({"train": tr, "test": t}).round(3))
    st.markdown("**Model comparison (cross-validation on the training set)**")
    st.dataframe(pd.DataFrame(meta["model_comparison"]).round(3), hide_index=True)
    for img in ("pred_vs_actual.png", "feature_importance.png"):
        if (IMAGES / img).exists():
            st.image(str(IMAGES / img))

# -------------------------------------------------------------------- about
with tab_about:
    c = meta["cleaning"]
    st.markdown(f"""
**Data.** {c['rows_clean']:,} cleaned listings (from {c['rows_raw']:,}); prices between
${c['min_price']:.0f} and ${c['price_cap']:.0f} per night.
**Trained:** {meta['trained_at_utc']} UTC with scikit-learn {meta['sklearn_version']}.

**Limitations**
- Data is from **2019**; prices today differ (inflation, regulation, post-2020 market changes).
- The model only sees location, room type, review/availability statistics and title keywords - **not** bedrooms,
  amenities, photos, ratings or seasonality, so two very different flats in one block get similar estimates.
- Listings above about ${c['price_cap']:.0f} per night were removed as outliers, so luxury listings are **under-estimated**.
- The data contains *asking* prices, not prices guests actually paid.
- The estimate is a statistical average; use the range shown, not just the single number.
""")
