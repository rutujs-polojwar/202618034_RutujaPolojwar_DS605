"""Streamlit app: NYC Airbnb nightly-price estimator.

Run locally:   streamlit run app.py
"""
import json
import sys
import warnings
from pathlib import Path

# Suppress NumPy 2.x and scikit-learn unpickle deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from airbnb_utils import predict_price, price_band, validate_inputs  # noqa: F401

MODEL_PATH = ROOT / "models" / "airbnb_price_pipeline.joblib"
META_PATH = ROOT / "models" / "metadata.json"
IMAGES = ROOT / "images"

st.set_page_config(
    page_title="NYC Airbnb Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading model pipeline...")
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(MODEL_PATH)
    except Exception as err:
        st.warning(f"Could not load saved model ({err}). Running in fallback estimation mode.")
        return None


@st.cache_data(show_spinner=False)
def load_meta():
    if not META_PATH.exists():
        return None
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


meta = load_meta()
if not meta:
    st.error(
        "Metadata file `models/metadata.json` not found. "
        "Please run `python train.py` first to generate model artifacts."
    )
    st.stop()

model = load_model()

NBHDS = meta["neighbourhoods"]
BOROUGHS = sorted(NBHDS)
ROOMS = meta["room_types"]
D = meta["defaults"]

# Realistic quick-start examples
EXAMPLES = {
    "Manhattan - Entire apartment": dict(
        borough="Manhattan",
        neighbourhood="Midtown",
        room_type="Entire home/apt",
        minimum_nights=3,
        number_of_reviews=45,
        reviews_per_month=1.2,
        calculated_host_listings_count=1,
        availability_365=200,
        title="Modern spacious 1BR near Times Square",
    ),
    "Brooklyn - Private room": dict(
        borough="Brooklyn",
        neighbourhood="Williamsburg",
        room_type="Private room",
        minimum_nights=2,
        number_of_reviews=20,
        reviews_per_month=0.8,
        calculated_host_listings_count=1,
        availability_365=120,
        title="Cozy private room in Williamsburg",
    ),
    "Queens - Shared room": dict(
        borough="Queens",
        neighbourhood="Astoria",
        room_type="Shared room",
        minimum_nights=1,
        number_of_reviews=5,
        reviews_per_month=0.3,
        calculated_host_listings_count=3,
        availability_365=300,
        title="Budget bunk bed near the subway",
    ),
}


def _nb_list(borough: str):
    return sorted(NBHDS.get(borough, {}).keys())


def _default_nb(borough: str):
    """The borough's neighbourhood with the most listings."""
    nbs = NBHDS.get(borough, {})
    if not nbs:
        return ""
    return max(nbs, key=lambda n: nbs[n].get("n", 0))


def _set_coords():
    borough = st.session_state.get("borough", BOROUGHS[0])
    nbhd = st.session_state.get("neighbourhood", _default_nb(borough))
    if borough in NBHDS and nbhd in NBHDS[borough]:
        info = NBHDS[borough][nbhd]
        st.session_state.lat = float(info["lat"])
        st.session_state.lon = float(info["lon"])


def on_borough_change():
    st.session_state.neighbourhood = _default_nb(st.session_state.borough)
    _set_coords()


def apply_example():
    ex_key = st.session_state.get("example_choice")
    if not ex_key or ex_key == "-":
        return
    ex = EXAMPLES.get(ex_key)
    if not ex:
        return
    borough = ex["borough"] if ex["borough"] in NBHDS else BOROUGHS[0]
    nb = ex["neighbourhood"] if ex["neighbourhood"] in NBHDS[borough] else _default_nb(borough)
    st.session_state.update(
        borough=borough,
        neighbourhood=nb,
        room_type=ex["room_type"] if ex["room_type"] in ROOMS else ROOMS[0],
        minimum_nights=int(ex["minimum_nights"]),
        number_of_reviews=int(ex["number_of_reviews"]),
        reviews_per_month=float(ex["reviews_per_month"]),
        host_listings=int(ex["calculated_host_listings_count"]),
        availability_365=int(ex["availability_365"]),
        title=ex["title"],
    )
    _set_coords()


# Initialize sensible defaults on first run
if "borough" not in st.session_state or st.session_state.borough not in BOROUGHS:
    initial_b = "Manhattan" if "Manhattan" in NBHDS else BOROUGHS[0]
    st.session_state.update(
        borough=initial_b,
        neighbourhood=_default_nb(initial_b),
        room_type=ROOMS[0],
        minimum_nights=int(D["minimum_nights"]),
        number_of_reviews=int(D["number_of_reviews"]),
        reviews_per_month=float(D["reviews_per_month"]),
        host_listings=int(D["calculated_host_listings_count"]),
        availability_365=int(D["availability_365"]),
        title="",
        example_choice="-",
    )
    _set_coords()

# Ensure neighbourhood is always valid for the selected borough
current_nb_list = _nb_list(st.session_state.borough)
if st.session_state.get("neighbourhood") not in current_nb_list:
    st.session_state.neighbourhood = _default_nb(st.session_state.borough)
    _set_coords()

st.title("🏠 NYC Airbnb Nightly Price Estimator")
st.caption("Fast machine-learning inference trained on the New York City Airbnb Open Data.")

tab_pred, tab_perf, tab_about = st.tabs(
    ["💵 Estimate Price", "📊 Model Performance", "ℹ️ About & Limitations"]
)

# ------------------------------------------------------------------ Tab 1: Predict
with tab_pred:
    col_example, _ = st.columns([1, 1])
    with col_example:
        st.selectbox(
            "⚡ Quick Preset Example (Optional)",
            ["-"] + list(EXAMPLES.keys()),
            key="example_choice",
            on_change=apply_example,
            help="Select a typical scenario to auto-fill listing details.",
        )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Listing Details")

        b_col, nb_col = st.columns(2)
        b_col.selectbox(
            "Borough",
            BOROUGHS,
            key="borough",
            on_change=on_borough_change,
        )
        nb_col.selectbox(
            "Neighbourhood",
            _nb_list(st.session_state.borough),
            key="neighbourhood",
            on_change=_set_coords,
        )

        st.selectbox("Room Type", ROOMS, key="room_type")
        st.text_input(
            "Listing Title / Highlights (optional)",
            key="title",
            placeholder="e.g. Spacious modern loft with city views",
            help="Keywords like 'luxury', 'modern', 'spacious' vs 'budget', 'shared' influence the model.",
        )

        coord1, coord2 = st.columns(2)
        coord1.number_input("Latitude", 40.4000, 41.0000, key="lat", step=0.001, format="%.5f")
        coord2.number_input("Longitude", -74.5000, -73.5000, key="lon", step=0.001, format="%.5f")

        n1, n2 = st.columns(2)
        n1.number_input("Minimum Nights", 1, 365, key="minimum_nights", step=1)
        n2.number_input("Availability (days/year)", 0, 365, key="availability_365", step=5)

        r1, r2 = st.columns(2)
        r1.number_input("Total Number of Reviews", 0, 1000, key="number_of_reviews", step=1)
        r2.number_input("Reviews Per Month", 0.0, 30.0, key="reviews_per_month", step=0.1)

        st.number_input("Host's Total Active Listings", 1, 500, key="host_listings", step=1)

        go = st.button("✨ Estimate Nightly Price", type="primary", use_container_width=True)

    with right:
        st.subheader("Price Estimate")

        inputs = dict(
            neighbourhood_group=st.session_state.borough,
            neighbourhood=st.session_state.neighbourhood,
            room_type=st.session_state.room_type,
            latitude=st.session_state.lat,
            longitude=st.session_state.lon,
            minimum_nights=st.session_state.minimum_nights,
            number_of_reviews=st.session_state.number_of_reviews,
            reviews_per_month=st.session_state.reviews_per_month,
            calculated_host_listings_count=st.session_state.host_listings,
            availability_365=st.session_state.availability_365,
            name=st.session_state.title,
        )

        if go:
            warnings_list = validate_inputs(inputs)
            if model is not None:
                price = predict_price(model, inputs)
            else:
                # Heuristic fallback using neighbourhood median + room factor
                nb_info = NBHDS[inputs["neighbourhood_group"]].get(inputs["neighbourhood"], {})
                med = float(nb_info.get("median_price", 100.0))
                factor = 1.0 if inputs["room_type"] == "Entire home/apt" else (0.55 if inputs["room_type"] == "Private room" else 0.35)
                price = round(med * factor, 2)

            rmsle = meta.get("test_metrics", {}).get("rmsle", 0.38)
            lo, hi = price_band(price, rmsle)
            st.session_state["latest_estimate"] = {
                "price": price,
                "lo": lo,
                "hi": hi,
                "inputs": inputs,
                "warnings": warnings_list,
            }

        est = st.session_state.get("latest_estimate")
        if est:
            for w in est["warnings"]:
                st.warning(f"⚠️ {w}")

            p_val = est["price"]
            lo_val = est["lo"]
            hi_val = est["hi"]

            st.metric(
                label="Estimated Nightly Price",
                value=f"${p_val:,.0f}",
                delta=f"Range: ${lo_val:,.0f} - ${hi_val:,.0f}",
                delta_color="off",
            )
            st.info(
                f"💡 **Expected range**: **${lo_val:,.0f} to ${hi_val:,.0f}** per night "
                f"(±1 typical model error band)."
            )

            est_inputs = est["inputs"]
            b_data = NBHDS.get(est_inputs["neighbourhood_group"], {})
            info = b_data.get(est_inputs["neighbourhood"], {})
            if info:
                st.markdown(
                    f"**Neighbourhood benchmark**: The median listing in "
                    f"**{est_inputs['neighbourhood']}** ({est_inputs['neighbourhood_group']}) is "
                    f"**${info['median_price']:,.0f}** / night based on **{info['n']:,}** listings in the dataset."
                )
        else:
            st.info("👈 Set your listing details on the left and click **Estimate Nightly Price**.")

        # Lightweight collapsible map to prevent heavy WebGL rerenders from lagging the UI
        with st.expander("📍 Location Map Preview", expanded=False):
            st.caption(f"{st.session_state.neighbourhood}, {st.session_state.borough} ({st.session_state.lat:.4f}, {st.session_state.lon:.4f})")
            map_df = pd.DataFrame({"lat": [st.session_state.lat], "lon": [st.session_state.lon]})
            st.map(map_df, zoom=12)

# ------------------------------------------------------------------ Tab 2: Performance
with tab_perf:
    st.subheader(f"Model Architecture: {meta.get('best_model', 'Hist Gradient Boosting')}")
    t = meta.get("test_metrics", {})
    tr = meta.get("train_metrics", {})

    cols = st.columns(4)
    cols[0].metric("Test R²", f"{t.get('r2', 0):.3f}")
    cols[1].metric("Test MAE", f"${t.get('mae', 0):.1f}")
    cols[2].metric("Test RMSE", f"${t.get('rmse', 0):.1f}")
    cols[3].metric("Median % Error", f"{t.get('median_ape_pct', 0):.1f}%")

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Train vs. Hold-out Test Performance**")
        if tr and t:
            metrics_df = pd.DataFrame({"Train": tr, "Test": t}).round(3)
            st.dataframe(metrics_df, use_container_width=True)

    with c_right:
        st.markdown("**Cross-Validation Model Comparison**")
        comp = meta.get("model_comparison", [])
        if comp:
            st.dataframe(pd.DataFrame(comp).round(3), hide_index=True, use_container_width=True)

    # Diagnostic Visualizations
    st.markdown("### 📈 Visual Diagnostics & Feature Importances")
    plot_images = [
        ("07_model_comparison.png", "Model Comparison Across Algorithms"),
        ("09_error_analysis.png", "Residual Error Distribution & Actual vs Predicted"),
        ("10_permutation_importance.png", "Permutation Feature Importance"),
    ]
    img_cols = st.columns(len(plot_images))
    for col, (filename, caption) in zip(img_cols, plot_images):
        p = IMAGES / filename
        if p.exists():
            with col:
                st.image(str(p), caption=caption, use_container_width=True)

# ------------------------------------------------------------------ Tab 3: About
with tab_about:
    c = meta.get("cleaning", {})
    st.markdown(f"""
### 📖 Dataset & Methodology
- **Data Source:** NYC Airbnb Open Data (2019).
- **Cleaned Dataset:** {c.get('rows_clean', 0):,} listings retained from {c.get('rows_raw', 0):,} raw records.
- **Price Filter:** Nightly prices bounded between **${c.get('min_price', 10):.0f}** and **${c.get('price_cap', 500):.0f}** to eliminate extreme outliers and zero-dollar dummy listings.
- **Trained Model:** `{meta.get('best_model', 'Hist Gradient Boosting')}` with hyperparameter tuning via randomized cross-validation.
- **Engineered Signals:** Geographic haversine distance to Manhattan demand hubs (Times Square, Wall St, Central Park), log transformations, listing title keyword NLP markers, host multi-listing flags, and availability indicators.

### ⚠️ Real-World Limitations
1. **Historical Baseline (2019):** Current market rates may differ substantially due to inflation, post-2020 economic changes, and Local Law 18 short-term rental regulations in NYC.
2. **Unobserved Attributes:** The model does not observe bedroom counts, bathroom counts, square footage, specific interior amenities (washer/dryer, air conditioning), photos, or host ratings.
3. **Outlier Cap:** Extreme luxury listings (above ~${c.get('price_cap', 500):.0f}/night) were trimmed during preprocessing to prevent quadratic loss explosion, so ultra-luxury properties are conservatively estimated.
4. **Asking vs. Transacted:** The training dataset reflects host asking rates, not finalized transacted booking amounts.
""")
