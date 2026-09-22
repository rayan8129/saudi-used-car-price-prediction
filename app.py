"""
Saudi Car Value — Streamlit app
================================
A real, working price-estimator app for the Saudi Used Car Price Prediction project.

Run locally:
    pip install streamlit scikit-learn pandas numpy joblib
    streamlit run app.py

Requires in the same folder:
    - rf_model.joblib          (trained model pipeline)
    - app_reference_data.json  (dropdown options + market medians, generated from the training data)
"""

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Saudi Car Value", page_icon="🚗", layout="wide")

# ----------------------------------------------------------------------
# Load model + reference data
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib")

@st.cache_data
def load_reference():
    with open("app_reference_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
ref = load_reference()

MAE = 13200  # mean absolute error from the tuned Random Forest on the held-out test set
R2 = 0.799

# ----------------------------------------------------------------------
# Styling — dark theme matching the project's design
# ----------------------------------------------------------------------
st.markdown("""
<style>
:root{
    --bg:#1B1D22; --panel:#22252C; --field:#2A2E36; --border:#383C46;
    --text:#F3F1EC; --muted:#9A9FAB; --amber:#F2994A; --green:#3DDC97;
}
.stApp{ background:var(--bg); color:var(--text); }
h1,h2,h3,h4,h5,h6,p,span,label,div { color:var(--text); }
.block-container{ padding-top:2rem; }

.brand-row{ display:flex; align-items:center; gap:12px; margin-bottom:6px;}
.brand-mark{ width:34px;height:34px;border-radius:8px;
    background:linear-gradient(135deg,#F2994A,#C96A2B); display:flex;align-items:center;justify-content:center;
    font-size:18px;}
.brand-name{ font-size:20px; font-weight:700; }
.brand-sub{ font-size:12.5px; color:var(--muted); }

.price-num{ font-size:56px; font-weight:800; letter-spacing:-0.02em; color:var(--text); }
.price-cur{ font-size:20px; color:var(--muted); font-weight:500; margin-left:6px;}
.result-badge{ font-size:12px; color:var(--green); background:rgba(61,220,151,0.12);
    padding:5px 12px; border-radius:20px; font-weight:600; display:inline-block;}

.info-card{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px 18px;}
.info-card .k{ font-size:11.5px; color:var(--muted); margin-bottom:6px;}
.info-card .v{ font-size:19px; font-weight:700;}
.info-card .v.green{ color:var(--green);}

.footnote{ font-size:11.5px; color:var(--muted); border-top:1px solid var(--border); padding-top:14px; margin-top:24px;}

div[data-baseweb="select"] > div{ background:var(--field) !important; border-color:var(--border) !important; }
.stNumberInput input{ background:var(--field) !important; color:var(--text) !important; border-color:var(--border) !important;}
.stButton button{
    background:var(--amber) !important; color:#1B1D22 !important; border:none !important;
    font-weight:700 !important; border-radius:10px !important; padding:0.65rem 1rem !important;
    width:100%;
}
.stRadio > label, .stSelectbox > label, .stNumberInput > label, .stSlider > label{
    color:var(--muted) !important; font-size:12px !important; font-weight:600 !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown("""
<div class="brand-row">
  <div class="brand-mark">🚗</div>
  <div>
    <div class="brand-name">Saudi Car Value</div>
    <div class="brand-sub">Instant market price estimator</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.write("")

col_form, col_result = st.columns([1, 1.3], gap="large")

# ----------------------------------------------------------------------
# LEFT: input form
# ----------------------------------------------------------------------
with col_form:
    st.markdown("### Tell us about your car")
    st.caption("Enter your vehicle's details and get an instant, data-driven price estimate "
               "based on real Saudi market listings.")

    makes = ref["cat_options"]["Make"]
    default_make_idx = makes.index("Toyota") if "Toyota" in makes else 0
    make = st.selectbox("Make", makes, index=default_make_idx)

    types_for_make = ref["make_types"].get(make, ref["cat_options"]["Type"])
    car_type = st.selectbox("Model / Type", types_for_make)

    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Year", min_value=ref["year_range"][0], max_value=2026,
                                value=min(2021, ref["year_range"][1]))
    with c2:
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000,
                                   value=ref["mileage_median"], step=1000)

    c3, c4 = st.columns(2)
    with c3:
        engine_size = st.number_input("Engine Size (L)", min_value=ref["engine_range"][0],
                                       max_value=ref["engine_range"][1], value=2.5, step=0.1)
    with c4:
        region = st.selectbox("Region", ref["cat_options"]["Region"])

    fuel_type = st.radio("Fuel Type", ref["cat_options"]["Fuel_Type"], horizontal=True)
    gear_type = st.radio("Transmission", ref["cat_options"]["Gear_Type"], horizontal=True)
    options_level = st.radio("Options", ref["cat_options"]["Options"], horizontal=True)

    c5, c6 = st.columns(2)
    with c5:
        origin = st.selectbox("Origin", ref["cat_options"]["Origin"])
    with c6:
        color = st.selectbox("Color", ref["cat_options"]["Color"])

    submit = st.button("Get Estimated Price", use_container_width=True)

# ----------------------------------------------------------------------
# RIGHT: result
# ----------------------------------------------------------------------
with col_result:
    if submit:
        car_age = 2026 - year
        row = pd.DataFrame([{
            "Make": make, "Type": car_type, "Year": year, "Origin": origin,
            "Color": color, "Options": options_level, "Engine_Size": engine_size,
            "Fuel_Type": fuel_type, "Gear_Type": gear_type, "Mileage": mileage,
            "Region": region, "Car_Age": car_age,
        }])

        pred_log = model.predict(row)[0]
        pred_price = float(np.expm1(pred_log))

        market_avg = ref["median_by_make"].get(make, ref["overall_median_price"])
        diff_pct = (pred_price - market_avg) / market_avg * 100 if market_avg else 0
        badge_txt = "Good deal vs. market" if diff_pct <= 0 else "Above market average"
        badge_color = "green" if diff_pct <= 5 else ""

        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div style="font-size:13px;color:#9A9FAB;">Estimated Market Price</div>
          <div class="result-badge">● {badge_txt}</div>
        </div>
        <div class="price-num">{pred_price:,.0f}<span class="price-cur">SAR</span></div>
        """, unsafe_allow_html=True)

        low = max(0, pred_price - MAE)
        high = pred_price + MAE
        st.markdown(f"""
        <div style="font-size:11.5px;color:#9A9FAB;margin-top:6px;">
          Estimated range: SAR {low:,.0f} &ndash; {high:,.0f} (&plusmn; mean model error)
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f'<div class="info-card"><div class="k">Market Average ({make})</div>'
                        f'<div class="v">SAR {market_avg:,.0f}</div></div>', unsafe_allow_html=True)
        with k2:
            cls = "green" if diff_pct <= 5 else ""
            st.markdown(f'<div class="info-card"><div class="k">vs. Market Average</div>'
                        f'<div class="v {cls}">{diff_pct:+.1f}%</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="info-card"><div class="k">Model Confidence</div>'
                        f'<div class="v">{R2*100:.1f}% (R&sup2;)</div></div>', unsafe_allow_html=True)

        st.write("")
        st.markdown("**How this price compares**")
        max_ref = max(pred_price, market_avg) * 1.15
        st.markdown(f"Your car — SAR {pred_price:,.0f}")
        st.progress(min(1.0, pred_price / max_ref))
        st.markdown(f"Market average ({make}) — SAR {market_avg:,.0f}")
        st.progress(min(1.0, market_avg / max_ref))

        st.markdown(
            f'<div class="footnote">Estimate generated by a Random Forest regression model trained on '
            f'5,370 real Saudi used-car listings &middot; R&sup2; = {R2} &middot; Mean error &plusmn; SAR {MAE:,}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Fill in your car's details on the left and click **Get Estimated Price** "
                "to see the market valuation.")
