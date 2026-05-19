import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Invoice Amount Predictor",
    page_icon="📦",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 760px;
}

.result-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    border: 1px solid #0f3460;
}
.result-label {
    color: #94a3b8;
    font-size: 13px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-amount {
    color: #f0f9ff;
    font-size: 3rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    letter-spacing: -1px;
}
.result-sub {
    color: #64748b;
    font-size: 12px;
    margin-top: 0.5rem;
}
.info-pill {
    display: inline-block;
    background: #0f3460;
    color: #7dd3fc;
    border-radius: 99px;
    padding: 3px 12px;
    font-size: 12px;
    margin: 0 4px;
}
.section-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin: 1.5rem 0 0.5rem;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

@st.cache_resource
def load_model():
    model        = joblib.load(os.path.join(ARTIFACT_DIR, "xgboost_model.pkl"))
    label_maps   = joblib.load(os.path.join(ARTIFACT_DIR, "label_maps.pkl"))
    feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "feature_columns.pkl"))
    return model, label_maps, feature_cols

try:
    model, label_maps, feature_cols = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 📦 Invoice Amount Predictor")
st.markdown("Predict total invoice amount for non-woven fabric orders using a trained XGBoost model.")

if not model_loaded:
    st.error("⚠️ Model artifacts not found. Make sure the `artifacts/` folder is in the same directory as `app.py`.")
    st.code("artifacts/\n  xgboost_model.pkl\n  label_maps.pkl\n  feature_columns.pkl", language="text")
    st.stop()

# ── Load metrics if available ────────────────────────────────────────────────
metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
if os.path.exists(metrics_path):
    import json
    with open(metrics_path) as f:
        metrics = json.load(f)
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{metrics['r2']:.4f}")
    c2.metric("MAE", f"₹ {metrics['mae']:,.0f}")
    c3.metric("CV R² Mean", f"{metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")

st.divider()


# ── Input Form ───────────────────────────────────────────────────────────────
MONTH_TO_NUM = {
    'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
    'July':7,'August':8,'September':9,'October':10,'November':11,'December':12
}

# Real customers and products from the dataset (remove junk header rows)
REAL_CUSTOMERS = [
    'A G Healthcare','Apex Packaging','Breathe Hygiene','Excel Plastics',
    'Garg healthcare','Global Exports','GreenTech Bags','Hygiene City',
    'Kiran Traders','Lotus Enterprises','Milan Industries','Modern Packaging',
    'National Pride','Neelkanth Fabrics','Om Industries','Parth Industries',
    'Prime Packaging','Rotech Healthcare','Royal Polyfab','Sai Enterprises',
    'Shree Textiles','Silverline Industries','Sunrise Traders','Urban Fabrics',
    'Vardhman Fabrics','Vihana healthcare','Vision Polytech',
    'meril medical innovations','millenium babycare','uniclan healthcare'
]

PRODUCTS = [
    'Colored Non-Woven Fabric',
    'Laminated Non-Woven Fabric',
    'Non-Woven Fabric 120 GSM',
    'Non-Woven Fabric 60 GSM',
    'Non-Woven Fabric 75 GSM',
    'Non-Woven Fabric 90 GSM',
    'Printed Non-Woven Fabric',
]

MONTHS = list(MONTH_TO_NUM.keys())

st.markdown('<div class="section-header">Order Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    customer = st.selectbox("Customer Name", REAL_CUSTOMERS, index=REAL_CUSTOMERS.index("Vardhman Fabrics"))
    product  = st.selectbox("Product", PRODUCTS, index=PRODUCTS.index("Non-Woven Fabric 60 GSM"))
with col2:
    month    = st.selectbox("Month", MONTHS, index=MONTHS.index("November"))
    quantity = st.number_input("Quantity (KG)", min_value=100, max_value=100000, value=2353, step=50)

avg_price = st.number_input("Average Price / KG (₹)", min_value=10.0, max_value=1000.0, value=87.45, step=0.5, format="%.2f")

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
def encode_and_predict(customer, product, month_name, quantity, avg_price):
    month_num = MONTH_TO_NUM[month_name]

    # Encode using saved maps
    enc_customer = label_maps['Customer Name'].get(customer)
    enc_product  = label_maps['Product'].get(product)
    enc_month    = label_maps['Month_Num'].get(month_num)

    if any(v is None for v in [enc_customer, enc_product, enc_month]):
        return None, "Encoding error — value not found in training data."

    row = {
        'Customer Name':    enc_customer,
        'Product':          enc_product,
        'Month_Num':        enc_month,
        'Quantity (KG)':    float(quantity),
        'Average Price/KG': float(avg_price),
    }
    X = pd.DataFrame([row])[feature_cols]
    pred = float(model.predict(X)[0])
    return round(pred, 2), None


if st.button("Predict Total Amount", type="primary", use_container_width=True):
    prediction, error = encode_and_predict(customer, product, month, quantity, avg_price)

    if error:
        st.error(f"Prediction error: {error}")
    else:
        expected = round(quantity * avg_price, 2)
        diff_pct  = abs(prediction - expected) / expected * 100

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Total Amount</div>
            <div class="result-amount">₹ {prediction:,.2f}</div>
            <div class="result-sub">
                <span class="info-pill">{customer}</span>
                <span class="info-pill">{product}</span>
                <span class="info-pill">{month}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Qty × Price", f"₹ {expected:,.2f}")
        c2.metric("Model Prediction", f"₹ {prediction:,.2f}")
        c3.metric("Variance", f"{diff_pct:.2f}%")

        # Bar chart
        fig = go.Figure(go.Bar(
            x=["Simple (Qty × Price)", "XGBoost Prediction"],
            y=[expected, prediction],
            marker_color=["#334155", "#0ea5e9"],
            text=[f"₹{expected:,.0f}", f"₹{prediction:,.0f}"],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="Amount (₹)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=20, b=20),
            height=280,
            font=dict(family="DM Sans"),
        )
        fig.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig, use_container_width=True)


# ── Batch Prediction ──────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">Batch Prediction (CSV Upload)</div>', unsafe_allow_html=True)
st.caption("Upload a CSV with columns: `Customer Name`, `Product`, `Month`, `Quantity (KG)`, `Average Price/KG`")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df_batch = pd.read_csv(uploaded)
    st.dataframe(df_batch.head(), use_container_width=True)

    results = []
    for _, row in df_batch.iterrows():
        try:
            pred, err = encode_and_predict(
                row.get('Customer Name', ''),
                row.get('Product', ''),
                row.get('Month', 'January'),
                row.get('Quantity (KG)', 0),
                row.get('Average Price/KG', 0),
            )
            results.append(pred if pred else "ERROR")
        except Exception as e:
            results.append("ERROR")

    df_batch['Predicted Total Amount'] = results
    st.success(f"✅ Predicted {len(df_batch)} rows")
    st.dataframe(df_batch, use_container_width=True)

    csv_out = df_batch.to_csv(index=False).encode()
    st.download_button("⬇ Download Predictions CSV", csv_out, "predictions.csv", "text/csv", use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("XGBoost model trained on 1,200 invoice records · Non-Woven Fabric sales prediction")
