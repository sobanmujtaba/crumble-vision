import streamlit as st
import pandas as pd
import cv2
import plotly.express as px
from sklearn.metrics import confusion_matrix
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Crumble Pakistan | Quality Assurance",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">

<style>
    :root {
        --primary: #000059;
        --background: #fbf9f1;
        --surface: #f6f4ec;
        --surface-white: #ffffff;
        --cream:#fffee0;
        --outline: #d8d6cf;
        --text-muted: #5d5e67;
        --error: #ba1a1a;
    }

    .stApp {
        background-color: var(--background);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3 {
        color: var(--primary) !important;
        font-weight: 800;
    }

    .metric-box {
        background: white;
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid var(--outline);
        border-bottom: 4px solid var(--primary);
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: var(--text-muted);
    }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: var(--primary);
    }

    .card {
        background: white;
        border-radius: 18px;
        padding: 1.25rem;
        border: 1px solid var(--outline);
    }

    .progress-track {
        width: 100%;
        height: 12px;
        background: #e6e4dd;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .progress-fill {
        height: 100%;
        background: var(--primary);
        border-radius: 999px;
    }

    .sidebar-title {
        font-size: 2rem;
        font-weight: 800;
        color: var(--cream);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------- DATA ----------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("assets/trays/tray_predictions.csv")
        df["correct"] = df["true_label"] == df["predicted_label"]
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

tray_ids = sorted(df["tray_id"].unique())

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("""
    <div style="padding:2rem 0;">
        <div class="sidebar-title">CRUMBLE</div>
        <div style="text-align:center; font-size:0.7rem; letter-spacing:2px; font-weight:700;">
            Pakistan Ops
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<h1 style="font-size:3rem;">Quality Assurance</h1>
<p style="font-size:1.1rem; color:#5d5e67;">
AI-driven monitoring for the perfect batch
</p>
""", unsafe_allow_html=True)

# ---------- TOP METRICS ----------
m1, m2, m3, m4 = st.columns(4)

accuracy = df["correct"].mean() * 100
defects = (df["predicted_label"] != "Defect_No").sum()

with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Processed Trays</div>
        <div class="metric-value">{len(tray_ids)}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Total Cookies</div>
        <div class="metric-value">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">AI Accuracy</div>
        <div class="metric-value">{accuracy:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Flagged Items</div>
        <div class="metric-value" style="color:#ba1a1a;">{defects}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- CONFUSION MATRIX ----------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Performance Matrix")

CLASS_NAMES = [
    "Defect_No",
    "Defect_Shape",
    "Defect_Object",
    "Defect_Color"
]

cm = confusion_matrix(
    df["true_label"],
    df["predicted_label"],
    labels=CLASS_NAMES
)

fig = px.imshow(
    cm,
    text_auto=True,
    x=CLASS_NAMES,
    y=CLASS_NAMES,
    color_continuous_scale=["#fbf9f1", "#cfe2f7", "#000059"],
    aspect="auto"
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=10, b=10)
)

st.plotly_chart(fig, use_container_width=True)

# ---------- TRAY FOCUS ----------
st.markdown("---")
st.subheader("Tray Focus")

selected_tray = st.selectbox("Select Active Tray", tray_ids)

tray_df = df[df["tray_id"] == selected_tray]
pass_rate = (tray_df["predicted_label"] == "Defect_No").mean() * 100

col1, col2, col3 = st.columns([1, 1.25, 1.35])

# LEFT PANEL
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("##### PASS RATE")
    st.markdown(
        f"<h1 style='color:#000059; margin-top:-10px;'>{pass_rate:.1f}%</h1>",
        unsafe_allow_html=True
    )

    st.progress(pass_rate / 100)

    st.markdown(f"**Tray ID:** {selected_tray}")

    st.markdown('</div>', unsafe_allow_html=True)

# CENTER PANEL
with col2:
    st.markdown("##### Prediction Table")
    st.dataframe(
        tray_df[
            ["row", "col", "true_label", "predicted_label", "correct"]
        ],
        use_container_width=True,
        hide_index=True
    )

# RIGHT PANEL
with col3:
    st.markdown("##### Tray Image")

    possible_paths = [
        f"assets/trays/{selected_tray}.jpg",
        f"assets/trays/{selected_tray}.png",
        f"assets/trays/{selected_tray}.jpeg"
    ]

    image_path = next(
        (p for p in possible_paths if os.path.exists(p)),
        None
    )

    if image_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(
            image,
            caption=f"Tray {selected_tray}",
            use_container_width=True
        )
    else:
        st.warning("Tray image not found.")

# ---------- FOOTER ----------
st.markdown("""
<div style="margin-top: 4rem; text-align: center; color: #5d5e67; font-size: 0.8rem; font-weight: 700; letter-spacing: 1px;">
    CRUMBLE PAKISTAN | <a href="https://sobanmujtaba.github.io/" target="_blank" style="color: #5d5e67; text-decoration: none; border-bottom: 1px solid #c6c5d6;">SOBANMUJTABA</a>
</div>
""", unsafe_allow_html=True)
