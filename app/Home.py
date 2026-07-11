from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Satellite Land Use Classification & Temporal Change Detection",
    page_icon="🛰️",
    layout="wide",
    
)

# -------------------------------------------------
# Custom Styling
# -------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0f1720;
    }
    .app-title {
        font-size: 44px;
        font-weight: 800;
        color: #E8EEF5;
        margin-bottom: 0px;
    }
    .app-subtitle {
        font-size: 18px;
        color: #9AA9B8;
        margin-top: 6px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #16202B;
        border: 1px solid #24313F;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #4FB0E8;
    }
    .metric-label {
        font-size: 14px;
        color: #9AA9B8;
        margin-top: 4px;
    }
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #E8EEF5;
        margin-bottom: 12px;
    }
    .feature-item, .workflow-item {
        font-size: 16px;
        color: #C7D2DC;
        padding: 6px 0px;
    }
    .workflow-arrow {
        text-align: center;
        color: #4FB0E8;
        font-size: 18px;
        margin: 0px 0px 0px 18px;
    }
    .info-box {
        background-color: #16202B;
        border-left: 4px solid #4FB0E8;
        border-radius: 8px;
        padding: 20px 24px;
        color: #C7D2DC;
        font-size: 16px;
        line-height: 1.6;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Title & Subtitle
# -------------------------------------------------
st.markdown('<div class="app-title">🛰️ Satellite Land Use Classification & Temporal Change Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">An AI/ML system for classifying land-use patterns from satellite imagery '
    'and detecting temporal changes across time using deep feature embeddings.</div>',
    unsafe_allow_html=True,
)

st.divider()

# -------------------------------------------------
# Metric Cards
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">96.48%</div>
            <div class="metric-label">Classification Accuracy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">10</div>
            <div class="metric-label">Land Use Classes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">0.6345</div>
            <div class="metric-label">Change Detection Threshold</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.write("")

# -------------------------------------------------
# Features & Workflow
# -------------------------------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.markdown('<div class="section-header">Project Features</div>', unsafe_allow_html=True)
    features = [
        "Land-use classification",
        "Transfer Learning using ResNet18",
        "Temporal Change Detection",
        "Embedding-based Similarity",
        "Cosine Similarity",
        "Automatic Threshold Selection",
        "Heatmap Visualization",
    ]
    for feature in features:
        st.markdown(f'<div class="feature-item">• {feature}</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-header">Project Workflow</div>', unsafe_allow_html=True)
    workflow_steps = [
        "1. Dataset",
        "2. Training",
        "3. Classification",
        "4. Feature Embeddings",
        "5. Change Detection",
        "6. Visualization",
    ]
    for i, step in enumerate(workflow_steps):
        st.markdown(f'<div class="workflow-item">{step}</div>', unsafe_allow_html=True)
        if i != len(workflow_steps) - 1:
            st.markdown('<div class="workflow-arrow">↓</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Information Box
# -------------------------------------------------
st.markdown(
    """
    <div class="info-box">
    This project leverages deep learning to classify land-use categories from satellite imagery and
    monitor changes over time. Using transfer learning with a ResNet18 backbone, the system extracts
    robust feature embeddings that enable both accurate classification and cosine similarity-based
    change detection between temporal image pairs, visualized through intuitive heatmaps.
    </div>
    """,
    unsafe_allow_html=True,
)
st.success(
    """
    👈 Use the navigation menu to explore:

    • Land Use Classification

    • Temporal Change Detection
    """
)