"""Streamlit dashboard for Fake News Detection using saved model artifacts."""

import json
import os
import pickle
import re
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    TENSORFLOW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TENSORFLOW_AVAILABLE = False
    load_model = None
    pad_sequences = None


MAX_SEQUENCE_LEN = 300
MODEL_DIR = "model_artifacts"


st.set_page_config(
    page_title="Fake News Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --bg: #08111f;
            --panel: #0f1b2f;
            --panel-2: #13233c;
            --line: rgba(255, 255, 255, 0.08);
            --text: #e8eef9;
            --muted: #a8b3c7;
            --accent: #ff6b57;
            --accent-2: #f4b860;
            --real: #35c57a;
            --fake: #f04c5c;
            --shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
        }

        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 107, 87, 0.12), transparent 30%),
                radial-gradient(circle at top right, rgba(53, 197, 122, 0.10), transparent 28%),
                linear-gradient(180deg, #06101d 0%, #08111f 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1526 0%, #08111f 100%);
            border-right: 1px solid var(--line);
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(255, 107, 87, 0.12), rgba(53, 197, 122, 0.08));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.5rem 1.6rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .eyebrow {
            color: var(--accent-2);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            font-size: clamp(2rem, 3vw, 3rem);
            font-weight: 800;
            line-height: 1.1;
            margin: 0;
            color: var(--text);
        }

        .hero-subtitle {
            color: var(--muted);
            margin-top: 0.5rem;
            max-width: 60rem;
            line-height: 1.7;
        }

        .section-card {
            background: rgba(15, 27, 47, 0.92);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.2rem;
            box-shadow: var(--shadow);
        }

        .metric-box {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            text-align: center;
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
            box-shadow: var(--shadow);
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            color: var(--text);
        }
        .metric-label {
            font-size: 0.82rem;
            color: var(--muted);
            margin-top: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .stTextArea textarea {
            background: rgba(7, 16, 29, 0.92) !important;
            color: var(--text) !important;
            border: 1px solid var(--line) !important;
            border-radius: 16px !important;
            padding: 0.9rem !important;
            line-height: 1.6 !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: #fff;
            border: none;
            border-radius: 14px;
            font-weight: 700;
            padding: 0.7rem 1.2rem;
            box-shadow: 0 10px 24px rgba(255, 107, 87, 0.24);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(255, 107, 87, 0.30);
        }

        .stSelectbox [data-baseweb="select"] > div,
        .stFileUploader,
        .stDataFrame,
        .stJson {
            border-radius: 16px;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.03);
            color: var(--muted);
            font-size: 0.82rem;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }

        .pill-real { color: var(--real); border-color: rgba(53, 197, 122, 0.35); background: rgba(53, 197, 122, 0.08); }
        .pill-fake { color: var(--fake); border-color: rgba(240, 76, 92, 0.35); background: rgba(240, 76, 92, 0.08); }

        .callout {
            border-radius: 18px;
            padding: 1rem 1.1rem;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.03);
        }

        .result-card {
            border-radius: 20px;
            padding: 1.2rem;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.03);
            box-shadow: var(--shadow);
        }

        .result-real {
            background: linear-gradient(135deg, rgba(53, 197, 122, 0.16), rgba(53, 197, 122, 0.06));
            border-color: rgba(53, 197, 122, 0.35);
        }

        .result-fake {
            background: linear-gradient(135deg, rgba(240, 76, 92, 0.18), rgba(240, 76, 92, 0.06));
            border-color: rgba(240, 76, 92, 0.35);
        }

        .small-label {
            color: var(--muted);
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
            color: var(--text);
        }

        .sidebar-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hero(title: str, subtitle: str, eyebrow: str = "Fake News Detection") -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">{eyebrow}</div>
            <h1 class="hero-title">{title}</h1>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(label: str, confidence: float, real_prob: float) -> None:
    card_class = "result-real" if label == "REAL" else "result-fake"
    accent_class = "pill-real" if label == "REAL" else "pill-fake"
    display_color = "#35c57a" if label == "REAL" else "#f04c5c"
    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="small-label">Model decision</div>
            <h2 style="margin:0.35rem 0 0.2rem 0;color:{display_color};font-size:2rem;">{label}</h2>
            <div class="pill {accent_class}">Confidence: {confidence * 100:.2f}%</div>
            <div class="pill">Real probability: {real_prob * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts() -> Tuple[Optional[object], Optional[object], Optional[dict], dict]:
    """Load model artifacts from disk and return availability details."""
    info = {
        "model_path": os.path.join(MODEL_DIR, "lstm_fake_news_model.keras"),
        "tokenizer_path": os.path.join(MODEL_DIR, "tokenizer.pkl"),
        "metrics_path": os.path.join(MODEL_DIR, "metrics.json"),
        "cm_test_path": os.path.join(MODEL_DIR, "confusion_matrix.png"),
        "cm_val_path": os.path.join(MODEL_DIR, "unseen_confusion_matrix.png"),
    }

    model = None
    tokenizer = None
    metrics = None

    if not TENSORFLOW_AVAILABLE:
        return model, tokenizer, metrics, info

    if os.path.exists(info["model_path"]):
        model = load_model(info["model_path"])

    if os.path.exists(info["tokenizer_path"]):
        with open(info["tokenizer_path"], "rb") as f:
            tokenizer = pickle.load(f)

    if os.path.exists(info["metrics_path"]):
        with open(info["metrics_path"], "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return model, tokenizer, metrics, info


def preprocess_text(text: str) -> str:
    """Apply same normalization used during training."""
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)


def predict_one(model, tokenizer, text: str) -> Tuple[str, float, float]:
    """Run one prediction and return (label, confidence, real_probability)."""
    cleaned = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")
    real_probability = float(model.predict(padded, verbose=0)[0][0])

    if real_probability >= 0.5:
        label = "REAL"
        confidence = real_probability
    else:
        label = "FAKE"
        confidence = 1 - real_probability

    return label, float(confidence), real_probability


model, tokenizer, metrics, paths = load_artifacts()
model_ready = model is not None and tokenizer is not None and TENSORFLOW_AVAILABLE


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div style="font-size:1.25rem;font-weight:800;color:#fff;">📰 Fake News Dashboard</div>
            <div style="color:#a8b3c7;font-size:0.9rem;margin-top:0.35rem;">Saved model • Live inference • Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["Predict", "Analytics", "Batch Prediction", "About"],
        label_visibility="collapsed",
    )

    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("<div class='small-label'>Runtime status</div>", unsafe_allow_html=True)
    st.write("Artifacts folder:", MODEL_DIR)
    st.write("TensorFlow:", "Available" if TENSORFLOW_AVAILABLE else "Missing")
    st.write("Model:", "Loaded" if model is not None else "Missing")
    st.write("Tokenizer:", "Loaded" if tokenizer is not None else "Missing")
    st.markdown("</div>", unsafe_allow_html=True)


if not TENSORFLOW_AVAILABLE:
    st.error("TensorFlow is not available. Install dependencies from requirements.txt.")
    st.stop()

if page in {"Predict", "Batch Prediction"} and not model_ready:
    st.error("Saved model artifacts are missing or failed to load.")
    st.code("Expected: model_artifacts/lstm_fake_news_model.keras and model_artifacts/tokenizer.pkl")
    st.stop()


if page == "Predict":
    render_hero(
        "Fake News Detection",
        "<span class='small-label' style='font-size:0.95rem;color:#c6d1e2;'>Single Article Prediction</span><br>Paste a news article or headline and the saved model will classify it as REAL or FAKE with a confidence score.",
        eyebrow="Live Inference",
    )
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Input Article</div>", unsafe_allow_html=True)

        input_text = st.text_area(
            "Article text",
            value="",
            height=240,
            placeholder="Paste a news article, headline, or paragraph here...",
        )

        analyze = st.button("Run Prediction", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)

        if analyze:
            if not input_text.strip():
                st.warning("Please enter text before predicting.")
            else:
                label, confidence, real_prob = predict_one(model, tokenizer, input_text)
                fake_prob = 1 - real_prob

                render_result_card(label, confidence, real_prob)

                c1, c2, c3 = st.columns(3)
                c1.metric("Word Count", str(len(input_text.split())))
                c2.metric("Real %", f"{real_prob * 100:.2f}")
                c3.metric("Fake %", f"{fake_prob * 100:.2f}")

                chart_df = pd.DataFrame(
                    {
                        "Class": ["REAL", "FAKE"],
                        "Probability": [real_prob * 100, fake_prob * 100],
                    }
                )
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor("#0f1b2f")
                ax.set_facecolor("#0f1b2f")
                ax.barh(chart_df["Class"], chart_df["Probability"], color=["#35c57a", "#f04c5c"])
                ax.set_xlim(0, 100)
                ax.set_xlabel("Probability (%)", color="#e8eef9")
                ax.tick_params(colors="#e8eef9")
                for spine in ax.spines.values():
                    spine.set_color("#26364d")
                for i, v in enumerate(chart_df["Probability"]):
                    ax.text(v + 1, i, f"{v:.2f}%", va="center", color="#e8eef9")
                st.pyplot(fig)
                plt.close(fig)
        else:
            pass
        st.markdown("</div>", unsafe_allow_html=True)


elif page == "Analytics":
    render_hero(
        "Model Analytics",
        "View the saved model's test performance, validation performance, loss curves, and confusion matrices in one place.",
        eyebrow="Performance Overview",
    )

    if metrics is None:
        st.warning("metrics.json not found. Showing only available artifact images.")

    col1, col2, col3 = st.columns(3)
    with col1:
        test_acc = metrics.get("test_accuracy", 0.0) if metrics else 0.0
        st.markdown(
            f"<div class='metric-box'><div class='metric-value'>{test_acc:.2f}%</div>"
            "<div class='metric-label'>Test Accuracy</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        val_acc = metrics.get("unseen_accuracy", 0.0) if metrics else 0.0
        st.markdown(
            f"<div class='metric-box'><div class='metric-value'>{val_acc:.2f}%</div>"
            "<div class='metric-label'>Validation Accuracy</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        epochs_run = metrics.get("epochs_run", 0) if metrics else 0
        st.markdown(
            f"<div class='metric-box'><div class='metric-value'>{epochs_run}</div>"
            "<div class='metric-label'>Epochs Run</div></div>",
            unsafe_allow_html=True,
        )

    if metrics:
        train_acc = metrics.get("train_accuracy", [])
        val_acc_hist = metrics.get("val_accuracy", [])
        train_loss = metrics.get("train_loss", [])
        val_loss = metrics.get("val_loss", [])

        if train_acc and val_acc_hist:
            st.subheader("Accuracy Curves")
            fig, ax = plt.subplots(figsize=(8, 4))
            epochs = list(range(1, len(train_acc) + 1))
            ax.plot(epochs, train_acc, marker="o", label="Train Accuracy")
            ax.plot(epochs, val_acc_hist, marker="s", label="Validation Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Training vs Validation Accuracy")
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)

        if train_loss and val_loss:
            st.subheader("Loss Curves")
            fig, ax = plt.subplots(figsize=(8, 4))
            epochs = list(range(1, len(train_loss) + 1))
            ax.plot(epochs, train_loss, marker="o", label="Train Loss")
            ax.plot(epochs, val_loss, marker="s", label="Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training vs Validation Loss")
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.subheader("Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Test Set Confusion Matrix</div>", unsafe_allow_html=True)
        if os.path.exists(paths["cm_test_path"]):
            st.image(paths["cm_test_path"], width="stretch")
        else:
            st.info("confusion_matrix.png not found in model_artifacts.")
        st.markdown("</div>", unsafe_allow_html=True)

    with cm_col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Validation (Unseen) Confusion Matrix</div>", unsafe_allow_html=True)
        if os.path.exists(paths["cm_val_path"]):
            st.image(paths["cm_val_path"], width="stretch")
        else:
            st.info("unseen_confusion_matrix.png not found in model_artifacts.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Show raw metrics.json"):
        st.json(metrics if metrics else {})


elif page == "Batch Prediction":
    render_hero(
        "Batch Prediction",
        "Upload a CSV file to score many articles at once. The dashboard will add predictions and confidence columns automatically.",
        eyebrow="Bulk Workflow",
    )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.caption("CSV must include one text column: text or content")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.write("Rows loaded:", len(data))
        st.dataframe(data.head(5), width="stretch")

        text_col = None
        if "text" in data.columns:
            text_col = "text"
        elif "content" in data.columns:
            text_col = "content"

        if text_col is None:
            st.error("No valid text column found. Add either 'text' or 'content'.")
        else:
            if st.button("Run Batch Inference", width="stretch"):
                labels = []
                confidences = []
                real_probs = []

                progress = st.progress(0)
                total = len(data)

                for idx, row in data.iterrows():
                    label, conf, real_prob = predict_one(model, tokenizer, str(row[text_col]))
                    labels.append(label)
                    confidences.append(round(conf * 100, 2))
                    real_probs.append(round(real_prob * 100, 2))
                    progress.progress((idx + 1) / total)

                data["prediction"] = labels
                data["confidence_percent"] = confidences
                data["real_probability_percent"] = real_probs

                st.success("Batch inference completed.")
                st.dataframe(data, width="stretch")

                csv_bytes = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results CSV",
                    data=csv_bytes,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "About":
    render_hero(
        "About This Project",
        "A clean, end-to-end fake news detection system with a saved model, reproducible training notebook, and a dashboard designed for live demo use.",
        eyebrow="Project Summary",
    )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.write("Training notebook:", "Fake_News_Detection.ipynb")
    st.write("Artifacts path:", MODEL_DIR)

    st.subheader("Problem Statement")
    st.write(
        "This project addresses automatic identification of fake news by classifying "
        "news text into REAL or FAKE classes using an NLP deep learning pipeline."
    )

    st.subheader("Objectives")
    st.markdown(
        """
        - Build a reproducible fake-news classification workflow.
        - Evaluate with clear train/test/validation metrics.
        - Deploy a usable dashboard for single and batch inference.
        - Provide transparent evaluation artifacts and model health checks.
        """
    )

    st.subheader("Methodology")
    st.markdown(
        """
        1. Preprocess text: lowercase, clean URLs/symbols, stopword removal, stemming.
        2. Tokenize and pad to fixed-length sequences.
        3. Train Bidirectional LSTM model with dropout and early stopping.
        4. Evaluate using accuracy curves and confusion matrices.
        5. Serve saved model for real-time inference in this dashboard.
        """
    )

    checks = {
        "Model file": os.path.exists(paths["model_path"]),
        "Tokenizer file": os.path.exists(paths["tokenizer_path"]),
        "Metrics file": os.path.exists(paths["metrics_path"]),
        "Test confusion matrix": os.path.exists(paths["cm_test_path"]),
        "Validation confusion matrix": os.path.exists(paths["cm_val_path"]),
    }

    st.subheader("Artifact Health Check")
    health_df = pd.DataFrame(
        {
            "Artifact": list(checks.keys()),
            "Available": ["Yes" if value else "No" for value in checks.values()],
        }
    )
    st.dataframe(health_df, width="stretch", hide_index=True)

    if metrics:
        st.subheader("Key Outcomes")
        c1, c2 = st.columns(2)
        c1.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0.0):.2f}%")
        c2.metric("Validation Accuracy", f"{metrics.get('unseen_accuracy', 0.0):.2f}%")

    st.subheader("Limitations")
    st.markdown(
        """
        - Performance may degrade on out-of-domain or heavily novel content.
        - Binary labels do not capture nuanced misinformation categories.
        - Confidence score is not equivalent to fact-check certainty.
        """
    )

    st.subheader("References")
    st.markdown(
        """
        - Fake and Real News Dataset (Kaggle, Clément Bisaillon)
        - Hochreiter & Schmidhuber (1997), LSTM
        - Schuster & Paliwal (1997), Bidirectional RNN
        - TensorFlow/Keras and Streamlit official documentation
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
