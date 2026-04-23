"""
Fake News Detection — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pickle
import json
import os
import re
import time

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords', quiet=True)

# Graceful TensorFlow import with error handling
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TENSORFLOW_AVAILABLE = False
    load_model = None
    pad_sequences = None

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MAX_SEQUENCE_LEN = 300
MODEL_DIR = "Artifacts"

st.set_page_config(
    page_title="FakeDetect AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #e63946 0%, #f4a261 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .result-box-fake {
        background: linear-gradient(135deg, #ff4d4d15, #ff000008);
        border: 2px solid #e63946;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-box-real {
        background: linear-gradient(135deg, #2dc65315, #00ff5508);
        border: 2px solid #2dc653;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-label {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .result-score {
        font-size: 1rem;
        color: #aaa;
    }

    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #f4a261;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.3rem;
    }

    .section-title {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        color: #e0e0e0;
        border-left: 4px solid #e63946;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    .stTextArea textarea {
        background-color: #0d0d1a !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #e63946, #f4a261);
        color: white;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: transform 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(230, 57, 70, 0.4);
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'Syne', sans-serif;
    }

    .badge-fake { background: #e6394620; color: #e63946; border: 1px solid #e63946; }
    .badge-real { background: #2dc65320; color: #2dc653; border: 1px solid #2dc653; }

    [data-testid="stSidebar"] {
        background: #0d0d1a;
        border-right: 1px solid #1a1a2e;
    }

    .stApp {
        background: #070714;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODEL & ARTIFACTS
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not TENSORFLOW_AVAILABLE:
        return None, None, None
    
    model_path = os.path.join(MODEL_DIR, "lstm_fake_news_model.keras")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None, None, None

    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return model, tokenizer, metrics


def preprocess_input(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


def predict(model, tokenizer, text):
    cleaned = preprocess_input(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LEN, padding='post', truncating='post')
    prob = model.predict(padded, verbose=0)[0][0]
    label = "REAL" if prob > 0.5 else "FAKE"
    confidence = prob if prob > 0.5 else (1 - prob)
    return label, float(confidence), float(prob)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.4rem; font-weight: 800; color: #e63946;'>🔍 FakeDetect AI</div>
        <div style='font-size: 0.75rem; color: #555; margin-top: 0.3rem;'>LSTM · NLP · Deep Learning</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Detect News", "📊 Model Analytics", "📋 Batch Analysis", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.75rem; color: #555; padding: 0.5rem;'>
        <b style='color:#888'>Model:</b> Bidirectional LSTM<br>
        <b style='color:#888'>Dataset:</b> WELFake / Kaggle<br>
        <b style='color:#888'>Task:</b> Binary Classification
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────
model, tokenizer, metrics = load_artifacts()
model_loaded = model is not None

# ─────────────────────────────────────────
# PAGE 1: DETECT NEWS
# ─────────────────────────────────────────
if "🏠 Detect News" in page:
    st.markdown('<div class="main-header">Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Bidirectional LSTM · Enter a news article to analyze</div>', unsafe_allow_html=True)

    if not model_loaded:
        if not TENSORFLOW_AVAILABLE:
            st.error("⚠️ TensorFlow is not available in this environment. The model requires TensorFlow to run.")
            st.info("This is a known limitation with Python 3.14+. Please use Python 3.9-3.12 for deployment.")
        else:
            st.error("⚠️ Model not found! Please run `python model_training.py` first to train and save the model.")
            st.code("python model_training.py", language="bash")
        st.stop()

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="section-title">Input Article</div>', unsafe_allow_html=True)

        # Quick test examples
        st.write("**Quick Examples:**")
        examples = {
            "Example 1: Real News": "U.S. Stock Market Rises on Positive Economic Data. The S&P 500 index gained 1.2 percent today following the release of stronger than expected employment figures. Analysts attribute the rise to investor confidence in the economic recovery. Federal Reserve officials are expected to comment on the data at their next meeting.",
            "Example 2: Real News": "World Health Organization Announces New Safety Guidelines. Health officials from around the world gathered at the annual summit to discuss pandemic preparedness. The new protocols aim to improve coordination between nations during health emergencies. Scientists emphasized the importance of early detection systems.",
            "Example 3: Fake News": "SHOCKING: World leaders caught in secret meetings controlling weather patterns. This hidden government experiment has been happening for 50 years. Click here before they take this down!",
            "Example 4: Fake News": "Celebrity endorses miracle weight loss pill that doctors dont want you to know about. Lose 50 pounds in 2 weeks guaranteed or your money back. Limited time offer ends today!",
        }

        selected_example = st.selectbox("Load a demo example:", list(examples.keys()), index=None, label_visibility="collapsed")
        
        if selected_example:
            st.session_state.news_input = examples[selected_example]

        news_text = st.text_area(
            "Paste your news article or headline here:",
            value=st.session_state.get("news_input", ""),
            height=220,
            placeholder="Paste any news article, headline or paragraph here...",
            label_visibility="collapsed"
        )

        analyze_btn = st.button("🔍 Analyze Article")

    with col2:
        st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

        if analyze_btn and news_text.strip():
            with st.spinner("Analyzing..."):
                time.sleep(0.5)
                label, confidence, prob = predict(model, tokenizer, news_text)

            if label == "FAKE":
                st.markdown(f"""
                <div class="result-box-fake">
                    <div class="result-label" style="color: #e63946;">⚠️ FAKE NEWS</div>
                    <div class="result-score">Confidence: {confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-real">
                    <div class="result-label" style="color: #2dc653;">✅ REAL NEWS</div>
                    <div class="result-score">Confidence: {confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown("**Probability Breakdown**")
            prob_df = pd.DataFrame({
                'Category': ['Real News', 'Fake News'],
                'Probability': [prob * 100, (1 - prob) * 100]
            })

            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_facecolor('#0d0d1a')
            ax.set_facecolor('#0d0d1a')

            colors = ['#2dc653', '#e63946']
            bars = ax.barh(prob_df['Category'], prob_df['Probability'],
                           color=colors, height=0.5, edgecolor='none')

            for bar, val in zip(bars, prob_df['Probability']):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', color='white', fontsize=10, fontweight='bold')

            ax.set_xlim(0, 115)
            ax.set_xlabel('')
            ax.tick_params(colors='#888')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Word count info
            word_count = len(news_text.split())
            st.markdown(f"<small style='color:#555'>Article word count: {word_count} words</small>", unsafe_allow_html=True)

        elif analyze_btn:
            st.warning("Please enter some text to analyze.")
        else:
            st.markdown("""
            <div style='text-align:center; color:#333; padding: 3rem 1rem;'>
                <div style='font-size: 3rem;'>🔍</div>
                <div style='font-family: Syne, sans-serif; font-size: 1rem; margin-top: 0.5rem;'>Results will appear here</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE 2: MODEL ANALYTICS
# ─────────────────────────────────────────
elif "📊 Model Analytics" in page:
    st.markdown('<div class="main-header">Model Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Training performance and evaluation metrics</div>', unsafe_allow_html=True)

    if not model_loaded or metrics is None:
        st.error("⚠️ Model artifacts not found. Run `python model_training.py` first.")
        st.stop()

    # Metric cards
    c1, c2 = st.columns(2)
    with c1:
        accuracy = metrics.get('accuracy', 0) * 100
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{accuracy:.2f}%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">Bi-LSTM</div>
            <div class="metric-label">Model Architecture</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    cm = np.array(metrics.get('confusion_matrix', [[0, 0], [0, 0]]))
    
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'],
                cbar_kws={'label': 'Count'},
                ax=ax)
    ax.set_xlabel('Predicted', color='#e0e0e0', fontsize=12)
    ax.set_ylabel('True Label', color='#e0e0e0', fontsize=12)
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#2a2a4a')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Architecture
    st.markdown('<div class="section-title">Model Architecture</div>', unsafe_allow_html=True)
    arch_data = {
        "Layer": ["Embedding", "SpatialDropout1D", "Bidirectional LSTM (64)", "Dropout", "LSTM (32)", "Dropout", "Dense (32, ReLU)", "Dropout", "Dense (1, Sigmoid)"],
        "Output Shape": ["(300, 128)", "(300, 128)", "(300, 128)", "(300, 128)", "(32,)", "(32,)", "(32,)", "(32,)", "(1,)"],
        "Purpose": ["Word vectors", "Regularization", "Sequence learning ↔", "Overfitting control", "Sequence compression", "Overfitting control", "Feature extraction", "Overfitting control", "Binary output"]
    }
    df_arch = pd.DataFrame(arch_data)
    st.dataframe(df_arch, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────
# PAGE 3: BATCH ANALYSIS
# ─────────────────────────────────────────
elif "📋 Batch Analysis" in page:
    st.markdown('<div class="main-header">Batch Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV file to analyze multiple articles at once</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ Model not found. Run `python model_training.py` first.")
        st.stop()

    st.markdown('<div class="section-title">Upload CSV File</div>', unsafe_allow_html=True)
    st.markdown("Your CSV should have a column named **`text`** containing the news articles.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'text' not in df.columns:
            st.error("CSV must contain a column named 'text'")
        else:
            st.success(f"Loaded {len(df)} articles")
            st.dataframe(df.head(3), use_container_width=True)

            if st.button("🔍 Run Batch Analysis"):
                progress = st.progress(0)
                results = []
                for i, row in df.iterrows():
                    label, conf, prob = predict(model, tokenizer, str(row['text']))
                    results.append({'label': label, 'confidence': round(conf * 100, 1), 'real_prob': round(prob * 100, 1)})
                    progress.progress((i + 1) / len(df))

                df['Prediction'] = [r['label'] for r in results]
                df['Confidence'] = [r['confidence'] for r in results]
                df['Real_Probability'] = [r['real_prob'] for r in results]

                # Summary
                fake_count = (df['Prediction'] == 'FAKE').sum()
                real_count = (df['Prediction'] == 'REAL').sum()

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#e63946">{fake_count}</div><div class="metric-label">Fake Articles</div></div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#2dc653">{real_count}</div><div class="metric-label">Real Articles</div></div>""", unsafe_allow_html=True)
                with c3:
                    pct = round(real_count / len(df) * 100, 1)
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{pct}%</div><div class="metric-label">Real Percentage</div></div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Pie chart
                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_facecolor('#0d0d1a')
                ax.set_facecolor('#0d0d1a')
                ax.pie([real_count, fake_count], labels=['Real', 'Fake'],
                       colors=['#2dc653', '#e63946'], autopct='%1.1f%%',
                       textprops={'color': '#e0e0e0'}, startangle=90,
                       wedgeprops={'edgecolor': '#0d0d1a', 'linewidth': 2})
                ax.set_title('Prediction Distribution', color='#e0e0e0', fontweight='bold')
                st.pyplot(fig)
                plt.close()

                # Results table
                st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)

                # Download
                csv = df.to_csv(index=False)
                st.download_button("⬇️ Download Results CSV", csv, "fake_news_results.csv", "text/csv")


# ─────────────────────────────────────────
# PAGE 4: ABOUT
# ─────────────────────────────────────────
elif "ℹ️ About" in page:
    st.markdown('<div class="main-header">About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#1a1a2e; border-radius:16px; padding:2rem; border: 1px solid #2a2a4a; margin-bottom:1.5rem;'>
        <div style='font-family:Syne,sans-serif; font-size:1.2rem; font-weight:700; color:#f4a261; margin-bottom:1rem;'>🎯 Project Overview</div>
        <p style='color:#bbb; line-height:1.8;'>
        This project builds a <strong style='color:#e0e0e0'>Fake News Detection system</strong> using deep learning.
        A Bidirectional LSTM model is trained on thousands of real and fake news articles to learn linguistic
        patterns that distinguish misinformation from credible reporting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#1a1a2e; border-radius:16px; padding:1.5rem; border: 1px solid #2a2a4a;'>
            <div style='font-family:Syne,sans-serif; font-weight:700; color:#e63946; margin-bottom:0.8rem;'>🧠 Technical Stack</div>
            <ul style='color:#bbb; line-height:2;'>
                <li><strong>Model:</strong> Bidirectional LSTM</li>
                <li><strong>Framework:</strong> TensorFlow / Keras</li>
                <li><strong>NLP:</strong> NLTK (stemming, stopwords)</li>
                <li><strong>Dashboard:</strong> Streamlit</li>
                <li><strong>Dataset:</strong> WELFake (Kaggle)</li>
                <li><strong>Language:</strong> Python 3.8+</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#1a1a2e; border-radius:16px; padding:1.5rem; border: 1px solid #2a2a4a;'>
            <div style='font-family:Syne,sans-serif; font-weight:700; color:#2dc653; margin-bottom:0.8rem;'>🔄 Pipeline</div>
            <ol style='color:#bbb; line-height:2;'>
                <li>Load & label True/Fake CSVs</li>
                <li>Clean text (lowercase, remove URLs, stem)</li>
                <li>Tokenize & pad sequences</li>
                <li>Train Bidirectional LSTM</li>
                <li>Evaluate with accuracy + confusion matrix</li>
                <li>Deploy with Streamlit dashboard</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style='background:#1a1a2e; border-radius:16px; padding:1.5rem; border: 1px solid #2a2a4a;'>
        <div style='font-family:Syne,sans-serif; font-weight:700; color:#f4a261; margin-bottom:0.8rem;'>📁 How to Run</div>
        <div style='color:#bbb;'>
            <strong style='color:#e0e0e0;'>Step 1:</strong> Download dataset from Kaggle (True.csv + Fake.csv)<br>
            <a href='https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset' style='color:#e63946;'>
            kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset</a><br><br>
            <strong style='color:#e0e0e0;'>Step 2:</strong> Install dependencies<br>
            <code style='background:#0d0d1a; padding:2px 8px; border-radius:4px;'>pip install -r requirements.txt</code><br><br>
            <strong style='color:#e0e0e0;'>Step 3:</strong> Train the model<br>
            <code style='background:#0d0d1a; padding:2px 8px; border-radius:4px;'>python model_training.py</code><br><br>
            <strong style='color:#e0e0e0;'>Step 4:</strong> Launch dashboard<br>
            <code style='background:#0d0d1a; padding:2px 8px; border-radius:4px;'>streamlit run app.py</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
