# 🔍 Fake News Detection using Bidirectional LSTM

A deep learning project that detects fake news using a Bidirectional LSTM neural network, with a Streamlit dashboard for interactive use.

---

## 📁 Project Structure

```
fake_news_detection/
│
├── model_training.py       # Train the LSTM model
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── True.csv                # ← You download this from Kaggle
├── Fake.csv                # ← You download this from Kaggle
│
└── model_artifacts/        # Auto-created after training
    ├── lstm_fake_news_model.keras
    ├── tokenizer.pkl
    ├── training_history.png
    ├── confusion_matrix.png
    └── metrics.json
```

---

## 🚀 Setup & Run

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download Dataset
Download from Kaggle:
👉 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place `True.csv` and `Fake.csv` in the **same folder** as `model_training.py`.

### Step 3 — Train the Model
```bash
python model_training.py
```
This will:
- Preprocess and clean the text data
- Tokenize and pad sequences
- Train a Bidirectional LSTM model
- Save the model, tokenizer, plots, and metrics to `model_artifacts/`

⏱️ Training time: ~10–20 minutes depending on your hardware.

### Step 4 — Launch Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 🧠 Model Architecture

| Layer | Details |
|---|---|
| Embedding | 10,000 vocab, 128 dims |
| SpatialDropout1D | 0.2 |
| Bidirectional LSTM | 64 units, return sequences |
| Dropout | 0.3 |
| LSTM | 32 units |
| Dropout | 0.3 |
| Dense | 32 units, ReLU |
| Dense | 1 unit, Sigmoid (output) |

---

## 📊 Dashboard Features

| Page | What it shows |
|---|---|
| 🏠 Detect News | Single article prediction with confidence score |
| 📊 Model Analytics | Training curves, confusion matrix, architecture |
| 📋 Batch Analysis | Upload CSV → get predictions for all articles |
| ℹ️ About | Project overview and pipeline explanation |

---

## 📈 Expected Results
- Test Accuracy: ~98–99% on the WELFake dataset
- Model uses Early Stopping to avoid overfitting

---

## 🛠️ Tech Stack
- **Python 3.8+**
- **TensorFlow / Keras** — LSTM model
- **NLTK** — Text preprocessing
- **Streamlit** — Web dashboard
- **Matplotlib / Seaborn** — Visualizations
- **Scikit-learn** — Evaluation metrics
