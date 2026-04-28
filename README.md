# 🔍 Fake News Detection using Bidirectional LSTM

A deep learning project that detects fake news using a Bidirectional LSTM neural network, with a Streamlit dashboard for interactive use.

---

## 🎯 Problem Statement & Objectives

Fake and misleading online news can spread quickly, making manual verification difficult. This project builds an NLP-based binary classifier to label news text as REAL or FAKE and provides an interactive dashboard for practical use.

Primary objectives:
- Build a reproducible deep learning pipeline for fake-news classification.
- Train and evaluate a Bidirectional LSTM model with clear train/test/validation splits.
- Deploy an interactive dashboard that supports single and batch inference.
- Provide transparent evaluation artifacts (metrics, confusion matrices, training curves).

---

## 📁 Project Structure

```
fake_news_detection/
│
├── app.py                          # 🎯 Streamlit dashboard (main application)
├── model_training.ipynb            # 📓 Model training notebook
├── requirements.txt                # 📦 Python dependencies
├── README.md                       # 📄 This file
│
├── True.csv                        # 📊 Real news dataset (21,417 articles)
├── Fake.csv                        # 📊 Fake news dataset (23,481 articles)
│
├── model_artifacts/                # 🤖 Trained model & artifacts
│   ├── lstm_fake_news_model.keras  #    Final trained model
│   ├── best_model.keras            #    Best checkpoint
│   ├── tokenizer.pkl               #    Text tokenizer
│   ├── metrics.json                #    Training metrics & accuracies
│   ├── training_history.png        #    Loss/accuracy curves
│   ├── confusion_matrix.png        #    Test set confusion matrix
│   └── unseen_confusion_matrix.png #    Validation set confusion matrix
│
├── datasets/                       # 📁 Split metadata (CSVs generated locally)
│   └── dataset_metadata.json       #    Split statistics
│
├── save_datasets.py                # 🔧 Script to create train/test/val splits
├── test_model.py                   # ✅ Script to test model inference
├── DEPLOYMENT.md                   # 🚀 Cloud deployment guide
├── Dockerfile                      # 🐳 Docker configuration
└── packages.txt                    # 📦 Docker system dependencies
```

---

## 🚀 Quick Start

### Option 1: Use Pre-Trained Model ⚡ (Recommended)

The model is already trained and saved in `model_artifacts/`. Just run the dashboard!

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch dashboard
streamlit run app.py
```

Open your browser at `http://localhost:8501` → **Done!**

---

### Option 2: Train Your Own Model 🔄

If you want to retrain the model from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open and run the training notebook
jupyter notebook model_training.ipynb
```

This will:
- Preprocess and clean the text data from CSV files
- Tokenize and pad sequences (max 300 words)
- Train a Bidirectional LSTM model (7 epochs)
- Save the model, tokenizer, plots, and metrics to `model_artifacts/`
- Generate test & unseen validation metrics

⏱️ Training time: ~10-20 minutes depending on your hardware.

Then run the dashboard:
```bash
streamlit run app.py
```

---

### Option 3: Test the Model 🧪

Test the trained model with sample news articles:

```bash
python test_model.py
```

Or save train/test/validation dataset splits:

```bash
python save_datasets.py
```

---

## ☁️ Cloud Deployment

**⚠️ Note:** Streamlit Cloud uses Python 3.14+ which doesn't support TensorFlow.

### Recommended: Hugging Face Spaces 🤗

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete setup guide.

### Docker Options
- Railway.app
- Render.com
- Any Docker-compatible platform

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

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

## 🔄 Methodology Summary

1. Data acquisition: load [True.csv](True.csv) and [Fake.csv](Fake.csv).
2. Labeling: REAL = 1, FAKE = 0.
3. Text preprocessing: lowercase, URL removal, non-alphabet removal, stopword removal, stemming.
4. Tokenization: top 10,000 vocabulary with OOV handling.
5. Sequence padding: fixed length of 300 tokens.
6. Data split: 80% train, 10% test, 10% unseen validation (stratified).
7. Modeling: Embedding + BiLSTM + LSTM + Dense sigmoid output.
8. Regularization: dropout, spatial dropout, early stopping, best-model checkpointing.
9. Evaluation: accuracy, loss curves, confusion matrices, and unseen validation performance.

---

## 📊 Dashboard Features

| Page | Feature |
|---|---|
| 🏠 Predict | Analyze single article with real-time prediction & confidence score |
| 📊 Model Analytics | View validation metrics, training curves, confusion matrix, architecture |
| 📋 Batch Prediction | Upload CSV file & get predictions for all articles |
| ℹ️ About | Project documentation and pipeline overview |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **99.92%** ✨ |
| **Validation Accuracy** | **99.91%** ✨ |
| **Training Epochs** | 7 (with early stopping) |
| **Data Split** | 80% train, 10% test, 10% validation |
| **Dataset** | 44,898 articles (21,417 real + 23,481 fake) |
| **Overfitting Control** | Dropout (0.3), SpatialDropout1D (0.2), Early Stopping |

---

## ⚠️ Limitations

- Domain sensitivity: model quality can drop on topics/styles not represented in the training data.
- Label noise risk: public datasets can contain imperfect labels.
- Confidence is not calibrated certainty: high confidence does not guarantee factual correctness.
- Binary framing: nuanced misinformation types are reduced to REAL/FAKE labels.

---

## 🎤 Presentation & Viva Readiness

Use this checklist before the final demo:

- Show objective and problem relevance using the section above.
- Run [app.py](app.py) and demonstrate all four pages.
- Show test/validation metrics and both confusion matrices in Analytics.
- Perform one live single-text prediction and one batch CSV prediction.
- Explain preprocessing, architecture choice, and overfitting controls.
- Explain one failure case and one planned improvement.

Common viva questions you should be ready for:

- Why use Bidirectional LSTM instead of simple RNN or traditional ML?
- Why keep a separate unseen validation split after training?
- What does confusion matrix reveal beyond accuracy?
- How would you improve robustness for out-of-domain news?

---

## 📚 References

1. Kaggle dataset: Fake and Real News Dataset by Clément Bisaillon.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
3. Schuster, M., & Paliwal, K. K. (1997). Bidirectional Recurrent Neural Networks.
4. TensorFlow/Keras Documentation: https://www.tensorflow.org/
5. Streamlit Documentation: https://docs.streamlit.io/

---

## 🛠️ Tech Stack
- **Python 3.9–3.12** (not 3.14+)
- **TensorFlow 2.13+** — Bidirectional LSTM model
- **NLTK** — Tokenization, stemming, stopword removal
- **Streamlit** — Interactive web dashboard
- **Matplotlib / Seaborn** — Training visualization
- **Scikit-learn** — Performance metrics
