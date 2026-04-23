"""
Fake News Detection - Model Training Script
Uses LSTM (Long Short-Term Memory) neural network
Dataset: WELFake / Kaggle Fake News dataset (True.csv + Fake.csv)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import warnings
warnings.filterwarnings('ignore')

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
MAX_VOCAB_SIZE = 10000     # Top N most frequent words
MAX_SEQUENCE_LEN = 300     # Max words per article
EMBEDDING_DIM = 128        # Word embedding dimensions
LSTM_UNITS = 64            # LSTM hidden units
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
EPOCHS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

OUTPUT_DIR = "model_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
def load_data():
    """
    Load True.csv and Fake.csv from the current directory.
    Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    """
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)

    try:
        true_df = pd.read_csv("True.csv")
        fake_df = pd.read_csv("Fake.csv")
    except FileNotFoundError:
        print("\n[ERROR] True.csv or Fake.csv not found!")
        print("Please download the dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("Place True.csv and Fake.csv in the same folder as this script.\n")
        raise

    # Label: 1 = Real, 0 = Fake
    true_df['label'] = 1
    fake_df['label'] = 0

    print(f"  Real news articles : {len(true_df)}")
    print(f"  Fake news articles : {len(fake_df)}")

    # Combine title + text for richer features
    true_df['content'] = true_df['title'].fillna('') + " " + true_df['text'].fillna('')
    fake_df['content'] = fake_df['title'].fillna('') + " " + fake_df['text'].fillna('')

    df = pd.concat([true_df[['content', 'label']], fake_df[['content', 'label']]], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"  Total samples      : {len(df)}")
    print(f"  Label distribution :\n{df['label'].value_counts()}\n")
    return df


# ─────────────────────────────────────────
# STEP 2: TEXT PREPROCESSING
# ─────────────────────────────────────────
def preprocess_text(text):
    """Clean and normalize a single text string."""
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords and apply stemming
    tokens = [ps.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


def preprocess_data(df):
    print("=" * 60)
    print("STEP 2: Preprocessing Text")
    print("=" * 60)
    print("  Cleaning text (this may take a minute)...")
    df['cleaned'] = df['content'].apply(preprocess_text)
    print(f"  Sample cleaned text:\n  {df['cleaned'].iloc[0][:150]}...\n")
    return df


# ─────────────────────────────────────────
# STEP 3: TOKENIZE & PAD
# ─────────────────────────────────────────
def tokenize_and_pad(df):
    print("=" * 60)
    print("STEP 3: Tokenizing & Padding Sequences")
    print("=" * 60)

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned'])

    sequences = tokenizer.texts_to_sequences(df['cleaned'])
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN, padding='post', truncating='post')

    print(f"  Vocabulary size    : {len(tokenizer.word_index)}")
    print(f"  Sequence shape     : {padded.shape}\n")

    # Save tokenizer
    with open(os.path.join(OUTPUT_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"  Tokenizer saved to {OUTPUT_DIR}/tokenizer.pkl\n")

    return padded, tokenizer


# ─────────────────────────────────────────
# STEP 4: BUILD LSTM MODEL
# ─────────────────────────────────────────
def build_model():
    print("=" * 60)
    print("STEP 4: Building LSTM Model")
    print("=" * 60)

    model = Sequential([
        Embedding(input_dim=MAX_VOCAB_SIZE,
                  output_dim=EMBEDDING_DIM,
                  input_length=MAX_SEQUENCE_LEN),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
        Dropout(DROPOUT_RATE),
        LSTM(32),
        Dropout(DROPOUT_RATE),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')   # Binary: Real(1) or Fake(0)
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()
    return model


# ─────────────────────────────────────────
# STEP 5: TRAIN MODEL
# ─────────────────────────────────────────
def train_model(model, X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("STEP 5: Training Model")
    print("=" * 60)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model.keras"),
                        monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    return history


# ─────────────────────────────────────────
# STEP 6: EVALUATE & SAVE PLOTS
# ─────────────────────────────────────────
def evaluate_and_save(model, history, X_test, y_test):
    print("\n" + "=" * 60)
    print("STEP 6: Evaluation & Saving Artifacts")
    print("=" * 60)

    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc * 100:.2f}%\n")
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

    # ── Plot 1: Training History ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Model Training History", fontsize=14, fontweight='bold')

    axes[0].plot(history.history['accuracy'], label='Train Acc', color='steelblue', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Acc', color='coral', linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train Loss', color='steelblue', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='coral', linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 2: Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'],
                linewidths=0.5)
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Save final model
    model.save(os.path.join(OUTPUT_DIR, "lstm_fake_news_model.keras"))

    # Save training metrics as JSON for dashboard
    import json
    metrics = {
        "test_accuracy": round(acc * 100, 2),
        "train_accuracy": [round(v * 100, 2) for v in history.history['accuracy']],
        "val_accuracy": [round(v * 100, 2) for v in history.history['val_accuracy']],
        "train_loss": [round(v, 4) for v in history.history['loss']],
        "val_loss": [round(v, 4) for v in history.history['val_loss']],
        "epochs_run": len(history.history['accuracy'])
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  All artifacts saved to '{OUTPUT_DIR}/' folder:")
    print(f"    - lstm_fake_news_model.keras")
    print(f"    - tokenizer.pkl")
    print(f"    - training_history.png")
    print(f"    - confusion_matrix.png")
    print(f"    - metrics.json")

    return acc


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "🔍 FAKE NEWS DETECTION — LSTM MODEL TRAINING ".center(60, "="))
    print()

    # Load
    df = load_data()

    # Preprocess
    df = preprocess_data(df)

    # Tokenize
    X, tokenizer = tokenize_and_pad(df)
    y = df['label'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train samples: {len(X_train)} | Test samples: {len(X_test)}\n")

    # Build
    model = build_model()

    # Train
    history = train_model(model, X_train, X_test, y_train, y_test)

    # Evaluate
    acc = evaluate_and_save(model, history, X_test, y_test)

    print("\n" + "✅ TRAINING COMPLETE ".center(60, "="))
    print(f"  Final Test Accuracy: {acc * 100:.2f}%")
    print("  Run 'streamlit run app.py' to launch the dashboard!\n")
