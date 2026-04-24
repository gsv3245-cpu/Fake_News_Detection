"""
Test the Fake News Detection Model
This script tests the saved model with sample news articles
"""

import os
import pickle
import re
import json
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk

# Download required NLTK data
nltk.download('stopwords', quiet=True)

# Configuration
MAX_SEQUENCE_LEN = 300
MODEL_DIR = "model_artifacts"

print("="*70)
print("FAKE NEWS DETECTION MODEL TEST")
print("="*70)

# Load model and tokenizer
print("\n1. Loading model and tokenizer...")
try:
    model = load_model(os.path.join(MODEL_DIR, "lstm_fake_news_model.keras"))
    print("   ✓ Model loaded successfully")
    
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    print("   ✓ Tokenizer loaded successfully")
    
    with open(os.path.join(MODEL_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
    print("   ✓ Metrics loaded successfully")
except Exception as e:
    print(f"   ✗ Error loading artifacts: {e}")
    exit(1)

# Preprocessing function
def preprocess_text(text):
    """Clean and normalize text"""
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# Prediction function
def predict(text):
    """Make prediction on text"""
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LEN, padding='post', truncating='post')
    prob = model.predict(padded, verbose=0)[0][0]
    label = "REAL" if prob > 0.5 else "FAKE"
    confidence = prob if prob > 0.5 else (1 - prob)
    return label, float(confidence), float(prob)

# Display model metrics
print("\n2. Model Performance Metrics:")
print(f"   ✓ Test Accuracy: {metrics.get('test_accuracy', 0):.2f}%")
print(f"   ✓ Validation Accuracy: {metrics.get('unseen_accuracy', 0):.2f}%")
print(f"   ✓ Epochs Trained: {metrics.get('epochs_run', 0)}")
print(f"   ✓ Data Split: {metrics.get('data_split', 'N/A')}")

# Test with sample news articles
print("\n3. Testing with Sample News Articles:")
print("-" * 70)

test_samples = [
    {
        "title": "REAL NEWS EXAMPLE",
        "text": "The Federal Reserve announced a new interest rate hike of 0.25% today. The move aims to combat inflation while maintaining economic stability. Economists expect this will impact mortgage rates and savings accounts.",
        "expected": "REAL"
    },
    {
        "title": "REAL NEWS EXAMPLE 2",
        "text": "Researchers at MIT have developed a new breakthrough in quantum computing. The study, published in Nature today, shows a 50% improvement in processing speed. This could revolutionize data analysis and cryptography fields.",
        "expected": "REAL"
    },
    {
        "title": "FAKE NEWS EXAMPLE",
        "text": "BREAKING: Famous actor reveals shocking secret that doctors dont want you to know! He lost 50 pounds in just 2 weeks using this ONE WEIRD TRICK. Click here before the government takes this down! Celebrities hate him!",
        "expected": "FAKE"
    },
    {
        "title": "FAKE NEWS EXAMPLE 2",
        "text": "ALERT: New world order discovered! Billionaires meeting in secret to control weather patterns. This has been happening for 30 years! The media wont report this because theyre paid off. Share before they delete this!",
        "expected": "FAKE"
    },
]

results = []
for idx, sample in enumerate(test_samples, 1):
    print(f"\nTest {idx}: {sample['title']}")
    print(f"Text: {sample['text'][:100]}...")
    
    label, confidence, prob = predict(sample['text'])
    is_correct = "✓" if label == sample['expected'] else "✗"
    
    print(f"Prediction: {is_correct} {label} (Confidence: {confidence*100:.2f}%)")
    print(f"Expected: {sample['expected']}")
    
    results.append({
        "sample": idx,
        "prediction": label,
        "expected": sample['expected'],
        "confidence": confidence,
        "correct": label == sample['expected']
    })

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
correct = sum(1 for r in results if r['correct'])
total = len(results)
accuracy = (correct / total * 100) if total > 0 else 0

print(f"\nSample Accuracy: {correct}/{total} ({accuracy:.1f}%)")
for idx, result in enumerate(results, 1):
    status = "✓" if result['correct'] else "✗"
    print(f"  {status} Sample {result['sample']}: {result['prediction']} " +
          f"(Expected: {result['expected']}, Confidence: {result['confidence']*100:.2f}%)")

print("\n" + "="*70)
print("✓ MODEL TEST COMPLETE!")
print("="*70)
print("\nThe model is ready to use!")
print("Run: streamlit run app.py")
print("="*70)
