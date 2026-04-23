# Deployment Guide

## Local Deployment ✅ (Working)

```bash
pip install -r requirements.txt
streamlit run app.py
```
Open: http://localhost:8501

---

## Hugging Face Spaces (Recommended) 🤗

### Setup:
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose **Streamlit** runtime
4. Select **Public**
5. Click "Create"

### Deploy:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
git remote add origin https://github.com/gsv3245-cpu/Fake_News_Detection.git
git pull origin main
git push
```

**Pros:**
- ✅ Full Python 3.11 support
- ✅ TensorFlow works perfectly
- ✅ Easy integration with GitHub
- ✅ Free tier available

---

## Docker Deployment 🐳

### Local Docker:
```bash
docker build -t fake-news-detector .
docker run -p 8501:8501 fake-news-detector
```

### Deploy to Railway.app:
1. Go to https://railway.app
2. Connect GitHub repo
3. Add Dockerfile
4. Deploy

### Deploy to Render.com:
1. Go to https://render.com
2. Create New Service → Web Service
3. Connect GitHub
4. Select **Docker** runtime
5. Deploy

**Pros:**
- ✅ Python 3.11 full support
- ✅ TensorFlow works
- ✅ Better performance
- ✅ More configuration options

---

## Environment Limitations

❌ **Streamlit Cloud:** Python 3.14.4 (no TensorFlow support)
✅ **Hugging Face Spaces:** Python 3.11 (TensorFlow supported)
✅ **Docker (Railway/Render):** Python 3.11 (TensorFlow supported)
✅ **Local Machine:** Any Python 3.9-3.12 (TensorFlow supported)

---

## Quick Fix for Streamlit Cloud

If you want to keep using Streamlit Cloud, you can:
1. Create a REST API using Flask/FastAPI on Railway
2. Call it from the Streamlit app via HTTP

But **Hugging Face Spaces is the easiest** - just push your code there!
