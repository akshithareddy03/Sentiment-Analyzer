import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from transformers import pipeline
from langdetect import detect

st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: white;
    }

    h1, h4, p, label, .markdown-text-container {
        color: #ffffff !important;
    }

    h1 {
        text-align: center;
        color: #26d07c !important;
        text-shadow: 1px 1px 5px rgba(0,255,170,0.4);
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    p {
        text-align: center;
        font-size: 16px;
        margin-top: 0;
        color: #bdbdbd !important;
    }

    textarea {
        background-color: #161b22 !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 0 10px rgba(38, 208, 124, 0.3) !important;
        padding: 10px !important;
        font-size: 16px !important;
    }

    button[kind="primary"] {
        background-color: #26d07c !important;
        color: #000000 !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        padding: 0.6em 1.5em !important;
        font-size: 16px !important;
        border: none !important;
        margin-top: 10px;
    }

    div[data-testid="metric-container"] {
        background-color: #161b22;
        color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 10px rgba(0,255,170,0.2);
    }

    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    return sentiment_model, emotion_model

sentiment_model, emotion_model = load_models()

st.markdown("<h1>Multilingual Sentiment & Emotion Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p>Enter text in English, Hindi, Tamil, Bengali, and more</p>", unsafe_allow_html=True)

user_input = st.text_area("Enter your message", height=150, placeholder="Type or paste your text here...")

if st.button("Analyze Text") and user_input.strip():
    with st.spinner("Analyzing..."):
        lang = detect(user_input)
        sentiment_result = sentiment_model(user_input)
        emotion_result = emotion_model(user_input)

        sentiment_label = sentiment_result[0]['label'].capitalize()
        sentiment_score = sentiment_result[0]['score']

        emotion_label = max(emotion_result[0], key=lambda x: x['score'])['label']
        emotion_score = max(emotion_result[0], key=lambda x: x['score'])['score']

    st.markdown("---")
    st.markdown("### Analysis Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Sentiment", sentiment_label, f"{sentiment_score:.2f}")
    with col2:
        st.metric("Emotion", emotion_label, f"{emotion_score:.2f}")
    with col3:
        st.metric("Language", lang.upper())

    if sentiment_label == "Positive":
        color = "#1e4426"
    elif sentiment_label == "Negative":
        color = "#4a1e1e"
    else:
        color = "#1e2d44"

    st.markdown(
        f"<div style='padding: 20px; border-radius: 12px; background-color: {color}; text-align: center;'>"
        f"<h4>Prediction Summary</h4>"
        f"<p><strong>Sentiment:</strong> {sentiment_label} ({sentiment_score:.2f})</p>"
        f"<p><strong>Emotion:</strong> {emotion_label} ({emotion_score:.2f})</p>"
        f"<p><strong>Detected Language:</strong> {lang.upper()}</p>"
        f"</div>",
        unsafe_allow_html=True
    )
