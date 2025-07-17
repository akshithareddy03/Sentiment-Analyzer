# Sentiment Analyzer

This is a simple, clean, and interactive Streamlit app that analyzes the **sentiment** and **emotion** of any text in multiple languages using Hugging Face Transformers.

---

## Overview

This project uses Hugging Face Transformers to:
- Detect the language of a text
- Analyze the sentiment (Positive, Neutral, Negative)
- Identify the dominant emotion (Joy, Anger, Sadness, etc.)
- Display confidence scores for each result
- Present results in a sleek, dark-themed interface
---

## Features

- Sentiment Analysis: Positive, Negative, Neutral
- Emotion Detection: Joy, Sadness, Anger, Fear, etc.
- Language Detection: Supports English, Hindi, Tamil, Bengali, and more
- Dark-themed UI with styled input, buttons, and output
- Powered by: 
  - `cardiffnlp/twitter-roberta-base-sentiment`
  - `j-hartmann/emotion-english-distilroberta-base`

---
## Deployment Instructions

 1.To deploy this on a platform like Streamlit Cloud:
 2.Push this project to a public GitHub repository.
 3.Go to the platform and select your repository.
 4.Set app.py as the entry point.
 5.Click Deploy.
 
 ---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/sentiment-emotion-analyzer.git
cd sentiment-emotion-analyzer
