import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from streamlit_lottie import st_lottie

# Load animations
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

lottie_discovery = load_lottie("assets/discovery.json")
lottie_predict = load_lottie("assets/drug.json")
lottie_about = load_lottie("assets/about.json")

# Load model and encoder
model = joblib.load('model/drug_predictor.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

# Disease options
disease_options = [
    "Parkinson's disease",
    "Lung Cancer",
    "COVID-19",
    "Depression",
    "Alzheimer's disease"
]

# App configuration
st.set_page_config(page_title="AI Drug Discovery", page_icon="🧬", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 AI Drug Discovery")
    st.markdown("🚀 _Intelligent drug suggestions using AI_")
    st.markdown("---")
    app_mode = st.radio("📍 Navigate", ["🏠 Home", "📖 About", "💊 Predict Drug"])
    st.markdown("---")
    st.info("Developed by group (KHUSHI , ANKIT , ANSHUMAN , SHIVAM , YASH)❤️", icon="💻")

# Main content
if app_mode == "🏠 Home":
    st.title("Welcome to Drug Discovery App 🔬💊")
    st.subheader("AI-powered solutions for precision medicine 🚀")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        ### 🌟 Key Features
        - AI predicts top drugs for known diseases
        - Built using Scikit-learn, Streamlit & Lottie
        - Real-time predictions with probabilities
        - Fit for educational & research purposes 📚
        """)
    with col2:
        st_lottie(lottie_discovery, height=300, key="home")

elif app_mode == "📖 About":
    st.title("About This Project 📚")
    col1, col2 = st.columns(2)
    with col1:
        st_lottie(lottie_about, height=300, key="about")
    with col2:
        st.markdown("""
        ### 🧪 Purpose:
        This project aims to use machine learning to suggest potential drugs for a given disease using a trained model.

        ### 📁 Technologies Used:
        - `Streamlit` for the web interface
        - `Scikit-learn` for model training
        - `TF-IDF` and `Naive Bayes` classifier
        - `Lottie` animations for user engagement

        ### 🔍 Diseases Supported:
        - 🧠 Parkinson's disease
        - 🫁 Lung Cancer
        - 🦠 COVID-19
        - 💭 Depression
        - 🧳 Alzheimer’s disease
        """)

elif app_mode == "💊 Predict Drug":
    st.title("Predict Drugs for a Disease 💡")
    col1, col2 = st.columns([2, 1])
    with col2:
        st_lottie(lottie_predict, height=250, key="predict")
    with col1:
        selected_disease = st.selectbox("🧬 Choose a Disease:", disease_options)

        if st.button("🔍 Predict Drugs"):
            probs = model.predict_proba([selected_disease])[0]
            top_indices = np.argsort(probs)[::-1][:5]

            st.success(f"🧪 Top 5 Drug Predictions for **{selected_disease}**:")
            for idx in top_indices:
                drug = label_encoder.inverse_transform([idx])[0]
                prob = round(probs[idx] * 100, 2)
                st.markdown(f"➡️ **{drug}** — `{prob}%` 💊")

       

