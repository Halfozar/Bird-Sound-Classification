import streamlit as st
import tempfile
import joblib
import numpy as np
import os

from utils.feature_extraction import extract_features

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Bird Sound Classifier",
    page_icon="🐦",
    layout="centered"
)

st.title("🐦 Bird Sound Classification")
st.write("Record or upload bird audio to identify bird species.")

# ============================================
# LOAD MODEL
# ============================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "bird_svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)



# ============================================
# FILE UPLOAD
# ============================================

uploaded_file = st.file_uploader(
    "Or Upload Audio File",
    type=["wav", "mp3"]
)

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_bird(audio_path):

    features = extract_features(audio_path)

    features = features.reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    return prediction

# ============================================
# RECORDING PREDICTION
# ============================================

if audio_file is not None:

    st.audio(audio_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:

        tmp.write(audio_file.read())

        temp_audio_path = tmp.name

    prediction = predict_bird(temp_audio_path)

    st.success(f"Predicted Bird: {prediction}")

# ============================================
# FILE UPLOAD PREDICTION
# ============================================

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:

        tmp.write(uploaded_file.read())

        temp_audio_path = tmp.name

    prediction = predict_bird(temp_audio_path)

    st.success(f"Predicted Bird: {prediction}")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.write("Supported Classes:")
st.write("crow, cuckoo, kingfisher, bluejay, noise")
