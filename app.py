import streamlit as st
import librosa
import numpy as np
import pickle

# Load model, scaler, and label encoder
with open("Trained Model/mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Trained Model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("Trained Model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, contrast, centroid])

# Streamlit UI
st.title("🎤 Emotion Classifier from Audio")
st.write("Upload a `.wav` audio file to detect the emotion.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        features = extract_features("temp.wav").reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        emotion = le.inverse_transform(prediction)[0]
        st.success(f" Predicted Emotion: `{emotion.upper()}`")
    except Exception as e:
        st.error(f"Error: {e}")

