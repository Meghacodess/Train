import librosa
import numpy as np
import pickle
import sys
import os

#  Loading saved components 
with open("models/mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

#  Feature extraction step
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, contrast, centroid])

#  Main Execution 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(" Usage: python test_model.py path_to_audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        print(f" File not found: {audio_path}")
        sys.exit(1)

    try:
        features = extract_features(audio_path).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        emotion = le.inverse_transform(prediction)[0]
        print(f"🎯 Predicted Emotion: {emotion}")
    except Exception as e:
        print(f" Error during prediction: {e}")
