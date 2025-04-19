# speech_emotion.py
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def extract_mfcc_features(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def train_speech_model():
    # Example: You should train on actual speech data
    X, y = [], []  # Features and labels (your dataset here)

    # Example data (load your actual labeled data)
    audio_file = "path_to_audio.wav"
    features = extract_mfcc_features(audio_file)
    X.append(features)
    y.append("happy")  # Label (your actual labels)

    X = np.array(X)
    y = LabelEncoder().fit_transform(y)

    # Split and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "speech_emotion_model.pkl")

def predict_speech_emotion(audio_file):
    model = joblib.load("speech_emotion_model.pkl")
    features = extract_mfcc_features(audio_file).reshape(1, -1)
    emotion = model.predict(features)
    return emotion

# Example usage
if __name__ == "__main__":
    # Train model (run once)
    # train_speech_model()

    # Predict emotion
    emotion = predict_speech_emotion("path_to_audio.wav")
    print(f"Predicted speech emotion: {emotion}")
