import streamlit as st
from transformers import pipeline
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time

# Set page config
st.set_page_config(page_title="Emotion Aware AI Diary", layout="wide")

# Load models once
@st.cache_resource
def load_text_model():
    return pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

@st.cache_resource
def load_face_model():
    return DeepFace

# Initialize models
text_model = load_text_model()
face_model = load_face_model()

# Function to analyze text emotion
def analyze_text(text):
    result = text_model(text)[0]
    return {"label": result["label"], "score": result["score"]}

# Function to analyze face emotion
def analyze_face(image):
    try:
        result = face_model.analyze(np.array(image), actions=["emotion"], enforce_detection=False)
        return result[0]["emotion"]
    except:
        return {"neutral": 100}

# Mental health suggestions
def get_suggestions(emotion):
    suggestions = {
        "sadness": "Try listening to uplifting music or calling a friend.",
        "joy": "Great! Keep doing what makes you happy!",
        "anger": "Deep breathing exercises might help calm you.",
        "fear": "Practice mindfulness meditation to ease anxiety.",
        "surprise": "Embrace new experiences!",
        "neutral": "Try journaling more to explore your feelings.",
    }
    return suggestions.get(emotion, "Reflect on your day and practice gratitude.")

# Main app
def main():
    st.title("📖 Emotion Aware AI Diary")
    
    # Sidebar for webcam
    with st.sidebar:
        st.subheader("Capture Your Mood")
        img_file_buffer = st.camera_input("Take a selfie for emotion analysis")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Write Your Diary Entry")
        diary_text = st.text_area("How are you feeling today?", height=200)
        
        if st.button("Analyze Emotions"):
            if diary_text or img_file_buffer:
                with st.spinner("Analyzing your emotions..."):
                    # Analyze text
                    text_result = analyze_text(diary_text) if diary_text else None
                    
                    # Analyze face
                    face_result = None
                    if img_file_buffer:
                        image = Image.open(img_file_buffer)
                        face_result = analyze_face(image)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    if text_result:
                        st.write(f"📝 **Text Emotion**: {text_result['label']} ({(text_result['score']*100):.1f}%)")
                    
                    if face_result:
                        dominant_face_emotion = max(face_result, key=face_result.get)
                        st.write(f"📸 **Facial Emotion**: {dominant_face_emotion} ({(face_result[dominant_face_emotion]):.1f}%)")
                        
                        # Show face emotion chart
                        fig, ax = plt.subplots()
                        pd.Series(face_result).plot(kind='bar', ax=ax)
                        st.pyplot(fig)
                    
                    # Show suggestions
                    emotion = text_result["label"] if text_result else dominant_face_emotion
                    st.subheader("💡 Mental Health Suggestion")
                    st.info(get_suggestions(emotion))
                    
                    # Save to session state (replace with database in real app)
                    if "entries" not in st.session_state:
                        st.session_state.entries = []
                    
                    st.session_state.entries.append({
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "text": diary_text,
                        "text_emotion": text_result,
                        "face_emotion": face_result
                    })
            else:
                st.warning("Please write something or take a photo!")
    
    with col2:
        if "entries" in st.session_state and st.session_state.entries:
            st.subheader("Your Emotion History")
            for entry in st.session_state.entries[-5:][::-1]:  # Show last 5 entries
                with st.expander(f"{entry['time']}"):
                    st.write(f"**Entry**: {entry['text']}")
                    if entry["text_emotion"]:
                        st.write(f"Text Emotion: {entry['text_emotion']['label']} ({entry['text_emotion']['score']:.2f})")
                    if entry["face_emotion"]:
                        st.write("Face Emotions:")
                        st.json(entry["face_emotion"])

if __name__ == "__main__":
    main()