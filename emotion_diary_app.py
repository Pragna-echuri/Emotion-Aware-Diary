import streamlit as st
from transformers import pipeline
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
st.set_page_config(page_title="Enhanced Emotion Diary", layout="wide")
# ======================
# DATA PERSISTENCE SETUP
# ======================
def init_db():
    conn = sqlite3.connect('emotion_diary.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS entries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  text TEXT,
                  text_emotion TEXT,
                  text_score REAL,
                  face_emotion TEXT,
                  face_score REAL)''')
    conn.commit()
    return conn

# Initialize database
conn = init_db()

# ======================
# MODEL LOADING
# ======================
@st.cache_resource
def load_text_model():
    return pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

@st.cache_resource
def load_face_model():
    return DeepFace

text_model = load_text_model()
face_model = load_face_model()

# ======================
# ANALYSIS FUNCTIONS
# ======================
def analyze_text(text):
    result = text_model(text)[0]
    return result["label"], result["score"]

def analyze_face(image):
    try:
        result = face_model.analyze(np.array(image), actions=["emotion"], enforce_detection=False)
        emotions = result[0]["emotion"]
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]/100
    except:
        return "neutral", 1.0

# ======================
# ENHANCED RECOMMENDATIONS
# ======================
def get_suggestions(emotion, history):
    base_suggestions = {
        "sadness": [
            "Try a 10-minute mindfulness meditation",
            "Reach out to a close friend today",
            "Write down three things you're grateful for"
        ],
        "joy": [
            "Share your positive energy with someone",
            "Document what made you happy today",
            "Try a new activity to build on this mood"
        ],
        "anger": [
            "Practice 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s)",
            "Go for a brisk 10-minute walk",
            "Write a letter you'll never send to express feelings"
        ],
        "fear": [
            "Practice grounding techniques (5-4-3-2-1 method)",
            "Break down concerns into smaller, manageable pieces",
            "Recall past challenges you've overcome"
        ],
        "surprise": [
            "Reflect on what surprised you and why",
            "Channel this energy into creative expression",
            "Share the experience with someone"
        ],
        "neutral": [
            "Try journaling about subtle feelings",
            "Engage in sensory awareness exercises",
            "Explore new stimuli to spark emotions"
        ]
    }
    
    # Time-based personalized suggestions
    if len(history) > 3:
        last_week = [e for e in history if datetime.strptime(e[1], "%Y-%m-%d %H:%M:%S") > datetime.now() - timedelta(days=7)]
        if len(last_week) > 2:
            mood_counts = pd.DataFrame(last_week).groupby(3).size()
            if mood_counts.get("sadness", 0) >= 3:
                base_suggestions["neutral"].append("Consider talking to a professional about persistent low mood")
    
    return base_suggestions.get(emotion, ["Reflect on your emotional patterns this week"])

# ======================
# VISUALIZATION FUNCTIONS
# ======================
def plot_trends(df):
    fig = px.line(df, x='date', y='text_score', color='text_emotion',
                  title="Weekly Mood Trend",
                  labels={"text_score": "Emotion Intensity", "date": "Date", "text_emotion": "Emotion"},
                  template="plotly_white")
    fig.update_layout(hovermode="x unified")
    return fig

def plot_distribution(df):
    fig = px.pie(df, names='emotion', values='count',
                 title="Emotion Distribution",
                 hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

# ======================
# MAIN APP
# ======================
def main():
    st.title("📊 Enhanced Emotion Aware AI Diary")
    
    # ===== DATA COLLECTION =====
    with st.sidebar:
        st.subheader("Capture Your Mood")
        img_file_buffer = st.camera_input("Take a selfie for emotion analysis")
        
        st.subheader("Journal Entry")
        diary_text = st.text_area("How are you feeling today?", height=150)
        
        if st.button("Analyze & Save"):
            if diary_text or img_file_buffer:
                with st.spinner("Analyzing..."):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Text analysis
                    text_emotion, text_score = analyze_text(diary_text) if diary_text else (None, None)
                    
                    # Face analysis
                    face_emotion, face_score = (None, None)
                    if img_file_buffer:
                        image = Image.open(img_file_buffer)
                        face_emotion, face_score = analyze_face(image)
                    
                    # Save to database
                    c = conn.cursor()
                    c.execute("INSERT INTO entries VALUES (NULL,?,?,?,?,?,?)",
                             (timestamp, diary_text, text_emotion, text_score, face_emotion, face_score))
                    conn.commit()
                    
                    st.success("Entry saved successfully!")
            else:
                st.warning("Please write or capture an image")

    # ===== ANALYTICS DASHBOARD =====
    st.header("Your Emotional Analytics")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        time_range = st.selectbox("Analysis Period", 
                                ["Last 7 days", "Last 30 days", "All time"])
    with col2:
        metric = st.selectbox("View By", 
                            ["Daily Trends", "Weekly Averages", "Emotion Distribution"])
    
    # Fetch data
    c = conn.cursor()
    if time_range == "Last 7 days":
        cutoff = datetime.now() - timedelta(days=7)
    elif time_range == "Last 30 days":
        cutoff = datetime.now() - timedelta(days=30)
    else:
        cutoff = datetime.min
    
    query = f"SELECT * FROM entries WHERE timestamp > '{cutoff}'"
    data = c.execute(query).fetchall()
    
    if data:
        df = pd.DataFrame(data, columns=["id","timestamp","text","text_emotion","text_score",
                                       "face_emotion","face_score"])
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Prepare visualization data
        if metric == "Emotion Distribution":
            emotion_counts = df['text_emotion'].value_counts().reset_index()
            emotion_counts.columns = ['emotion', 'count']
            st.plotly_chart(plot_distribution(emotion_counts), use_container_width=True)
        else:
            if metric == "Weekly Averages":
                df['week'] = pd.to_datetime(df['timestamp']).dt.to_period('W')
                viz_df = df.groupby(['week', 'text_emotion'])['text_score'].mean().reset_index()
                viz_df['date'] = viz_df['week'].dt.start_time
            else:
                viz_df = df.groupby(['date', 'text_emotion'])['text_score'].mean().reset_index()
            
            st.plotly_chart(plot_trends(viz_df), use_container_width=True)
            
            # Enhanced recommendations
            st.header("Personalized Suggestions")
            latest_emotion = df.iloc[-1]['text_emotion']
            history = c.execute("SELECT * FROM entries").fetchall()
            suggestions = get_suggestions(latest_emotion, history)
            
            for i, suggestion in enumerate(suggestions[:3], 1):
                st.markdown(f"{i}. {suggestion}")
    else:
        st.info("No entries found for selected period")

if __name__ == "__main__":
    main()