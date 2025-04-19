# text_emotion.py
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline

# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Initialize pipeline for sentiment analysis
emotion_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def detect_emotion(text):
    result = emotion_pipeline(text)
    return result

# Example usage
if __name__ == "__main__":
    text = "I feel really happy today!"
    emotion = detect_emotion(text)
    print(emotion)
