# facial_emotion.py
from fer import FER
import cv2

def detect_facial_emotion(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Initialize FER detector
    detector = FER()

    # Analyze emotion
    emotion, score = detector.top_emotion(image)
    return emotion, score

# Example usage
if __name__ == "__main__":
    emotion, score = detect_facial_emotion("path_to_selfie.jpg")
    print(f"Detected emotion: {emotion} with score {score}")
