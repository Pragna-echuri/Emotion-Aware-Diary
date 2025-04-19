# dashboard.py
from flask import Flask, render_template, request
from text_emotion import detect_emotion
from facial_emotion import detect_facial_emotion
from speech_emotion import predict_speech_emotion

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        image = request.files.get("image")
        audio = request.files.get("audio")

        # Detect emotion from text
        text_emotion = detect_emotion(text)

        # Detect emotion from facial expression
        if image:
            image_path = f"static/{image.filename}"
            image.save(image_path)
            facial_emotion = detect_facial_emotion(image_path)
        
        # Detect emotion from audio
        if audio:
            audio_path = f"static/{audio.filename}"
            audio.save(audio_path)
            speech_emotion = predict_speech_emotion(audio_path)

        return render_template("index.html", text_emotion=text_emotion, facial_emotion=facial_emotion, speech_emotion=speech_emotion)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
