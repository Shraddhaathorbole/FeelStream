from flask import Flask, jsonify, render_template
from emotion_recognition import EmotionRecognizer
import threading

app = Flask(__name__)

model_path = 'main_emotion_detection_model.h5'
csv_path = 'data_moods.csv'

emotion_recognizer = EmotionRecognizer(model_path, csv_path)

# Shared state to store the latest emotion and songs
latest_emotion = None
latest_songs = []

def update_emotion():
    global latest_emotion, latest_songs
    while True:
        # Run emotion recognition headless (no GUI)
        emotion, songs = emotion_recognizer.recognize_emotion(detection_duration=10, show_gui=False)
        latest_emotion = emotion
        latest_songs = songs
        print(f"[THREAD] Updated emotion: {latest_emotion}")
        print(f"[THREAD] Updated songs: {latest_songs}")

# Start background thread to continuously update emotion
threading.Thread(target=update_emotion, daemon=True).start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_emotion')
def get_emotion():
    return jsonify({
        'emotion': latest_emotion,
        'songs': latest_songs
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
