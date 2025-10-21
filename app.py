# Note: app.py uses Flask to make a web app that contains the code from the other files
# Run python app.py to boot up the local host

# Necessary imports
from flask import Flask, render_template, request
import os
import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from PIL import Image
from expressionClassifier import EmotionCNN

# sets up Flask app
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Loads my trained CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("emotionModel.pth", map_location=device))
model.eval()

# Image transform
image_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Video frame transform
video_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Image prediction
def predict_emotion(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return emotion_labels[predicted.item()]

# processes the video input and returns emotion timeline and summary
def process_video_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], "Error reading video"

    emotion_timeline = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        timestamp = frame_count / fps

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = video_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            label = emotion_labels[predicted.item()]
            emotion_timeline.append({"time": round(timestamp, 2), "emotion": label})

    cap.release()
    summary = f"Processed {frame_count} frames"
    return emotion_timeline, summary


# app route for switching between pages on the web app
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            emotion = predict_emotion(path)
            return render_template('imageTrack.html', prediction=emotion, filename=file.filename)
    return render_template('imageTrack.html', prediction=None)

@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        file = request.files['video']
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            emotion_data, summary = process_video_emotions(path)
            return render_template('videoTrack.html',
                                   prediction=summary,
                                   filename=file.filename,
                                   emotion_data=emotion_data)
    return render_template('videoTrack.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)