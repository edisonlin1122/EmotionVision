# File that essentially combines the testingVideo.py file's video player with the CNN model

import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageTk
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, ttk

# CNN model definition
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Emotion prediction function
def predict_emotion(frame, model, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    frame = transform(frame).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(frame)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# File explorer to select video
def get_video_path():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    root.destroy()
    return video_path if video_path else None

# Main GUI-based video player
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load('emotionModel.pth', map_location=device))

    video_path = get_video_path()
    if not video_path:
        return

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        return

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS) or 30.0
    video_duration = total_frames / frames_per_second

    is_paused = False
    current_frame = 0
    is_slider_dragging = False
    global_image_tk = None

    emotions = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    # Setup Tkinter window
    main_window = tk.Tk()
    main_window.title("Emotion Recognition Video Player")
    main_window.configure(bg="#1e1e1e")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TScale", background="#1e1e1e", troughcolor="#333333", sliderthickness=15)
    style.configure("TButton", background="#333333", foreground="#ffffff", padding=6)
    style.map("TButton", background=[("active", "#444444")])

    frame_label = tk.Label(main_window, bg="#1e1e1e")
    frame_label.pack()

    time_label = tk.Label(main_window, text=f"0.00 / {video_duration:.2f} s", fg="#ffffff", bg="#1e1e1e", font=("Segoe UI", 10))
    time_label.pack(pady=(5, 0))

    frame_slider = ttk.Scale(main_window, from_=0, to=total_frames - 1, orient="horizontal", length=500)
    frame_slider.pack(pady=(5, 10))

    controls_label = tk.Label(main_window, text="Controls: Spacebar or button = Pause/Resume, Q = Quit", fg="#aaaaaa", bg="#1e1e1e", font=("Segoe UI", 9))
    controls_label.pack()

    def format_time(frame_number):
        return frame_number / frames_per_second

    # function for showing the frames from the video input

    def show_frame():
        nonlocal current_frame, is_paused, global_image_tk, is_slider_dragging

        if not is_paused and not is_slider_dragging:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            success, frame = video_capture.read()
            if not success:
                is_paused = True
                return

            emotion = predict_emotion(frame, model, device)
            emotion_label = emotions[emotion]
            cv2.putText(frame, f"Emotion: {emotion_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            global_image_tk = ImageTk.PhotoImage(image)
            frame_label.configure(image=global_image_tk)

            frame_slider.set(current_frame)
            time_label.config(text=f"{format_time(current_frame):.2f} / {video_duration:.2f} s")

            current_frame += 1
            if current_frame >= total_frames:
                current_frame = total_frames - 1
                is_paused = True
                update_slider(current_frame)

        main_window.after(int(1000 / frames_per_second), show_frame)


    # Function for updating the slider position when user drags it
    def update_slider(value):
        nonlocal current_frame, global_image_tk, is_slider_dragging
        is_slider_dragging = True
        current_frame = int(float(value))
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = video_capture.read()
        if success:
            emotion = predict_emotion(frame, model, device)
            emotion_label = emotions[emotion]
            cv2.putText(frame, f"Emotion: {emotion_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            global_image_tk = ImageTk.PhotoImage(image)
            frame_label.configure(image=global_image_tk)
            time_label.config(text=f"{format_time(current_frame):.2f} / {video_duration:.2f} s")
        is_slider_dragging = False

    frame_slider.config(command=update_slider)

    def toggle_pause(event=None):
        nonlocal is_paused
        is_paused = not is_paused

    def quit_player(event=None):
        main_window.destroy()
        video_capture.release()

    main_window.bind("<space>", toggle_pause)
    main_window.bind("q", quit_player)

    pause_button = ttk.Button(main_window, text="Play/Pause", command=toggle_pause)
    pause_button.pack(pady=(5, 10))

    show_frame()
    main_window.mainloop()

if __name__ == "__main__":
    main()