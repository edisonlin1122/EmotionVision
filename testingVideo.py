#This is just a video player, just testing how the video plays. It does not do any emotion detection.
# This file is prep for writing the realTimeEmotion.py file.

# Necessary imports
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

# Open file explorer to select a video
fileDialogRoot = tk.Tk()
fileDialogRoot.withdraw()
videoPath = filedialog.askopenfilename(
    title="Select a video file",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)
fileDialogRoot.destroy()

if not videoPath:
    exit()

# Open video
videoCapture = cv2.VideoCapture(videoPath)
if not videoCapture.isOpened():
    exit()

totalFrames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
framesPerSecond = videoCapture.get(cv2.CAP_PROP_FPS) or 30.0
videoDuration = totalFrames / framesPerSecond

isPaused = False
currentFrame = 0
isSliderDragging = False
globalImageTk = None  # Global reference for image

# Setup Tkinter window
mainWindow = tk.Tk()
mainWindow.title("Video Player")
mainWindow.configure(bg="#1e1e1e")  # Dark background

# Style configuration
style = ttk.Style()
style.theme_use("clam")
style.configure("TScale", background="#1e1e1e", troughcolor="#333333", sliderthickness=15)
style.configure("TButton", background="#333333", foreground="#ffffff", padding=6)
style.map("TButton", background=[("active", "#444444")])

# Frame label
frameLabel = tk.Label(mainWindow, bg="#1e1e1e")
frameLabel.pack()

# Time label
timeLabel = tk.Label(mainWindow, text=f"0.00 / {videoDuration:.2f} s", fg="#ffffff", bg="#1e1e1e", font=("Segoe UI", 10))
timeLabel.pack(pady=(5, 0))

# Slider
frameSlider = ttk.Scale(mainWindow, from_=0, to=totalFrames - 1, orient="horizontal", length=500)
frameSlider.pack(pady=(5, 10))

# Controls label
controlsLabel = tk.Label(mainWindow, text="Controls: Spacebar or button = Pause/Resume, Q = Quit", fg="#aaaaaa", bg="#1e1e1e", font=("Segoe UI", 9))
controlsLabel.pack()

def formatTime(frameNumber):
    return frameNumber / framesPerSecond

def showFrame():
    global currentFrame, isPaused, globalImageTk, isSliderDragging

    if not isPaused and not isSliderDragging:
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)
        success, frame = videoCapture.read()
        if not success:
            isPaused = True
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        globalImageTk = ImageTk.PhotoImage(image)
        frameLabel.configure(image=globalImageTk)

        frameSlider.set(currentFrame)
        timeLabel.config(text=f"{formatTime(currentFrame):.2f} / {videoDuration:.2f} s")

        currentFrame += 1
        if currentFrame >= totalFrames:
            currentFrame = totalFrames - 1
            isPaused = True
            updateSlider(currentFrame)

    mainWindow.after(int(1000 / framesPerSecond), showFrame)

def updateSlider(value):
    global currentFrame, globalImageTk, isSliderDragging
    isSliderDragging = True
    currentFrame = int(float(value))
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)
    success, frame = videoCapture.read()
    if success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        globalImageTk = ImageTk.PhotoImage(image)
        frameLabel.configure(image=globalImageTk)
        timeLabel.config(text=f"{formatTime(currentFrame):.2f} / {videoDuration:.2f} s")
    isSliderDragging = False

frameSlider.config(command=updateSlider)

def togglePause(event=None):
    global isPaused
    isPaused = not isPaused

def quitPlayer(event=None):
    mainWindow.destroy()
    videoCapture.release()

mainWindow.bind("<space>", togglePause)
mainWindow.bind("q", quitPlayer)

pauseButton = ttk.Button(mainWindow, text="Play/Pause", command=togglePause)
pauseButton.pack(pady=(5, 10))

showFrame()
mainWindow.mainloop()