# EMOTIONVISION

EMOTIONVISION is a web-based emotion recognition system that classifies facial expressions in real time or from static images. Built with Flask, PyTorch, and OpenCV, it provides visual feedback through pie charts, symbolic summaries, and timeline analytics.

## Features

- Real-time emotion detection from videos
- Image-based emotion classification
- Interactive pie chart visualization of emotion distribution
- Feedback for intuitive emotional summaries

## Tech Stack

- Python
- Flask
- PyTorch
- TorchVision
- OpenCV
- HTML/CSS/JavaScript
- Chart.js (for frontend analytics)

## Dataset Structure

The dataset is organized into `train/` and `validation/` folders, each containing subfolders for the following emotion classes:
angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/

## Install Dependencies

Before running anything, install the required Python packages:

pip install -r requirements.txt

## Model Setup

Before running the web app, you need to generate the trained model file `emotionModel.pth`.  
This file contains the weights for the Convolutional Neural Network used in the emotion classification.

To generate it, run the training script:

python expressionClassifier.py

## Run the Web App

Once the model is trained and `emotionModel.pth` is generated, launch the Flask app:

python app.py

## Image Demo / Preview

<img width="1919" height="879" alt="image" src="https://github.com/user-attachments/assets/5c1ea541-a2c4-4d9f-8555-b952a851c35b" />
<img width="1919" height="860" alt="Screenshot 2025-10-21 205651" src="https://github.com/user-attachments/assets/fcde8817-1982-492b-9327-a13d56c85366" />
<img width="1917" height="870" alt="Screenshot 2025-10-21 205700" src="https://github.com/user-attachments/assets/c487012d-5cf3-4461-a4b5-2cb92b0c0d38" />
<img width="952" height="893" alt="image" src="https://github.com/user-attachments/assets/ab933a9c-1a5f-408e-96bb-628101e085b9" />
