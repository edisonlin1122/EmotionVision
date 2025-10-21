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

## Model Setup

Before running the web app, you need to generate the trained model file `emotionModel.pth`.  
This file contains the weights for the Convolutional Neural Network used in the emotion classification.

To generate it, run the training script:

python expressionClassifier.py

## Run the Web App

Once the model is trained and `emotionModel.pth` is generated, launch the Flask app:

python app.py
