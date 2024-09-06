# AI Companion for Vision Impaired

## Overview

The **AI Companion for Vision Impaired** project integrates object detection and text-to-speech (TTS) capabilities to assist visually impaired individuals in recognizing objects in real-time. The system uses a camera module to capture live video, performs object detection using a deep learning model, and provides auditory feedback via a text-to-speech engine for each detected object. This project aims to provide visually impaired users with a seamless experience, helping them identify and interact with their surroundings.

## Features

- **Real-Time Object Detection**: Uses the TensorFlow SSD MobileNet model to detect objects from the COCO dataset.
- **Text-to-Speech Feedback**: Converts detected objects into speech using the `flite` text-to-speech engine.
- **Camera Stream**: Supports both live camera feed and video file input.
- **Customizable Object Detection**: Detects a wide range of objects (90 classes) and supports adjusting the confidence threshold.

## Technologies Used

- **OpenCV**: For handling the video stream and drawing bounding boxes around detected objects.
- **TensorFlow**: For deep learning-based object detection (SSD MobileNet model).
- **Flite**: For text-to-speech synthesis to provide real-time feedback to the user.
- **Python**: The primary programming language used for object detection and TTS functionality.

## Project Structure

├── ObjectDetection/
│   ├── obj.txt                # Stores the detected object label
│   ├── object_detection.py     # Main script for object detection
│   ├── texttospeech.py         # Script for text-to-speech conversion
│   ├── frozen_inference_graph.pb # Pre-trained model weights
│   ├── ssd_mobilenet_v1_coco.pbtxt # Model configuration file
└── README.md                   

