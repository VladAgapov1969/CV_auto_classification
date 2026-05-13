# CV_auto_classification
# 🚗 YOLOv5 Vehicle Detection & Classification Pipeline

> A streamlined, production-ready Jupyter Notebook pipeline for real-time vehicle detection, tracking, and detailed telemetry logging using YOLOv5 and OpenCV.

## 📋 Overview
This project provides a complete computer vision workflow for detecting and classifying vehicles (cars, buses, trucks) in video streams. It leverages a pre-trained YOLOv5s model, processes frames efficiently, outputs an annotated video, and generates a detailed, timestamped detection log for downstream analysis or reporting.

## ✨ Features
- 🎯 **Pre-trained YOLOv5s Model**: Optimized for speed and accuracy (CPU/GPU ready)
- 🚗 **Vehicle-Specific Filtering**: Targets COCO classes: `car` (2), `bus` (5), `truck` (7)
- ⏱️ **Flexible Processing**: Process entire videos or limit to the first `N` seconds
- 📝 **Structured Telemetry Logging**: Exports timestamp, frame ID, class, confidence, and bounding box coordinates to `.txt`
- 🎬 **Annotated Output Video**: Saves a `.avi` file with real-time bounding boxes and overlay timestamps
- 🔧 **Zero-Config Ready**: Works out-of-the-box with a single notebook execution

## 🛠️ Installation
Ensure you have Python `3.8+` installed, then install dependencies:
```bash
pip install yolov5 opencv-python numpy torch torchvision
