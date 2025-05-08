# YOLOv8-s Animal Pose Estimation - Horse Keypoint Detection

This repository contains an academic project focused on fine-tuning the YOLOv8s-pose model for horse pose estimation. The project demonstrates how to adapt a state-of-the-art pose detection model to work specifically with horse images, detecting 22 keypoints on horse anatomy.

## Project Overview

The goal of this project is to accurately detect and track keypoints on horses using the YOLOv8s model architecture. The model is trained on the Horse10 dataset to recognize specific anatomical features and joint positions of horses.

## Installation


To run this project, you'll need to install the Ultralytics package:

```bash
pip install ultralytics
```
## Dataset

The project uses the Horse10 dataset, which contains horse images with annotated keypoints. The dataset is organized into:

Training set: Images with 22 annotated keypoints per horse
Validation set: Separate images for evaluation
The data preprocessing pipeline transforms the original JSON annotations into YOLO-compatible format:

Extraction of image IDs and bounding boxes
Conversion of keypoint coordinates to normalized YOLO format
Generation of text files with bounding box and keypoint data
## Ressources 
You can find the dataset here : [Horses10](https://drive.google.com/drive/folders/1YfaYiLtecKTy9ckC8S8erU24_nlC4Xs5?usp=sharing)
## Training
The YOLOv8s-pose model is fine-tuned with the following configuration:

DATASET_YAML = "horse-keypoints.yaml"
MODEL = "yolov8s-pose.pt"
EPOCHS = 50
KPT_SHAPE = (22, 3)  # 22 keypoints, each with x, y, visibility
BATCH_SIZE = 32

The training process includes data augmentation techniques like mosaic (0.4) and handles learning rate scheduling automatically.

Evaluation
The trained model is evaluated on the validation set, producing metrics including:

Precision-Recall curves for both bounding box detection and pose estimation
F1 score curves
Visual comparison of ground truth and predictions
Results
The model demonstrates successful detection of horse keypoints across various poses, lighting conditions, and scenarios. Visual results and evaluation metrics are available in the notebook.

Usage
To train the model:

Prepare your dataset in the required format
Configure the training parameters in the notebook
Run the training cells
Evaluate the model on validation data
