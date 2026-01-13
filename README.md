# AI Fitness Pose Classification Pipeline

This repository contains an end-to-end pipeline for classifying exercise forms (e.g., Bicep Curls) as **"Correct"** or **"Incorrect"** using Pose Estimation data. The system supports multiple Machine Learning models (XGBoost, LightGBM, CatBoost) and Deep Learning sequences (LSTM).

## Project Overview

The goal is to analyze human body movements from video inputs (converted to CSV coordinates) and determine if the exercise technique is performed correctly. This is achieved by extracting geometric features from body landmarks and analyzing them over time.

---

## Data Pipeline & Methodology

### 1. Data Collection & Input
* **Source:** The input data consists of **CSV files** generated from Pose Estimation libraries (likely MediaPipe Pose).
* **Format:** Each row represents a video frame containing 33 keypoints (x, y, z coordinates).
* **Labeling Strategy:** Labels are extracted directly from filenames:
    * Filenames containing `_correct_` or `_cor_` -> **Class 1 (Correct)**
    * Filenames containing `_incorrect_` -> **Class 0 (Incorrect)**

### 2. Data Preprocessing & Cleaning
* **Missing Values:** Rows with missing essential coordinates are dropped.
* **Subsampling:** To reduce noise and redundancy, we may subsample frames (e.g., taking every 10th frame) or focus on the middle 60% of the video.
* **Coordinate Normalization:**
    * **Centering:** All points are shifted so the body center is at (0,0,0).
    * **Torso Scaling:** Coordinates are normalized by the torso size (distance between shoulders and hips) to make the model invariant to camera distance and subject height.

### 3. Feature Engineering
We transform raw (x, y, z) coordinates into meaningful biomechanical features:
* **Angles:** Calculation of joint angles (Elbow, Shoulder, Hip, Knee) using vector mathematics.
* **Velocities:** Change in angles between frames (Angular Velocity).
* **Distances:**
    * Pairwise distances between key joints.
    * Distance of limbs from the body center.
    * Statistical aggregates (Mean, Std, Max distances).
* **Sliding Windows (For Temporal Models):**
    * Grouping frames into time-series windows (e.g., Window Size = 5 frames).
    * Computing Mean and Std Dev for each window to capture movement stability.

### 4. Model Training
We implemented a flexible pipeline allowing selection between standard ML and Deep Learning:

#### Option A: Gradient Boosting Models (Frame-based)
Optimized using **Optuna** for hyperparameter tuning.
* **XGBoost:** High performance, supports GPU acceleration.
* **LightGBM:** Fast training speed, efficient memory usage.
* **CatBoost:** Handles categorical data well (used as a strong baseline).
* **Strategy:** Uses `StratifiedKFold` cross-validation and handles class imbalance with `class_weight`.

#### Option B: LSTM (Sequential/Time-Series)
Uses PyTorch to analyze motion patterns over time.
* **Input:** Sequence of features (Window Size = 5).
* **Architecture:** LSTM layers followed by a Fully Connected layer and Sigmoid activation.
* **Training:** Uses Binary Cross Entropy Loss and Adam Optimizer.

---
