# Posture AI - Real-Time Posture Tracker

An intelligent posture tracking application powered by Computer Vision and Machine Learning. The app analyzes webcam footage in real-time to detect the user's posture while sitting at a computer, classifying it as upright (Normal), leaning forward (Forward), or leaning back (Back).

This project was built using **Google MediaPipe** for body landmark extraction (pose estimation) and **XGBoost** as the machine learning engine, optimized via **Bayesian Optimization (Optuna)**.

---

## Key Features

- **👁️ Real-Time Inference:** Processes webcam feeds without lag using the MediaPipe Tasks API architecture.
- **🧠 Advanced Feature Engineering:** The model goes beyond raw data by utilizing specific anatomical feature extraction, such as:
  - *Nose-to-Shoulder Distance* (Detects forward head posture/"turtle neck").
  - *Ear-Shoulder Asymmetry* (Detects head tilting).
  - *Shoulder Width* (Detects slouching or hunched posture).
- **🎯 Smart Thresholding:** Uses custom thresholds (Normal: 70%, Back: 80%) to prevent flickering and ensure high AI confidence before changing posture status.
- **⏱️ Temporal Smoothing & Timer:** Prevents rapid status switching (flickering) and triggers a visual alert if poor posture persists for more than 10 seconds.
- **🔄 Active Learning Feedback:** Users can press the `f` key while the app is running to provide direct feedback to the AI. This data is saved to `dataset_feedback.csv` to retrain the model, allowing it to better adapt to the user's unique habits.

---

## Tech Stack

- **Python 3.x**
- **Computer Vision:** OpenCV, MediaPipe (Google APIs)
- **Machine Learning:** XGBoost, Scikit-Learn
- **Hyperparameter Tuning:** Optuna (Bayesian Optimization)
- **Data Analysis & Preprocessing:** Pandas, NumPy, Seaborn, Matplotlib

---

## Project Structure

- `system_interface.py`: Main script to run the live webcam tracking application.
- `train_final.py`: Script for training the XGBoost model using the cleaned dataset and Optuna.
- `utils.py`: Helper module containing custom functions (such as `TemporalSmoother` and `build_features`).
- `eda_advanced.py`: Script for Exploratory Data Analysis (Heatmap, 2D Scatter, Feature Importance).
- `preprocess_outliers.py`: Script for automated data cleaning using a class-based IQR (Interquartile Range) method.
- `posture_xgboost_production.pkl`: Pickle file containing the trained model and Label Encoder. (Ensure this matches the file size on GitHub).

---

## Installation and Usage

1. **Clone this repository**
   ```bash
   git clone [https://github.com/UsernameKamu/posture-pal-ai.git](https://github.com/UsernameKamu/posture-pal-ai.git)
   cd posture-pal-ai
