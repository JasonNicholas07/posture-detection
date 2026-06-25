# 🧘‍♂️ Posture AI — Real-Time Posture Tracker

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00A67E.svg?style=flat-square&logo=google&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-F37626.svg?style=flat-square)
![Optuna](https://img.shields.io/badge/Optuna-2565C1.svg?style=flat-square)

An intelligent posture tracking application powered by **Computer Vision** and **Machine Learning**. 

> The app analyzes webcam footage in real-time to detect your posture while sitting at a computer, classifying it as upright (**Normal**), leaning forward (**Forward**), or leaning back (**Back**).

This project was built using **Google MediaPipe** for body landmark extraction (pose estimation) and **XGBoost** as the machine learning engine, optimized via **Bayesian Optimization (Optuna)**.

---

## 🌟 Key Features

*   **👁️ Real-Time Inference:** Processes webcam feeds without lag using the highly efficient MediaPipe Tasks API architecture.
*   **🧠 Advanced Feature Engineering:** The model goes beyond raw coordinate data by utilizing specific anatomical feature extraction, such as *Nose-to-Shoulder Distance* (detects "turtle neck"), *Ear-Shoulder Asymmetry* (detects head tilting), and *Shoulder Width* (detects slouching).
*   **🎯 Smart Thresholding:** Uses custom confidence thresholds (**Normal: 70%**, **Back: 80%**) to prevent model flickering and ensure high AI certainty before changing posture status.
*   **⏱️ Temporal Smoothing & Timer:** Prevents rapid status switching and triggers a visual alert if poor posture persists for more than 10 seconds.
*   **🔄 Active Learning Feedback:** Users can press the `f` key while the app is running to provide direct feedback to the AI. This data is saved to `dataset_feedback.csv` to retrain the model, allowing it to dynamically adapt to the user's unique habits.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.x |
| **Computer Vision** | OpenCV, MediaPipe (Google APIs) |
| **Machine Learning** | XGBoost, Scikit-Learn |
| **Hyperparameter Tuning** | Optuna (Bayesian Optimization) |
| **Data Handling & EDA** | Pandas, NumPy, Seaborn, Matplotlib |

---

## 📁 Project Structure

```text
posture-pal-ai/
├── system_interface.py             # Main script to run the live webcam tracker
├── train_final.py                  # XGBoost model training and Optuna tuning script
├── utils.py                        # Helper module (TemporalSmoother, build_features)
├── eda_advanced.py                 # EDA script (Heatmaps, Scatters, Feature Importance)
├── preprocess_outliers.py          # Automated data cleaning via class-based IQR method
└── posture_xgboost_production.pkl  # Trained model and Label Encoder (Pickle file)
