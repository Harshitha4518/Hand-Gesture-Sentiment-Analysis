# Hand Gesture Sentiment Analysis

An end-to-end computer vision and deep learning framework utilizing MediaPipe hand tracking and a custom pipeline to interpret real-time physical hand gestures into emotional and sentiment indicators.

## 📌 Overview
This project bridges computer vision and sentiment analysis by tracking physical hand landmarks and mapping structural gesture movements to psychological sentiment states (e.g., positive validation, neutral stance, or negative/aggressive expressions). It serves as an intuitive interface for touchless human-computer interaction (HCI).

## 🛠️ Repository Structure
The repository contains the following core files:
*   `gesture_analyzer.py` — Main execution pipeline managing real-time video capture, landmark extraction, and model inference.
*   `gesture_model.pkl` — Pre-trained serialized pipeline containing optimized weights for gesture classification.
*   `gesture_dataset.csv` — Comprehensive metadata catalog containing image references, bounding box dimensions, and truth labels.
*   `sample_landmarks.csv` — Extracted normalized spatial $(X, Y, Z)$ coordinates used during training.
*   `dataset_summary.txt` — Log file summarizing dataset distribution, lighting condition adjustments, and preprocessing stats.

## 📊 Performance & Visualizations
*   `eda_distribution_plots.png` — Visualization showcasing class balances across the tracked gesture classes.
*   `confusion_matrix.png` — Graphic detailing classification accuracy and misclassification margins across sentiments.
*   `metrics_card.png` — High-level dashboard presenting global evaluation metrics (Precision, Recall, F1-Score).
*   `sentiment_trend_comparison.png` — Sequence plot showing sentiment predictions over continuous time-series frames.

## 🚀 Getting Started
1. Clone the repository and install required dependencies (`opencv-python`, `mediapipe`, `scikit-learn`, `pandas`).
2. Execute the prediction engine:
```bash
   python gesture_analyzer.py
