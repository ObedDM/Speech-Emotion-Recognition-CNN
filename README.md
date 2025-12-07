# 🎙️ Speech Emotion Recognition (SER) with CNN

A Deep Learning model that classifies human emotions from audio using **Mel-Spectrograms** and a **CNN**, achieving **93.62% validation accuracy** on the **MESD** dataset.

## 🔎 Overview
The system processes `.wav` files to classify 6 emotions: **Anger, Disgust, Fear, Happiness, Neutral, and Sadness**. It uses the **MESD** (Mexican Emotional Speech Database) containing recordings from men, women, and children.

## 🧠 Methodology

### 1. Preprocessing Pipeline
We standardize inputs using `librosa` to ensure consistent model performance:
* **Resampling:** 16 kHz.
* **Normalization:** Fixed duration of **2 seconds** (32k samples).
* **Augmentation:** Gaussian Noise injection to prevent overfitting.
* **Feature Extraction:** 128-band Mel-Spectrograms saved as `.npy`.

```python
# Feature Extraction Snippet
S = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
S_db = librosa.power_to_db(S, ref=np.max)
```

### 2. CNN Architecture

* **Input:** `(128, 126, 1)` Spectrogram.
* **Structure:** ![Conv2D](https://img.shields.io/badge/-3x_(Conv2D_&_MaxPooling)-blue) → ![Flatten](https://img.shields.io/badge/-Flatten-gray) → ![Dense](https://img.shields.io/badge/-Dense_128-orange) → ![Softmax](https://img.shields.io/badge/-Softmax-green)

## 📊 Results (18 Epochs w/Early Stopping)

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Val Accuracy** | **93.62%** | High generalization due to Early Stopping. |
| **Test Accuracy** | **~98.4%** | Near-perfect classification on test set. |
| **F1-Score** | **0.96** | Strong precision across all 6 classes. |

## 🚀 Quick Start

```bash
# 1. Clone & Install
git clone https://github.com/ObedDM/speech-emotion-recognition-cnn.git
pip install tensorflow librosa numpy matplotlib scikit-learn

# 2. Preprocess Data (Convert .wav to .npy)
python preprocessing.py

# 3. Train/Run Model via Notebook
jupyter notebook CNN_SER.ipynb
```
