# Acoustic Machine Fault Detection and Monitoring System System

An **AI-Powered Unsupervised Anomaly Detection Platform** designed to detect malfunctions and anomalous behaviors in machinery sounds.

Unlike traditional supervised classification models, this system is trained **exclusively on healthy, normal machine sounds**. By learning to reconstruct the spectrogram of a normal sound, the AI learns what "normal" looks like. When it hears an unfamiliar or broken sound (an anomaly), it fails to reconstruct the sound perfectly. We measure this failure using Mean Squared Error (MSE). If the error crosses a mathematical threshold, we instantly flag the machine as anomalous.

---

## 🌟 Key Features

*   **Unsupervised Learning Approach:** Detects unprecedented machine failures without needing prior recordings of those specific failures.
*   **Convolutional Autoencoder (CAE):** Uses state-of-the-art computer vision techniques to analyze Mel-Spectrograms of audio signals.
*   **Industrial-Grade Robustness:** Implements **SpecAugment** (Time/Frequency Masking) and Gaussian noise injection during training to ensure the model isn't confused by background factory noises.
*   **Hyperparameter Proof:** Includes an Optuna tuning script (`scripts/tune_hyperparams.py`) to mathematically prove the optimal latent bottleneck dimension for the autoencoder.
*   **Interactive Web Dashboard:** A beautiful, responsive glassmorphism dashboard built with Flask to test the model live via microphone or evaluate it against thousands of audio files.

---

## 🚀 Step-by-Step Setup Guide

Follow these instructions exactly to get the project running on your local machine.

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You will also need a microphone connected to your computer if you want to use the live dashboard testing feature.

### 2. Install Dependencies
Open your terminal/command prompt, navigate to the project directory, and run:
```bash
python -m pip install -r requirements.txt
```
*(Note: The audio preprocessing module uses `soundfile` to ensure cross-platform compatibility without needing complex FFmpeg C++ installations).*

### 3. Download the Dataset
This project uses the [ESC-50 Dataset](https://github.com/karoldvl/ESC-50).
1. Download the dataset ZIP file from GitHub.
2. Extract the `audio/` folder and `meta/` folder directly into the root directory of this project.

### 4. Generate Spectrograms (Data Preprocessing)
AI neural networks are better at analyzing images than raw audio. Run this script to convert all `.wav` audio files into visual heat-maps called Mel-Spectrograms:
```bash
python scripts/generate_spectrograms.py
```
This will create a `spectrograms/` folder filled with images. This process may take a minute or two.

---

## 💻 How to Use the System

There are two main ways to interact with this project:

### 1. The Live Web Dashboard (Recommended)
Start the dashboard by running:
```bash
python scripts/dashboard.py
```
Then, open your web browser and go to `http://127.0.0.1:5000`. 
*   **Live Mic Test:** Click the microphone icon to record a 5-second clip of your surroundings. The AI will instantly process it, generate a spectrogram, compute the reconstruction error, and tell you if it's "Normal" or an "Anomaly".
*   **Performance Evaluation:** Go to the "Model Performance Evaluation" tab and click "Evaluate Model". The system will process thousands of images and generate an ROC curve and a Confusion Matrix to prove the model's accuracy.

### 2. Training the Model from Scratch
If you want to train the model yourself (or adjust the data augmentations), run:
```bash
python scripts/train.py
```
This will train the Convolutional Autoencoder on the normal classes defined in `src/config.py` (e.g., engine, chainsaw) and save the best weights to `checkpoints/best_autoencoder.pth`.

### 3. Hyperparameter Tuning
To run the Optuna ablation study and find the perfect latent dimension size:
```bash
python scripts/tune_hyperparams.py
```

---

## 🧠 System Architecture

Here is exactly how the codebase is structured and what each part does:

```text
Acoustic_Machine_Fault_Detection_and_Monitoring_System/
├── audio/                   # Raw .wav files (from ESC-50)
├── checkpoints/             # Saved model weights (.pth files)
├── dashboard/               # The Web UI
│   ├── static/              # CSS styles (Light glassmorphism theme)
│   ├── templates/           # HTML templates
│   └── app.py               # Flask web server
├── meta/                    # CSV files containing dataset labels
├── scripts/                 
│   ├── dashboard.py         # Entry point to run the web server
│   ├── generate_spectrograms.py # Converts audio to images
│   ├── train.py             # Trains the autoencoder
│   └── tune_hyperparams.py  # Runs Optuna hyperparameter searches
├── spectrograms/            # Generated images used for training
└── src/                     # Core Logic
    ├── config.py            # Global settings (Sample rates, classes, paths)
    ├── data/
    │   ├── loader.py        # PyTorch Dataset (includes SpecAugment & Noise)
    │   └── preprocessing.py # Converts .wav to Tensor spectrograms
    ├── inference/
    │   └── predict.py       # Handles calculating MSE for new audio
    ├── models/
    │   └── autoencoder.py   # The PyTorch Neural Network architecture
    └── training/
        └── trainer.py       # The training loop
```

### The Autoencoder Pipeline Explained:
1. **Input:** A 1x128x313 Mel-Spectrogram image of a machine sound.
2. **Encoder:** Three Convolutional Layers compress the image down to a `64x8x39` mathematical bottleneck, forcing the model to learn only the most crucial frequencies of a "healthy" machine.
3. **Decoder:** Three Transposed Convolutional Layers attempt to redraw the original image from the compressed bottleneck.
4. **Loss:** If the sound is normal, the redrawn image matches the original (Low MSE). If the sound is an anomaly, the redrawn image is a blurry mess (High MSE).
