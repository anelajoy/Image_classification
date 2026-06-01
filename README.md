# Retinal Image Abnormality Classifier

A binary image classification model that takes a retinal fundus image and predicts whether it is **Normal** or **Abnormal**. Built with TensorFlow/Keras as an *AI in Healthcare* inspired project.

## Overview

This project trains a convolutional neural network (CNN) to screen retinal fundus photographs for signs of disease. You give it an image; it returns a single label — *Normal* or *Abnormal* — based on a learned probability score. It is intended as an educational demonstration of an end-to-end image classification pipeline applied to a medical-imaging task, not as a clinical tool.

The model was designed around the **RFMiD (Retinal Fundus Multi-Disease Image Dataset)**, which labels each image with a `Disease_Risk` flag (`0` = normal, `1` = abnormal).

## How It Works

The pipeline has four stages, one per file:

1. **Preprocessing** (`preprocess.py`) — Reads the labels CSV, selects a balanced sample of normal and abnormal images, loads each one, resizes it to 224×224, and normalizes pixel values to the `[0, 1]` range.
2. **Model definition** (`model.py`) — Defines the CNN architecture and compiles it.
3. **Training** (`train.py`) — Splits the data into training and validation sets, trains the model with early stopping, and saves the result.
4. **Prediction & evaluation** (`predict.py`) — Loads the trained model to classify a single image or to evaluate a whole folder of labeled test images.

## Model Architecture

A sequential CNN with on-the-fly data augmentation:

- **Input:** 224×224×3 RGB image
- **Augmentation:** random horizontal/vertical flip, random zoom, random rotation (applied during training only)
- **Convolutional blocks:** three Conv2D layers (32 → 64 → 128 filters), each followed by 2×2 max pooling
- **Classifier head:** Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid)
- **Compilation:** Adam optimizer, binary cross-entropy loss, accuracy metric

The final sigmoid neuron outputs a probability; scores ≥ 0.5 are labeled *Abnormal*.

## Project Structure

```
your_project/
├── preprocess.py                  # Data loading and preprocessing
├── model.py                       # CNN architecture
├── train.py                       # Training loop
├── predict.py                     # Inference and evaluation
├── RFMiD_Training_Labels.csv      # Training labels (not included — see below)
├── RFMiD_Testing_Labels.csv       # Test labels (not included — see below)
├── Training/                      # Training images (not included — see below)
└── Testing/
    └── Test/                      # Test images (not included — see below)
```

The labels CSVs are expected to contain at least two columns: `ID` (the image identifier) and `Disease_Risk` (`0` or `1`). Image filenames are constructed as `{ID}.png`.

## Installation

Requires **Python 3.10–3.13** (TensorFlow does not yet support 3.14).

```bash
# Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install dependencies
python -m pip install --upgrade pip
pip install tensorflow numpy pandas scikit-learn
```

## Usage

Run the scripts from the project folder, with the data in place and the virtual environment active.

**Train the model:**

```bash
python train.py
```

This produces `abnormal_detection_model.h5`.

**Classify a single image:**

```python
from predict import predict_image
print(predict_image("path/to/image.png"))   # -> "Normal" or "Abnormal"
```

**Evaluate against a labeled test set:**

```bash
python predict.py
```

This prints a classification report and confusion matrix and writes `prediction_results.csv`.

## Known Limitations

- **The image data and label CSVs are not included in this repository.** The RFMiD `Training`/`Testing` image folders and their corresponding label files are required to train or evaluate the model and must be obtained separately. Without them, the scripts will run but report that images cannot be found.
- The preprocessing step balances classes by sampling a fixed number of images per class; if a class has fewer images than requested, the available count is used instead.
- This is a baseline CNN for educational purposes. It has not been validated clinically and **must not be used for medical diagnosis.**

## Acknowledgments

Created as a fun, informative side project inspired by the elective **CPSC 298: AI in Healthcare** at Chapman University. Datasets were provided by the professor of the class. This is intended as a skeleton demonstrating what such a program can do, rather than a finished application.

## Authors
**Anela Quiroz**

**Sebastian Herrera**

**Jason Kim**

## References

- [Kody Simpson]. (2025). *[Data Augmentation - Deep Learning with Tensorflow | Ep. 19]* [Video]. YouTube. https://www.youtube.com/watch?v=yke3zUGgQ-w
- [Nicholas Renotte]. (2020). *[Tensorflow Tutorial for Python in 10 Minutes]* [Video]. YouTube. https://www.youtube.com/watch?v=6_2hzRopPbQ
- [Ryan Shihabi]. (2024). *[CNN Conceptual Demo]* [Video]. YouTube. https://www.youtube.com/watch?v=aCmKHGlCMHE
