"""
model_utils.py
---------------
Stage 2 (Preprocessing) and Stage 3 (Disaster Classification) helpers.

Loads the trained Keras transfer-learning model (EfficientNetB0 by default)
and exposes a single `predict_disaster()` function used by the Flask app.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (224, 224)

CLASS_NAMES = ["Cyclone", "Earthquake", "Fire", "Flood", "Landslide", "Normal"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "disaster_model.h5")

_model = None


def get_model():
    """Lazy-load the model once per process."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                f"Run `python train_model.py` first (see README.md)."
            )
        _model = load_model(MODEL_PATH)
    return _model


def read_image(file_path):
    """Reads an image from disk as BGR (OpenCV default)."""
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not read image at {file_path}. Unsupported or corrupt file.")
    return img


def preprocess_image(img_bgr):
    """
    Stage 2 — Image Preprocessing:
      - Resize to model input size
      - Denoise
      - Convert BGR -> RGB
      - Normalize via EfficientNet's preprocess_input
    Returns a (1, 224, 224, 3) batch ready for inference.
    """
    img = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)


def predict_disaster(file_path):
    """
    Runs the full Stage 2 + Stage 3 pipeline on an uploaded image.

    Returns:
        disaster_type (str), confidence_pct (float), all_probs (dict[str, float]), img_bgr (np.ndarray)
    """
    model = get_model()
    img_bgr = read_image(file_path)
    batch = preprocess_image(img_bgr)

    preds = model.predict(batch, verbose=0)[0]  # softmax probabilities
    all_probs = {CLASS_NAMES[i]: float(preds[i] * 100) for i in range(len(CLASS_NAMES))}

    top_idx = int(np.argmax(preds))
    disaster_type = CLASS_NAMES[top_idx]
    confidence_pct = float(preds[top_idx] * 100)

    return disaster_type, confidence_pct, all_probs, img_bgr
