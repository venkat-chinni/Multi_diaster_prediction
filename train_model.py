"""
train_model.py
---------------
Trains and compares three models for disaster-type classification:

    1. CNN            -> baseline, trained from scratch
    2. ResNet50        -> transfer learning (ImageNet weights)
    3. EfficientNetB0  -> transfer learning (ImageNet weights)  [recommended / final model]

Expects a dataset organized as:

    dataset/
        train/<class_name>/*.jpg
        validation/<class_name>/*.jpg
        test/<class_name>/*.jpg

Classes (folder names): cyclone, earthquake, fire, flood, landslide, normal

Usage:
    python train_model.py --model efficientnet --epochs 25
    python train_model.py --model resnet50 --epochs 25
    python train_model.py --model cnn --epochs 25
    python train_model.py --compare        # trains all 3 and prints a comparison table

Outputs:
    models/<model_name>.h5              trained weights
    models/<model_name>_history.png     accuracy/loss curves
    models/<model_name>_confusion.png   confusion matrix
    models/comparison_report.csv        accuracy/precision/recall/F1 for all trained models

The best-performing model should be copied/renamed to models/disaster_model.h5
so model_utils.py picks it up automatically for inference.
"""

import argparse
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CLASS_NAMES = ["cyclone", "earthquake", "fire", "flood", "landslide", "normal"]


# --------------------------------------------------------------------------
# Data generators
# --------------------------------------------------------------------------
def get_generators(preprocess_fn):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASS_NAMES, shuffle=True,
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "validation"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASS_NAMES, shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASS_NAMES, shuffle=False,
    )
    return train_gen, val_gen, test_gen


# --------------------------------------------------------------------------
# Model builders
# --------------------------------------------------------------------------
def build_cnn(num_classes):
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="baseline_cnn")
    model.compile(optimizer=optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model, None


def build_transfer_model(base_arch, num_classes):
    if base_arch == "resnet50":
        base = ResNet50(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
        preprocess_fn = resnet_preprocess
    elif base_arch == "efficientnet":
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
        preprocess_fn = efficientnet_preprocess
    else:
        raise ValueError(base_arch)

    base.trainable = False  # Phase 1: feature extraction

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name=base_arch)

    model.compile(optimizer=optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model, (base, preprocess_fn)


def fine_tune(model, base, unfreeze_from=-30, lr=1e-5):
    """Phase 2: unfreeze the top layers of the backbone and fine-tune at a low LR."""
    base.trainable = True
    for layer in base.layers[:unfreeze_from]:
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------
# Training + evaluation
# --------------------------------------------------------------------------
def plot_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title(f"{name} - Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title(f"{name} - Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f"{name}_history.png"), dpi=150)
    plt.close(fig)


def evaluate(model, test_gen, name):
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"{name} - Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f"{name}_confusion.png"), dpi=150)
    plt.close()

    return {"model": name, "accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1}


def train_one(arch, epochs, fine_tune_epochs=10):
    num_classes = len(CLASS_NAMES)

    if arch == "cnn":
        model, extra = build_cnn(num_classes)
        train_gen, val_gen, test_gen = get_generators(preprocess_fn=lambda x: x)  # Rescaling layer handles it
    else:
        model, extra = build_transfer_model(arch, num_classes)
        base, preprocess_fn = extra
        train_gen, val_gen, test_gen = get_generators(preprocess_fn)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ModelCheckpoint(os.path.join(MODELS_DIR, f"{arch}_best.h5"), monitor="val_accuracy", save_best_only=True),
    ]

    print(f"\n=== Training {arch} (feature extraction phase) ===")
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

    if arch != "cnn":
        print(f"=== Fine-tuning {arch} (unfreezing top layers) ===")
        model = fine_tune(model, extra[0])
        history_ft = model.fit(train_gen, validation_data=val_gen, epochs=fine_tune_epochs, callbacks=callbacks)
        for k in history.history:
            history.history[k] += history_ft.history[k]

    plot_history(history, arch)
    model.save(os.path.join(MODELS_DIR, f"{arch}.h5"))

    metrics = evaluate(model, test_gen, arch)
    print(f"{arch} results: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "resnet50", "efficientnet"], default="efficientnet")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--compare", action="store_true", help="Train and compare all three models")
    args = parser.parse_args()

    results = []
    if args.compare:
        for arch in ["cnn", "resnet50", "efficientnet"]:
            results.append(train_one(arch, args.epochs))
    else:
        results.append(train_one(args.model, args.epochs))

    csv_path = os.path.join(MODELS_DIR, "comparison_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "precision", "recall", "f1_score"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nComparison report saved to {csv_path}")
    print("Copy/rename the best model's .h5 file to models/disaster_model.h5 to use it in the Flask app.")


if __name__ == "__main__":
    main()
