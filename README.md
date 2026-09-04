# AI-Based Multi-Disaster Detection and Damage Severity Assessment Using Deep Learning

A final-year B.Tech major project: a web application that classifies natural
disasters (flood, fire, landslide, cyclone, earthquake, or normal) from an
uploaded photograph, estimates damage severity, and provides preliminary
recommendations to support rapid disaster-response decision-making.

> **Important:** This system is a decision-support / preliminary-assessment
> tool. It does **not** replace official disaster-management or emergency
> authorities.

## 1. Project Structure

```
disaster_project/
├── app.py                     # Flask web application (routes, upload, result, history)
├── model_utils.py             # Preprocessing + inference (Stage 2 & 3)
├── severity_assessment.py     # Severity/risk scoring + recommendations (Stage 4 & 6)
├── train_model.py             # Trains & compares CNN / ResNet50 / EfficientNetB0
├── database.py                # SQLite models (SQLAlchemy) for prediction history
├── requirements.txt
├── templates/                 # Jinja2 HTML (Bootstrap 5)
│   ├── base.html
│   ├── index.html             # Upload page
│   ├── result.html            # Result dashboard
│   ├── history.html
│   └── 404.html
├── static/
│   ├── css/style.css
│   └── js/script.js
├── models/                    # Trained .h5 models + evaluation plots (generated)
├── dataset/                   # train/validation/test image folders (you provide)
└── uploads/                   # User-uploaded images (generated at runtime)
```

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Prepare the Dataset

Collect a **curated** image dataset (do not scrape random Google Images —
dataset quality directly determines model accuracy). Good starting points:
Kaggle "Disaster Images Dataset", "Cyclone Wildfire Flood Earthquake Dataset",
or combine multiple public disaster-image datasets.

Organize it as:

```
dataset/
├── train/{cyclone,earthquake,fire,flood,landslide,normal}/*.jpg
├── validation/{cyclone,earthquake,fire,flood,landslide,normal}/*.jpg
└── test/{cyclone,earthquake,fire,flood,landslide,normal}/*.jpg
```

Aim for at least 300–500 images per class in `train/`, with an
80/10/10 or 70/15/15 train/validation/test split, and roughly balanced
classes for the most reliable accuracy.

## 4. Train the Model

Train and compare all three models (baseline CNN, ResNet50, EfficientNetB0):

```bash
python train_model.py --compare --epochs 25
```

Or train a single architecture:

```bash
python train_model.py --model efficientnet --epochs 25
```

This produces, per model, inside `models/`:
- `<model>.h5` — trained weights
- `<model>_history.png` — accuracy/loss curves
- `<model>_confusion.png` — confusion matrix
- `comparison_report.csv` — accuracy / precision / recall / F1 across all trained models

**Pick the best-performing model** (EfficientNetB0 is recommended as the
starting point) and copy it to the filename the app expects:

```bash
cp models/efficientnet.h5 models/disaster_model.h5
```

### Tips for higher accuracy
- Use data augmentation (already configured: rotation, shift, zoom, flip, brightness).
- Use transfer learning + fine-tuning (already implemented: freeze → train → unfreeze top layers → fine-tune at low LR).
- Balance your classes; if one class has far fewer images, apply class weights or oversample.
- Increase dataset size and image diversity (different angles, lighting, resolutions).
- Track validation accuracy/loss with EarlyStopping (already configured) to avoid overfitting.
- For a stronger final report, run `--compare` and present the confusion matrices and metrics table for all three models.

## 5. Run the Web Application

```bash
python app.py
```

Open **http://127.0.0.1:5000** in a browser, upload a disaster image, and
view the Disaster Type, Confidence, Severity, Risk Level, and Recommendation.

## 6. How Severity Is Computed

`severity_assessment.py` combines:
1. The model's classification confidence.
2. A disaster-specific OpenCV image heuristic (e.g. % of frame that is
   turbid water for floods, flame-colored pixels for fire, edge density
   for landslide/earthquake debris).

These are weighted into a single score mapped to **Low / Moderate / High /
Critical** severity and risk bands. This rule-based module is intentionally
kept explainable; see "Future Scope" below for the upgrade path (a trained
segmentation model).

## 7. Future Scope

- Real-time CCTV/drone-feed detection
- Satellite image analysis for large-scale damage mapping
- Object detection (YOLO) for damaged buildings, vehicles, roads
- Semantic segmentation (U-Net / DeepLab) for precise affected-area estimation
- Multimodal fusion (image + weather + location + historical data)
- GIS integration for disaster-risk maps
- Early-warning system using live weather data
- Native mobile application
- Structured report sharing with authorized disaster-management organizations

## 8. Technology Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript, Bootstrap 5, Chart.js |
| Backend | Python, Flask, Flask-SQLAlchemy |
| ML/DL | TensorFlow/Keras, OpenCV, NumPy, Pandas, Scikit-learn |
| Model | CNN (baseline), ResNet50, EfficientNetB0 (transfer learning) |
| Database | SQLite |
| Visualization | Matplotlib, Seaborn, Chart.js |
