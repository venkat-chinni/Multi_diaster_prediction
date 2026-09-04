"""
app.py
------
Flask web application tying together all 6 pipeline stages:

    1. Image Upload            -> routes: '/'  (GET/POST)
    2. Image Preprocessing     -> model_utils.preprocess_image()
    3. Disaster Classification -> model_utils.predict_disaster()
    4. Severity Assessment     -> severity_assessment.assess_severity()
    5. Result Generation       -> templates/result.html
    6. Recommendation          -> severity_assessment.RECOMMENDATIONS

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import json
import uuid
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from database import db, init_db, Prediction
from severity_assessment import assess_severity

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///disaster_predictions.db"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

init_db(app)

# Lazy import so the app can still boot (e.g. for the upload page) before a
# model file exists; predict_disaster() raises a clear error if it's missing.
from model_utils import predict_disaster


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No image selected.")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload a PNG or JPG image.")
            return redirect(request.url)

        # Stage 1: Image Upload
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)

        try:
            # Stage 2 + 3: Preprocessing + Disaster Classification
            disaster_type, confidence, all_probs, img_bgr = predict_disaster(filepath)

            # Stage 4: Severity Assessment
            severity, risk_level, affected_area_pct, recommendation = assess_severity(
                img_bgr, disaster_type, confidence
            )

            # Persist to DB (Stage 6 supports History / Report modules)
            record = Prediction(
                filename=unique_name,
                disaster_type=disaster_type,
                confidence=confidence,
                severity=severity,
                risk_level=risk_level,
                affected_area_pct=affected_area_pct,
                recommendation=recommendation,
                class_probabilities=json.dumps(all_probs),
            )
            db.session.add(record)
            db.session.commit()

            return redirect(url_for("result", prediction_id=record.id))

        except FileNotFoundError as e:
            flash(str(e))
            return redirect(request.url)
        except Exception as e:
            flash(f"Error processing image: {e}")
            return redirect(request.url)

    return render_template("index.html")


@app.route("/result/<int:prediction_id>")
def result(prediction_id):
    record = Prediction.query.get_or_404(prediction_id)
    probs = json.loads(record.class_probabilities) if record.class_probabilities else {}
    sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
    return render_template("result.html", r=record, probs=sorted_probs)


@app.route("/history")
def history():
    records = Prediction.query.order_by(Prediction.created_at.desc()).limit(50).all()
    return render_template("history.html", records=records)


@app.route("/api/predictions")
def api_predictions():
    """JSON API for the history dashboard / Chart.js visualizations."""
    records = Prediction.query.order_by(Prediction.created_at.desc()).limit(100).all()
    return jsonify([r.to_dict() for r in records])


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
