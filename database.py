"""
database.py
------------
SQLite database models for the AI-Based Multi-Disaster Detection and
Damage Severity Assessment System.

Stores every prediction made by the system (image reference, predicted
disaster class, confidence score, severity level, risk level and
timestamp) so a History / Report module can be built on top of it.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Prediction(db.Model):
    """A single disaster-image prediction record."""

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    disaster_type = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)          # 0-100 (%)
    severity = db.Column(db.String(20), nullable=False)        # Low/Moderate/High/Critical
    risk_level = db.Column(db.String(20), nullable=False)      # Low/Moderate/High/Critical
    affected_area_pct = db.Column(db.Float, nullable=True)     # optional, from segmentation
    recommendation = db.Column(db.Text, nullable=False)
    class_probabilities = db.Column(db.Text, nullable=True)    # JSON string of all class probs
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "disaster_type": self.disaster_type,
            "confidence": round(self.confidence, 2),
            "severity": self.severity,
            "risk_level": self.risk_level,
            "affected_area_pct": self.affected_area_pct,
            "recommendation": self.recommendation,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


def init_db(app):
    """Attach SQLAlchemy to the Flask app and create tables if missing."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
