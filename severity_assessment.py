"""
severity_assessment.py
-----------------------
Stage 4 of the pipeline: Damage / Severity Assessment.

Severity is estimated using a hybrid approach:
  1. Model confidence for the predicted disaster class (higher confidence ->
     the visual evidence is stronger/clearer, contributing to severity).
  2. Disaster-specific image heuristics computed with OpenCV, used as a
     lightweight stand-in for a dedicated severity-regression head or a
     segmentation model (see README / Future Scope for the upgrade path).
  3. A weighted score is mapped to one of four severity bands, and a
     matching risk level + recommendation is generated.

This module is intentionally rule-based and explainable so it can be
swapped later for a trained regression/segmentation model without
changing the Flask route that calls it.
"""

import cv2
import numpy as np

SEVERITY_LEVELS = ["Low", "Moderate", "High", "Critical"]

# Disaster-specific recommendation templates.
# NOTE: This system is a decision-support / preliminary-assessment tool.
# It does not replace official disaster-management or emergency authorities.
RECOMMENDATIONS = {
    "Flood": {
        "Low": "Minor water accumulation detected. Monitor local weather and drainage.",
        "Moderate": "Moderate flooding detected. Avoid low-lying roads and monitor updates.",
        "High": "High flood severity detected. Avoid low-lying areas and prioritize emergency assessment.",
        "Critical": "Critical flooding detected. Evacuate low-lying areas immediately and alert local authorities.",
    },
    "Fire": {
        "Low": "Small fire/smoke signature detected. Keep monitoring the area.",
        "Moderate": "Moderate fire activity detected. Alert nearby residents and fire services.",
        "High": "High-intensity fire detected. Evacuate the vicinity and notify fire authorities immediately.",
        "Critical": "Critical, large-scale fire detected. Immediate evacuation and emergency response required.",
    },
    "Landslide": {
        "Low": "Minor slope disturbance detected. Restrict heavy vehicle movement nearby.",
        "Moderate": "Moderate landslide debris detected. Avoid the slope area and inspect for structural risk.",
        "High": "High landslide severity detected. Cordon off the area and assess nearby structures urgently.",
        "Critical": "Critical landslide detected. Evacuate downhill structures and alert geological/disaster teams.",
    },
    "Cyclone": {
        "Low": "Minor storm damage indicators detected. Continue monitoring weather advisories.",
        "Moderate": "Moderate cyclone damage detected. Secure loose structures and stay updated on advisories.",
        "High": "High cyclone damage detected. Avoid coastal/open areas and follow evacuation advisories.",
        "Critical": "Critical cyclone damage detected. Immediate evacuation and emergency shelter required.",
    },
    "Earthquake": {
        "Low": "Minor structural cracks detected. Recommend routine structural inspection.",
        "Moderate": "Moderate earthquake damage detected. Restrict building access pending inspection.",
        "High": "High earthquake damage detected. Evacuate the structure and request urgent structural assessment.",
        "Critical": "Critical structural collapse detected. Immediate evacuation and search-and-rescue coordination required.",
    },
    "Normal": {
        "Low": "No significant disaster signature detected in the image.",
        "Moderate": "No significant disaster signature detected in the image.",
        "High": "No significant disaster signature detected in the image.",
        "Critical": "No significant disaster signature detected in the image.",
    },
}


def _flood_heuristic(img_bgr):
    """Fraction of the image that looks like turbid/standing water (blue-green hue range)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 40, 40])
    upper = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.mean(mask > 0))


def _fire_heuristic(img_bgr):
    """Fraction of the image that looks like fire/flame (orange-red-yellow, high brightness)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 100, 150])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.mean(mask > 0))


def _landslide_heuristic(img_bgr):
    """Texture roughness (edge density) as a proxy for debris/rubble coverage."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.mean(edges > 0))


def _cyclone_heuristic(img_bgr):
    """Combination of sky-grey coverage and edge chaos as a proxy for storm damage."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    grey_sky = np.mean((gray > 90) & (gray < 180))
    return float(0.5 * np.mean(edges > 0) + 0.5 * grey_sky)


def _earthquake_heuristic(img_bgr):
    """Edge density as a proxy for cracked/collapsed structures."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.mean(edges > 0))


HEURISTICS = {
    "Flood": _flood_heuristic,
    "Fire": _fire_heuristic,
    "Landslide": _landslide_heuristic,
    "Cyclone": _cyclone_heuristic,
    "Earthquake": _earthquake_heuristic,
}


def estimate_affected_area(img_bgr, disaster_type):
    """Returns an approximate % of the image affected, using the disaster-specific heuristic."""
    if disaster_type not in HEURISTICS:
        return 0.0
    frac = HEURISTICS[disaster_type](img_bgr)
    return round(min(frac * 100 * 1.6, 100.0), 1)  # scaled + capped at 100%


def assess_severity(img_bgr, disaster_type, confidence_pct):
    """
    Combines model confidence and image heuristics into a single severity score.

    Returns:
        severity (str), risk_level (str), affected_area_pct (float), recommendation (str)
    """
    if disaster_type == "Normal":
        return "Low", "Low", 0.0, RECOMMENDATIONS["Normal"]["Low"]

    affected_area_pct = estimate_affected_area(img_bgr, disaster_type)

    # Weighted score: 60% affected-area heuristic, 40% model confidence
    score = 0.6 * affected_area_pct + 0.4 * confidence_pct

    if score < 25:
        severity = "Low"
    elif score < 50:
        severity = "Moderate"
    elif score < 75:
        severity = "High"
    else:
        severity = "Critical"

    # Risk level escalates one band above severity for fast-onset disasters (fire, flood)
    fast_onset = disaster_type in ("Fire", "Flood")
    idx = SEVERITY_LEVELS.index(severity)
    risk_idx = min(idx + 1, len(SEVERITY_LEVELS) - 1) if fast_onset else idx
    risk_level = SEVERITY_LEVELS[risk_idx]

    recommendation = RECOMMENDATIONS.get(disaster_type, {}).get(severity, "Preliminary assessment recommended.")

    return severity, risk_level, affected_area_pct, recommendation
