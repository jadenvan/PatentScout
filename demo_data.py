"""
PatentScout — Shared Demo Data Builders

Provides build_solar_demo_data() and build_doorbell_demo_data() functions
that are used by BOTH app.py (to populate session state) and
generate_final_pdfs.py (to produce PDFs).

This guarantees app tabs and PDF reports always display identical data.
"""

from __future__ import annotations

import os
import copy
import random
from pathlib import Path

import pandas as pd


# Import the mock-data builders from generate_final_pdfs
from scripts.generate_final_pdfs import (
    build_solar_session,
    build_doorbell_session,
    SOLAR_INVENTION_TEXT,
    DOORBELL_INVENTION_TEXT,
    DOORBELL_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# Sketch file — support multiple extensions
_SKETCH_EXTENSIONS = [".png", ".jpg", ".jpeg"]
SKETCH_DIR = os.path.join(PROJECT_ROOT, "assets", "demo")


def _find_sketch_path() -> str | None:
    """Locate the doorbell sketch file, trying several extensions."""
    for ext in _SKETCH_EXTENSIONS:
        path = os.path.join(SKETCH_DIR, f"doorbell_sketch{ext}")
        if os.path.exists(path):
            return path
    return None



# Solar Charger Demo Data

def build_solar_demo_data() -> dict:
    """Return the complete solar charger session dict.

    Returns the same mock data that generate_final_pdfs uses for the PDF,
    ensuring app tabs and PDF report always show identical information.
    """
    return build_solar_session()



# Smart Doorbell Demo Data (with optional sketch)

def _set_feature_sources(features: list[dict]) -> list[dict]:
    """Set correct source attribution on doorbell features.

    Features whose core concept comes from the text get source="text".
    Features where the sketch provides spatial/geometric information
    that text alone doesn't specify get source="sketch".
    """
    features = copy.deepcopy(features)

    # Sketch-enhanced features: the sketch shows camera POSITION, field of
    # view, and the motion sensor DETECTION ZONE geometry — spatial info
    # that the text description does not convey.
    sketch_labels = {"Integrated Camera Module", "Motion Detection Sensor"}

    for f in features:
        if f["label"] in sketch_labels:
            f["source"] = "sketch"
        else:
            f["source"] = "text"

    return features


def build_doorbell_demo_data() -> dict:
    """Return the complete smart doorbell session dict with sketch support.

    Adds sketch-sourced features and loads the sketch image bytes if available.
    """
    session = build_doorbell_session()

    # Set correct source attribution on features (text vs sketch)
    original_features = session["search_strategy"]["features"]
    enhanced_features = _set_feature_sources(original_features)
    session["search_strategy"]["features"] = enhanced_features

    # Load sketch image bytes
    sketch_path = _find_sketch_path()
    sketch_bytes = None
    sketch_used = False

    if sketch_path:
        try:
            with open(sketch_path, "rb") as f:
                sketch_bytes = f.read()
            sketch_used = True
        except OSError:
            pass

    session["invention_image"] = sketch_bytes
    session["sketch_used"] = sketch_used

    return session
