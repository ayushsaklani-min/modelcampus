"""
BentoML service definition for the time-to-failure prediction model.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import bentoml
from bentoml.io import JSON

from model_predictor import TimeToFailurePredictor


MODEL_DIR = os.getenv("MODEL_DIR", ".")
REQUIRED_FIELDS = [
    "equipment_type",
    "manufacturer",
    "facility_type",
    "equipment_age_days",
    "month",
    "year",
]


def _validate_payload(payload: Dict[str, Any]) -> List[str]:
    """Return a list of missing required fields."""
    return [field for field in REQUIRED_FIELDS if field not in payload]


predictor = TimeToFailurePredictor(model_dir=MODEL_DIR)
svc = bentoml.Service("failure_predictor")


@svc.api(input=JSON(), output=JSON())
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict time-to-failure for a single equipment payload.
    """
    missing = _validate_payload(payload)
    if missing:
        return {
            "success": False,
            "error": f"Missing required fields: {missing}",
        }

    try:
        days = predictor.predict(payload)
    except Exception as exc:  # pragma: no cover - surfacing model errors
        return {
            "success": False,
            "error": str(exc),
        }

    return {
        "success": True,
        "prediction": {
            "days_to_failure": round(days, 2),
            "weeks_to_failure": round(days / 7, 2),
            "months_to_failure": round(days / 30, 2),
        },
    }


@svc.api(input=JSON(), output=JSON())
def batch_predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict time-to-failure for a batch of equipment payloads.
    Expects {"items": [ {...}, {...} ]}.
    """
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return {
            "success": False,
            "error": 'Expected JSON body with non-empty "items" array',
        }

    missing_errors = [
        (index, _validate_payload(item))
        for index, item in enumerate(items)
    ]
    missing_errors = [(idx, missing) for idx, missing in missing_errors if missing]
    if missing_errors:
        return {
            "success": False,
            "error": "Missing required fields in batch",
            "details": [
                {"index": idx, "missing_fields": fields}
                for idx, fields in missing_errors
            ],
        }

    try:
        predictions = predictor.predict_batch(items)
    except Exception as exc:  # pragma: no cover
        return {
            "success": False,
            "error": str(exc),
        }

    results = [
        {
            "days_to_failure": round(value, 2),
            "weeks_to_failure": round(value / 7, 2),
            "months_to_failure": round(value / 30, 2),
        }
        for value in predictions
    ]

    return {
        "success": True,
        "count": len(results),
        "predictions": results,
    }


