"""
First-party feedback endpoint for the Shoreline Monitor WPS.

Exposes:
  POST /feedback — submit profile reliability feedback and store it in Postgres

POST /feedback body (JSON):
  reliability         str   required  One of VALID_RELIABILITY
  details             list  optional  Cause checkboxes (required if Questionable)
  other_text          str   optional  Free text when 'Other:' is in details
  transect_id         str   optional
  profileid           int   optional  gctr.index
  center_lon          float optional
  center_lat          float optional
  country             str   optional
  continent           str   optional
  shore_type          str   optional
  change_rate         float optional
  change_rate_std_err float optional
  classification      str   optional
  r_squared           float optional

Client IP and User-Agent are captured server-side; do NOT send them in the body.
"""

import logging

from flask import Blueprint, abort, jsonify, request
from sqlalchemy import text

from .client_ip import get_client_ip
from .db import get_engine

logger = logging.getLogger("PYWPS")

feedback_bp = Blueprint("feedback", __name__)

VALID_RELIABILITY = {
    "Reliable / representative",
    "Questionable (interpret with caution)",
    "Not reliable",
}

_INSERT_SQL = text(
    """
    INSERT INTO public.shoreline_feedback (
        reliability, details, other_text,
        transect_id, profileid,
        center_lon, center_lat,
        country, continent, shore_type,
        change_rate, change_rate_std_err,
        classification, r_squared,
        client_ip, user_agent
    ) VALUES (
        :reliability, :details, :other_text,
        :transect_id, :profileid,
        :center_lon, :center_lat,
        :country, :continent, :shore_type,
        :change_rate, :change_rate_std_err,
        :classification, :r_squared,
        :client_ip, :user_agent
    )
    """
)


def _validate(body: dict) -> tuple[str, list, str | None]:
    """Validate the request body and return (reliability, details, other_text).

    Raises:
        HTTPException 422 on any validation failure.
    """
    reliability = (body.get("reliability") or "").strip()
    if reliability not in VALID_RELIABILITY:
        abort(422, description="Invalid or missing 'reliability'.")

    details: list = body.get("details") or []
    if not isinstance(details, list):
        abort(422, description="'details' must be a list.")

    other_text: str | None = (body.get("other_text") or "").strip() or None

    if "Questionable" in reliability:
        if not details:
            abort(422, description="At least one cause is required for 'Questionable'.")
        if "Other:" in details and not other_text:
            abort(422, description="'Other' selected but no text provided.")

    return reliability, details, other_text


@feedback_bp.post("/feedback")
def submit_feedback():
    """Accept a profile reliability feedback submission and persist it."""
    body = request.get_json(silent=True) or {}
    reliability, details, other_text = _validate(body)

    def _float(key):
        v = body.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _int(key):
        v = body.get(key)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    row = {
        "reliability": reliability,
        "details": details if details else None,
        "other_text": other_text,
        "transect_id": body.get("transect_id") or None,
        "profileid": _int("profileid"),
        "center_lon": _float("center_lon"),
        "center_lat": _float("center_lat"),
        "country": body.get("country") or None,
        "continent": body.get("continent") or None,
        "shore_type": body.get("shore_type") or None,
        "change_rate": _float("change_rate"),
        "change_rate_std_err": _float("change_rate_std_err"),
        "classification": body.get("classification") or None,
        "r_squared": _float("r_squared"),
        "client_ip": get_client_ip(),
        "user_agent": request.headers.get("User-Agent") or None,
    }

    try:
        with get_engine().begin() as conn:
            conn.execute(_INSERT_SQL, row)
    except Exception as err:
        logger.error("feedback: insert failed: %s", err)
        abort(500, description="Could not store feedback.")

    logger.info(
        "feedback: saved (reliability=%r, transect_id=%r, ip=%s)",
        reliability,
        row["transect_id"],
        row["client_ip"],
    )
    return jsonify({"status": "ok"}), 201


def init_feedback(app) -> None:
    """Register the feedback blueprint on the given Flask app.

    Call once per app instance in pywps.wsgi:
        from processes.feedback import init_feedback
        init_feedback(application)
    """
    app.register_blueprint(feedback_bp)
    logger.info("feedback: blueprint registered (POST /feedback)")
