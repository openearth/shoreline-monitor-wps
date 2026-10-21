"""
First-party feedback endpoint for the Shoreline Monitor WPS.

Exposes:
  POST /feedback — submit profile reliability feedback and store it in Postgres

POST /feedback body (JSON):
  reliability         str   required  One of VALID_RELIABILITY
  details             list  optional  Cause checkboxes (required if Questionable)
  other_text          str   optional  Free text when 'Other:' is in details
  transect_id         str   optional  Must match gctr.transect_id for profileid
  profileid           int   optional  gctr.index

Transect metadata (center_lon, center_lat, country, continent, shore_type,
change_rate, change_rate_std_err, classification, r_squared) is looked up
server-side from gctr by profileid; any values sent in the body are ignored.
If profileid is missing, unknown, or does not match transect_id, the feedback
is stored without metadata and a warning is logged.

Client IP and User-Agent are captured server-side; do NOT send them in the body.
"""

import logging

from flask import Blueprint, abort, jsonify, request
from sqlalchemy import text

from .classification import classify_change_rate
from .client_ip import get_client_ip
from .db import get_engine

logger = logging.getLogger("PYWPS")

feedback_bp = Blueprint("feedback", __name__)

VALID_RELIABILITY = {
    "Reliable / representative",
    "Questionable (interpret with caution)",
    "Not reliable",
}

_METADATA_SQL = text(
    """
    SELECT
        g.transect_id,
        ST_X(ST_Centroid(g.geometry)) AS center_lon,
        ST_Y(ST_Centroid(g.geometry)) AS center_lat,
        c.common_country_name AS country,
        ct.continent,
        g.class_shore_type AS shore_type,
        g.sds_change_rate AS change_rate,
        g.sds_change_rate_std_err AS change_rate_std_err,
        g.sds_r_squared AS r_squared
    FROM public.gctr g
    LEFT JOIN public.country c ON c.idccn = g.idccn
    LEFT JOIN public.continent ct ON ct.idcnt = g.idcnt
    WHERE g.index = :profileid
    """
)

_METADATA_FIELDS = (
    "center_lon",
    "center_lat",
    "country",
    "continent",
    "shore_type",
    "change_rate",
    "change_rate_std_err",
    "classification",
    "r_squared",
)

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


def _lookup_metadata(conn, profileid, transect_id) -> dict:
    """Return transect metadata from gctr, or all-None if it cannot be verified."""
    empty = dict.fromkeys(_METADATA_FIELDS)

    if profileid is None:
        logger.warning("feedback: no profileid; storing without metadata")
        return empty

    found = conn.execute(_METADATA_SQL, {"profileid": profileid}).mappings().first()
    if found is None:
        logger.warning(
            "feedback: profileid %s not found in gctr; storing without metadata",
            profileid,
        )
        return empty

    if found["transect_id"] != transect_id:
        logger.warning(
            "feedback: transect_id mismatch for profileid %s (sent %r, gctr %r); "
            "storing without metadata",
            profileid,
            transect_id,
            found["transect_id"],
        )
        return empty

    metadata = {key: found[key] for key in _METADATA_FIELDS if key != "classification"}
    metadata["classification"] = classify_change_rate(found["change_rate"])
    return metadata


@feedback_bp.post("/feedback")
def submit_feedback():
    """Accept a profile reliability feedback submission and persist it."""
    body = request.get_json(silent=True) or {}
    reliability, details, other_text = _validate(body)

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
        "client_ip": get_client_ip(),
        "user_agent": request.headers.get("User-Agent") or None,
    }

    try:
        with get_engine().begin() as conn:
            row.update(_lookup_metadata(conn, row["profileid"], row["transect_id"]))
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
