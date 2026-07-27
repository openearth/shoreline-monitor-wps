"""
Shared server-side client IP extraction.

Reads from nginx-forwarded headers in order of trust:
  X-Forwarded-For (leftmost entry) -> X-Real-IP -> remote_addr

Do NOT use client-side IP lookup services (e.g. ip-api.com).
See also: processes/wps_shoreline_getprofile.py for the equivalent inline usage.
"""

from flask import request as _flask_request


def get_client_ip(req=None) -> str:
    """Return the original client IP behind an nginx reverse proxy.

    Args:
        req: Flask request object. Defaults to the active request context.

    Returns:
        IP address string (may be IPv4 or IPv6).
    """
    req = req or _flask_request
    forwarded_for = req.headers.get("X-Forwarded-For", "")
    real_ip = req.headers.get("X-Real-IP", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if real_ip:
        return real_ip.strip()
    return req.remote_addr
