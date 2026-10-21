"""Shared shoreline change-rate classification.

Used by the profile plot (shoreline_getprofile.py) and the feedback endpoint
(feedback.py) so both apply identical rules.
"""

import math


def classify_change_rate(rate):
    """Return the change-rate class label for a rate in m/yr, or None if unknown."""
    if rate is None or (isinstance(rate, float) and math.isnan(rate)):
        return None
    if rate < -5:
        return "Extreme erosion (< -5 m/yr)"
    if rate < -3.0:
        return "Severe erosion (-3.0 to -5.0 m/yr)"
    if rate < -1.0:
        return "Intense erosion (-1.0 to -3.0 m/yr)"
    if rate < -0.4:
        return "Erosion (-0.4 to -1.0 m/yr)"
    if rate <= 0.4:
        return "Stable (-0.4 to 0.4 m/yr)"
    if rate <= 1.0:
        return "Accretion (0.4 to 1.0 m/yr)"
    if rate <= 3.0:
        return "Intense accretion (1.0 to 3.0 m/yr)"
    if rate <= 5.0:
        return "Severe accretion (3.0 to 5.0 m/yr)"
    return "Extreme accretion (> 5 m/yr)"
