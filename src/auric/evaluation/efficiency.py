"""Data-efficiency arithmetic for assignment 5.3 / 5.4: the recovery threshold and the score gained per doubling of labels."""
from __future__ import annotations

import numpy as np


def recovery_threshold(reference_scores: list[float], fraction: float = 0.9) -> dict:
    """The score a subset must reach: `fraction` of the mean of the full-data runs (different training seeds), with their spread.

    {"reference": mean, "threshold": fraction * mean, "min", "max", "n"}: the spread of the reference runs is the seed noise
    the 10% margin has to be read against.
    """
    ref = np.asarray(reference_scores, float)
    if ref.size == 0 or not np.isfinite(ref).all():
        raise ValueError("need at least one finite reference score")
    return {"reference": float(ref.mean()), "threshold": float(fraction * ref.mean()), "min": float(ref.min()),
            "max": float(ref.max()), "n": int(ref.size)}


def log2_slope(budgets: list[float], scores: list[float]) -> dict:
    """Least-squares line score = a + slope * log2(budget): {"slope" (score per doubling), "se", "intercept", "n"}.

    `se` is the standard error of the slope from the residuals (nan with fewer than three points).
    """
    x, y = np.log2(np.asarray(budgets, float)), np.asarray(scores, float)
    if len(x) < 2 or np.ptp(x) == 0:
        raise ValueError("need at least two different budgets")
    slope, intercept = np.polyfit(x, y, 1)
    se = np.nan
    if len(x) > 2:
        resid = y - (intercept + slope * x)
        se = float(np.sqrt(resid @ resid / (len(x) - 2) / ((x - x.mean()) ** 2).sum()))
    return {"slope": float(slope), "se": se, "intercept": float(intercept), "n": int(len(x))}


def gain_for_extra_labels(slope: float, current: float, extra: float) -> float:
    """Score change predicted for `extra` more labels on top of `current`, along a score-per-doubling slope."""
    return float(slope * np.log2((current + extra) / current))
