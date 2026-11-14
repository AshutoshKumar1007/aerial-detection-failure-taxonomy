"""Training subsets for the data-efficiency study (assignment 5.4: the smallest subset that keeps the full-data mAP50).

Size is the number of labelled *instances* (the annotation cost); images are what is kept or dropped, because the tile cache and
the detector work on whole source images. A subset is chosen from the `train_core` images only: dev and val are never touched.
Both methods fill the same instance budget with the same rule, so a method and its random control are matched on labels, not on images:

    random     images in a seeded random order
    coverage   images in a greedy order. Every cell (class x size bucket) has a target of `budget_frac` of its instances, raised to
               at most `min_cell` instances so that the rare cells are not left empty (a cell with fewer instances is taken whole).
               The next image is the one with the most still-missing cell mass per labelled instance it costs:
                   gain(i) = sum_c min(n_ic, deficit_c) / target_c ,   score(i) = gain(i) / n_i
               Images that no longer fill a deficit are added in seeded random order until the budget is spent.

An image is added when it still fits the budget; the fill stops at 98% of it (the largest scene holds 451 instances, so a
single image can be a large part of a small budget). Images without any object are not candidates: they carry no labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..boxes import size_buckets
from ..config import SUBSET_METHODS as METHODS
from ..structures import Annotation

FILL = 0.98


def cell_counts(annotations: dict[str, Annotation], ids: list[str], n_classes: int, size_bins: list[float]) -> np.ndarray:
    """(images, classes x size buckets) instance counts; cell = class * n_buckets + size bucket (sqrt of the box area)."""
    n_buckets = len(size_bins) + 1
    out = np.zeros((len(ids), n_classes * n_buckets), np.int64)
    for row, i in enumerate(ids):
        a = annotations[i]
        if len(a) == 0:
            continue
        side = np.sqrt((a.boxes[:, 2] - a.boxes[:, 0]) * (a.boxes[:, 3] - a.boxes[:, 1]))
        cell = a.cls * n_buckets + np.searchsorted(size_bins, side, side="right")
        out[row] = np.bincount(cell, minlength=out.shape[1])
    return out


def _coverage_order(cells: np.ndarray, budget_frac: float, min_cell: int, budget: int,
                    rng: np.random.Generator) -> tuple[list[int], int]:
    """Greedy order over the rows of `cells`, and how many leading rows were taken for a missing cell mass (the rest is random fill)."""
    n = cells.sum(1)
    total = cells.sum(0).astype(float)
    target = np.maximum(budget_frac * total, np.minimum(total, float(min_cell)))
    deficit, order, left = target.copy(), [], np.ones(len(cells), bool)
    used = 0
    while True:
        fits = left & (used + n <= budget)
        if not fits.any():
            break
        useful = np.minimum(cells, deficit) / np.where(target > 0, target, 1.0)
        score = np.where(fits, useful.sum(1) / np.maximum(n, 1), -1.0)
        best = int(np.argmax(score))
        if score[best] <= 0:
            break
        order.append(best)
        left[best] = False
        used += int(n[best])
        deficit = np.maximum(deficit - cells[best], 0.0)
        if used >= FILL * budget:
            return order, len(order)
    rest = [int(i) for i in rng.permutation(np.flatnonzero(left))]
    return order + rest, len(order)


def select_subset(annotations: dict[str, Annotation], ids: list[str], class_names: list[str], size_bins: list[float],
                  method: str, budget_frac: float, seed: int = 0, min_cell: int = 20) -> dict:
    """Choose training images for an instance budget of `budget_frac` of the instances of `ids`.

    Deterministic given its arguments. Returns {"ids", "method", "params", "instances", "budget", "images_from_coverage"}
    where `ids` keeps the order of the input list.
    """
    if method not in METHODS:
        raise ValueError(f"subset.method must be one of {METHODS}, got {method!r}")
    if not 0 < budget_frac <= 1:
        raise ValueError(f"subset.budget_frac must be in (0, 1], got {budget_frac}")
    ids = list(ids)
    cells = cell_counts(annotations, ids, len(class_names), size_bins)
    n = cells.sum(1)
    pool = np.flatnonzero(n > 0)
    budget = int(round(budget_frac * n.sum()))
    rng = np.random.default_rng(seed)
    if budget_frac == 1:                                 # the full set: every image that has labels
        order, n_gain = [int(i) for i in pool], 0
    elif method == "random":
        order, n_gain = [int(i) for i in rng.permutation(pool)], 0
    else:
        local, n_gain = _coverage_order(cells[pool], budget_frac, min_cell, budget, rng)
        order = [int(pool[i]) for i in local]
    chosen, used, n_cov = [], 0, 0
    for rank, i in enumerate(order):
        if used + n[i] <= budget:
            chosen.append(i)
            used += int(n[i])
            n_cov += rank < n_gain
            if used >= FILL * budget and budget_frac < 1:
                break
    keep = set(chosen)
    return {"ids": [ids[i] for i in range(len(ids)) if i in keep], "method": method,
            "params": {"budget_frac": budget_frac, "seed": seed, "min_cell": min_cell, "size_bins": list(size_bins)},
            "instances": used, "budget": budget, "images_from_coverage": int(n_cov)}


def describe_subset(annotations: dict[str, Annotation], all_ids: list[str], chosen_ids: list[str], class_names: list[str],
                    size_bins: list[float]) -> dict[str, pd.DataFrame]:
    """What a subset keeps, against the images it was chosen from: {"classes", "sizes", "images"} tables."""
    chosen = set(chosen_ids)
    labels = [b[0] for b in size_buckets(size_bins)]
    K, B = len(class_names), len(labels)
    cells = cell_counts(annotations, list(all_ids), K, size_bins)
    keep = np.array([i in chosen for i in all_ids])
    full, sub = cells.sum(0).reshape(K, B), cells[keep].sum(0).reshape(K, B)
    imgs_full, imgs_sub = (cells.reshape(len(cells), K, B).sum(2) > 0), (cells[keep].reshape(-1, K, B).sum(2) > 0)
    classes = pd.DataFrame({
        "instances (subset)": sub.sum(1), "instances (all)": full.sum(1),
        "kept %": 100 * sub.sum(1) / np.maximum(full.sum(1), 1),
        "share % (subset)": 100 * sub.sum(1) / sub.sum(), "share % (all)": 100 * full.sum(1) / full.sum(),
        "images with class (subset)": imgs_sub.sum(0), "images with class (all)": imgs_full.sum(0)}, index=class_names)
    sizes = pd.DataFrame({"instances (subset)": sub.sum(0), "instances (all)": full.sum(0),
                          "share % (subset)": 100 * sub.sum(0) / sub.sum(), "share % (all)": 100 * full.sum(0) / full.sum()},
                         index=labels)
    per_img = cells.sum(1)
    images = pd.DataFrame({"images": [int(keep.sum()), len(all_ids)],
                           "instances": [int(per_img[keep].sum()), int(per_img.sum())],
                           "instances / image (median)": [float(np.median(per_img[keep])), float(np.median(per_img))],
                           "instances / image (max)": [int(per_img[keep].max()), int(per_img.max())]},
                          index=["subset", "all"])
    return {"classes": classes.round(1), "sizes": sizes.round(1), "images": images}
