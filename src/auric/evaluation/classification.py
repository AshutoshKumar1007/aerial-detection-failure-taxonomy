"""Five-way typing scores for a crop classifier: the numbers behind assignment 5.1 (perfect locations).

`classification_report` takes true labels and class probabilities and returns accuracy, balanced accuracy
(mean per-class recall, the fair number under imbalance), per-class precision / recall / F1 and the confusion
matrix (rows = true class). The chance levels are reported next to them, because 5-way accuracy on a 55%-Cargo
dataset is meaningless without the majority-class rate.

`matched_pairs` links detector boxes to the ground-truth objects they find, so the same objects can be typed from the
boxes a detector actually produces (real localisation) and from their ground-truth boxes (perfect localisation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..boxes import iou_matrix
from ..structures import Annotation


def classification_report(y_true: np.ndarray, proba: np.ndarray, class_names: list[str]) -> dict:
    y_true = np.asarray(y_true, np.int64)
    K = len(class_names)
    pred = np.asarray(proba).argmax(1)
    conf = np.zeros((K, K), np.int64)
    np.add.at(conf, (y_true, pred), 1)
    support, predicted = conf.sum(1), conf.sum(0)
    tp = np.diag(conf).astype(float)
    recall = np.divide(tp, support, out=np.full(K, np.nan), where=support > 0)
    precision = np.divide(tp, predicted, out=np.full(K, np.nan), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.full(K, np.nan), where=(precision + recall) > 0)
    per_class = pd.DataFrame({"class": class_names, "support": support, "predicted": predicted, "precision": precision,
                              "recall": recall, "F1": f1}).set_index("class")
    top2 = float(np.mean([y in np.argsort(-p)[:2] for y, p in zip(y_true, proba)])) if len(y_true) else float("nan")
    return {
        "n": int(len(y_true)),
        "accuracy": float(tp.sum() / max(len(y_true), 1)),
        "balanced_accuracy": float(np.nanmean(recall)),
        "top2_accuracy": top2,
        "chance_majority": float(support.max() / max(len(y_true), 1)),
        "chance_balanced": 1.0 / K,
        "per_class": per_class,
        "confusion": pd.DataFrame(conf, index=pd.Index(class_names, name="true"), columns=class_names),
    }


def matched_pairs(det_boxes: np.ndarray, det_scores: np.ndarray, gt: Annotation, iou_thr: float = 0.5) -> np.ndarray:
    """(M, 2) array of (detection index, ground-truth index): each ground-truth object paired with the
    highest-scoring detection that overlaps it at IoU >= `iou_thr`. Objects nothing reaches are absent; a detection
    serves at most one object (its best-overlapping one)."""
    if len(det_boxes) == 0 or len(gt) == 0:
        return np.zeros((0, 2), np.int64)
    iou = iou_matrix(np.asarray(det_boxes, np.float64), gt.boxes)
    best = iou.argmax(1)
    hit = np.flatnonzero(iou[np.arange(len(iou)), best] >= iou_thr)
    pairs = {}
    for d in hit[np.argsort(-np.asarray(det_scores)[hit], kind="stable")]:
        pairs.setdefault(int(best[d]), int(d))
    return np.array([(d, g) for g, d in sorted(pairs.items())], np.int64).reshape(-1, 2)
