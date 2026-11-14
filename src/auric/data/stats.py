"""Per-object and per-image attribute tables (the input of the data audit and of later slice analyses).

Both are derived from parsed annotations only (no pixels), so they are cheap and identical everywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..boxes import iou_matrix, size_bucket_labels
from ..structures import Annotation
from .dataset import ImageRecord


def object_table(records: list[ImageRecord], annotations: dict[str, Annotation], class_names: list[str],
                 size_bins: list[float]) -> pd.DataFrame:
    """One row per object: geometry, size bucket and the largest overlap with another same / other-class box.

    `max_iou_same` > ~0.7 suggests a duplicate label; `max_iou_other` > ~0.5 two classes on one object.
    """
    frames = []
    for r in records:
        a = annotations[r.image_id]
        wh = a.boxes[:, 2:] - a.boxes[:, :2]
        iou = iou_matrix(a.boxes, a.boxes)
        np.fill_diagonal(iou, 0)
        same = a.cls[:, None] == a.cls[None, :]
        frames.append(pd.DataFrame({
            "image_id": r.image_id, "idx": np.arange(len(a)), "cls": a.cls,
            "x1": a.boxes[:, 0], "y1": a.boxes[:, 1], "x2": a.boxes[:, 2], "y2": a.boxes[:, 3],
            "w": wh[:, 0], "h": wh[:, 1],
            "max_iou_same": np.where(same, iou, 0).max(1, initial=0),
            "max_iou_other": np.where(~same, iou, 0).max(1, initial=0)}))
    df = pd.concat(frames, ignore_index=True)
    df.insert(3, "class", np.asarray(class_names, object)[df.cls.to_numpy()])
    df["sqrt_area"] = np.sqrt(df.w * df.h)
    df["aspect"] = np.maximum(df.w, df.h) / np.minimum(df.w, df.h)      # long side / short side, >= 1
    df["size"] = size_bucket_labels(df.sqrt_area.to_numpy(), size_bins)
    return df


def image_table(records: list[ImageRecord], annotations: dict[str, Annotation],
                class_names: list[str]) -> pd.DataFrame:
    """One row per image: size, object count, per-class object counts."""
    K = len(class_names)
    counts = np.stack([np.bincount(annotations[r.image_id].cls, minlength=K) for r in records])
    df = pd.DataFrame({"image_id": [r.image_id for r in records], "width": [r.width for r in records],
                       "height": [r.height for r in records], "n_objects": counts.sum(1)})
    return pd.concat([df, pd.DataFrame(counts, columns=[f"n {n}" for n in class_names])], axis=1)
