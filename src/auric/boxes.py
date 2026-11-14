"""Box geometry (xyxy, no +1 pixel convention, as in pycocotools), size buckets and NMS."""
from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import batched_nms


def box_area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) pairwise IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def size_buckets(bins: list[float]) -> list[tuple[str, float, float]]:
    """[(label, lo, hi)] on sqrt(box area) in px; bins [16, 32] -> xs [0,16), s [16,32), m [32,inf)."""
    edges = [0.0, *map(float, bins), float("inf")]
    names = ["xs", "s", "m", "l", "xl"][:len(edges) - 1]
    return [(f"{nm} [{lo:g},{hi:g})", lo, hi) for nm, lo, hi in zip(names, edges[:-1], edges[1:])]


def size_bucket_labels(side: np.ndarray, bins: list[float]) -> np.ndarray:
    """Bucket label of each sqrt(area) value."""
    buckets = size_buckets(bins)
    labels = np.array([b[0] for b in buckets], object)
    return labels[np.searchsorted([lo for _, lo, _ in buckets[1:]], side, side="right")]


def class_nms(boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray, iou: float, agnostic: bool = False) -> np.ndarray:
    """Greedy NMS, per class (default) or across classes (`agnostic`). Returns kept indices in descending score order."""
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    keep = batched_nms(torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(scores, dtype=torch.float32),
                       torch.zeros(len(cls), dtype=torch.int64) if agnostic else torch.as_tensor(cls, dtype=torch.int64),
                       iou)
    return keep.numpy()
