"""Image rendering for inspection (arrays in, arrays out; displaying them is the caller's business)."""
from __future__ import annotations

import cv2
import numpy as np

CLASS_COLORS = [(230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180)]   # RGB per class id


def draw_boxes(img: np.ndarray, boxes: np.ndarray, colors, thickness: int = 2) -> np.ndarray:
    """Copy of `img` with xyxy `boxes` drawn; `colors` is one colour per box."""
    img = np.ascontiguousarray(img).copy()
    for b, c in zip(np.asarray(boxes).round().astype(int), colors):
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), tuple(int(v) for v in c), thickness)
    return img


def crop_around(img: np.ndarray, box, context: float = 1.5, min_side: float = 12) -> tuple[np.ndarray, tuple[int, int]]:
    """Square-ish crop centred on `box` with `context` x its size around it. Returns (crop, (x0, y0))."""
    H, W = img.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(box[2] - box[0], box[3] - box[1], min_side) * (0.5 + context)
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, W)), int(min(cy + half, H))
    return img[y0:y1, x0:x1], (x0, y0)


def downscale(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Image resized so that its longer side is at most `max_side` (area interpolation). Returns (img, scale)."""
    s = min(1.0, max_side / max(img.shape[:2]))
    return (cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img), s
