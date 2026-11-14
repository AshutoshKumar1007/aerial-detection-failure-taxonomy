"""Containers shared by every stage, and the detections file format.

Convention: boxes are float64 (N, 4) arrays in absolute source-image pixels, xyxy; classes are int64 (N,).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Annotation:
    """Ground truth of one image."""
    boxes: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return len(self.cls)


@dataclass
class Detections:
    """Detections of one image (or one tile, in tile coordinates)."""
    boxes: np.ndarray
    scores: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return len(self.scores)

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64))

    @classmethod
    def concat(cls, items: list["Detections"]) -> "Detections":
        items = [d for d in items if len(d)]
        if not items:
            return cls.empty()
        return cls(np.concatenate([d.boxes for d in items]), np.concatenate([d.scores for d in items]),
                   np.concatenate([d.cls for d in items]))

    def select(self, idx) -> "Detections":
        return Detections(self.boxes[idx], self.scores[idx], self.cls[idx])


DET_COLUMNS = ["image_id", "cls", "score", "x1", "y1", "x2", "y2"]


def save_detections(dets: dict[str, Detections], path: str | Path) -> None:
    """CSV (gzip-compressed when the name ends in .gz), one row per detection."""
    frames = [pd.DataFrame({"image_id": i, "cls": d.cls, "score": d.scores, "x1": d.boxes[:, 0],
                            "y1": d.boxes[:, 1], "x2": d.boxes[:, 2], "y2": d.boxes[:, 3]})
              for i, d in dets.items() if len(d)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DET_COLUMNS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")


def load_detections(path: str | Path) -> dict[str, Detections]:
    """Images absent from the file have no detections (callers treat missing keys as empty)."""
    df = pd.read_csv(path, dtype={"image_id": str})
    return {i: Detections(g[["x1", "y1", "x2", "y2"]].to_numpy(np.float64), g["score"].to_numpy(np.float64),
                          g["cls"].to_numpy(np.int64))
            for i, g in df.groupby("image_id", sort=False)}
