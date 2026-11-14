"""Raw dataset access: class map, image index, YOLO labels -> absolute xyxy boxes.

Expected layout (as delivered):
    <root>/classmap.txt                      "<id> <name>" per line
    <root>/<split>/images/<image_id>.png
    <root>/<split>/labels/<image_id>.txt     "<cls> <xc> <yc> <w> <h>" normalised; may be empty

`parse_yolo_labels` is the only label parser: source images, cached tiles, training and evaluation all
read boxes through it. Only the directories named above are read; nothing else under the root is searched.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..structures import Annotation

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: Path
    label_path: Path
    width: int
    height: int


def read_class_names(root: str | Path) -> list[str]:
    names = []
    for line in (Path(root) / "classmap.txt").read_text().strip().splitlines():
        idx, name = line.split(maxsplit=1)
        if int(idx) != len(names):
            raise ValueError(f"classmap.txt: class ids must be 0..K-1 in order, got line {line!r}")
        names.append(name.strip())
    return names


def image_size(path: Path) -> tuple[int, int]:
    """(width, height); read from the PNG header when possible (no decoding)."""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    img = read_image(path)
    return img.shape[1], img.shape[0]


def read_image(path: str | Path) -> np.ndarray:
    """BGR uint8 (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image {path}")
    return img


def _natural_key(s: str):
    return (len(s), s)  # numeric ids sort numerically, others lexicographically


def index_split(root: str | Path, split: str) -> list[ImageRecord]:
    """All images of `<root>/<split>/images`, sorted by id. Every image must have a label file."""
    root = Path(root)
    image_dir = root / split / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    records = []
    for p in sorted(image_dir.iterdir(), key=lambda q: _natural_key(q.stem)):
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = root / split / "labels" / f"{p.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"missing label file for {p.name}: {label}")
        records.append(ImageRecord(p.stem, p, label, *image_size(p)))
    return records


def parse_yolo_labels(path: str | Path, width: int, height: int, num_classes: int) -> Annotation:
    """YOLO label file of a (width, height) image -> absolute xyxy boxes, clipped to the image.

    Clipping only absorbs float rounding in the files (20 train boxes overshoot a border by < 1e-6 px);
    anything else malformed raises.
    """
    text = Path(path).read_text().strip()
    if not text:
        return Annotation(np.zeros((0, 4)), np.zeros(0, np.int64))
    rows = np.array([ln.split() for ln in text.splitlines() if ln.strip()], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 5:
        raise ValueError(f"{path}: expected 5 columns per row")
    cls = rows[:, 0].astype(np.int64)
    if (cls < 0).any() or (cls >= num_classes).any() or (rows[:, 0] != cls).any():
        raise ValueError(f"{path}: class ids outside 0..{num_classes - 1}")
    xc, yc, w, h = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
    boxes = np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    if ((boxes[:, 2] - boxes[:, 0]) <= 0).any() or ((boxes[:, 3] - boxes[:, 1]) <= 0).any():
        raise ValueError(f"{path}: degenerate box")
    return Annotation(boxes, cls)


def load_annotations(records: list[ImageRecord], num_classes: int) -> dict[str, Annotation]:
    return {r.image_id: parse_yolo_labels(r.label_path, r.width, r.height, num_classes) for r in records}


def to_yolo_lines(boxes: np.ndarray, cls: np.ndarray, W: int, H: int) -> str:
    """Inverse of `parse_yolo_labels` for a (W, H) canvas."""
    if len(cls) == 0:
        return ""
    xc, yc = (boxes[:, 0] + boxes[:, 2]) / 2 / W, (boxes[:, 1] + boxes[:, 3]) / 2 / H
    w, h = (boxes[:, 2] - boxes[:, 0]) / W, (boxes[:, 3] - boxes[:, 1]) / H
    return "".join(f"{c} {a:.7f} {b:.7f} {d:.7f} {e:.7f}\n" for c, a, b, d, e in zip(cls, xc, yc, w, h))
