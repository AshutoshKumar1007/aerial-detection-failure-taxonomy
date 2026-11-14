"""Tile grid (shared by training and inference) and the training tile cache.

Grid: stride = size - overlap, with the last row/column snapped to the image border, so tiles are never
padded and every pixel is covered. With overlap > largest object, every object lies whole in >= 1 tile.

Cache layout (`<cache_dir>`, conventionally `<work_dir>/tiles/<TilingParams.key>`):
    images/<tile_id>.png    native-resolution crops of the kept tiles (lossless)
    labels/<tile_id>.txt    YOLO labels in tile coordinates
    manifest.csv            one row per grid tile of every source image, kept or not (`status`)
    tiling.json             parameters + source fingerprint; written last, so it marks a complete cache

Tile status:
    positive         >= 1 object with visible share >= min_visibility                     kept
    negative         object-free, deterministic per-tile hash < negative_frac, mostly imagery  kept
    neg_sampled_out  object-free, hash >= negative_frac                                    not written
    neg_nodata       object-free, black no-data share > max_nodata_frac                    not written
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..boxes import box_area
from ..structures import Annotation
from .dataset import ImageRecord, parse_yolo_labels, read_image, to_yolo_lines

Window = tuple[int, int, int, int]   # x0, y0, x1, y1 in image pixels
KEPT = ("positive", "negative")
FORMAT = 2                           # bump when the cache contents change meaning


@dataclass(frozen=True)
class TilingParams:
    size: int
    overlap: int             # > largest object side, so every object lies whole in at least one tile
    min_visibility: float    # a box cut by the tile border is kept (clipped) if >= this share of its area is inside
    negative_frac: float     # share of object-free tiles kept
    max_nodata_frac: float   # object-free tiles with more black no-data pixels than this are discarded

    def __post_init__(self):
        if not 0 <= self.overlap < self.size:
            raise ValueError("tile overlap must be in [0, size)")

    @property
    def key(self) -> str:
        return "s{size}_o{overlap}_v{min_visibility}_n{negative_frac}_d{max_nodata_frac}".format(**asdict(self))


@dataclass
class TileCache:
    dir: Path
    params: TilingParams
    manifest: pd.DataFrame   # all grid tiles; kept ones have status in KEPT
    built: bool              # True: built by this call, False: an existing cache was verified and reused

    @property
    def kept(self) -> pd.DataFrame:
        return self.manifest[self.manifest.status.isin(KEPT)]

    def image_path(self, tile_id: str) -> Path:
        return self.dir / "images" / f"{tile_id}.png"

    def label_path(self, tile_id: str) -> Path:
        return self.dir / "labels" / f"{tile_id}.txt"


def grid_starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size - overlap))
    if starts[-1] + size < length:
        starts.append(length - size)
    return starts


def tile_windows(W: int, H: int, size: int, overlap: int) -> list[Window]:
    return [(x, y, min(x + size, W), min(y + size, H))
            for y in grid_starts(H, size, overlap) for x in grid_starts(W, size, overlap)]


def crop_annotation(ann: Annotation, win: Window, min_visibility: float):
    """Boxes of `ann` inside `win`, in window coordinates.

    Returns (boxes, cls, kept_idx, n_dropped): a box cut by the window border is kept, clipped, when at
    least `min_visibility` of its area is inside; partially visible boxes below that are counted as dropped.
    """
    x0, y0, x1, y1 = win
    if len(ann) == 0:
        return np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0, np.int64), 0
    clipped = np.column_stack([ann.boxes[:, 0].clip(x0, x1), ann.boxes[:, 1].clip(y0, y1),
                               ann.boxes[:, 2].clip(x0, x1), ann.boxes[:, 3].clip(y0, y1)])
    visible = box_area(clipped) / box_area(ann.boxes)
    keep = visible >= min_visibility
    n_dropped = int(((visible > 0) & ~keep).sum())
    return clipped[keep] - np.array([x0, y0, x0, y0], float), ann.cls[keep], np.flatnonzero(keep), n_dropped


def count_uncovered(records: list[ImageRecord], annotations: dict[str, Annotation], size: int,
                    overlap: int) -> int:
    """Objects that do not lie whole in any tile of the grid (0 when overlap > largest object side)."""
    n = 0
    for r in records:
        a = annotations[r.image_id]
        seen = np.zeros(len(a), bool)
        for win in tile_windows(r.width, r.height, size, overlap):
            seen[crop_annotation(a, win, 1.0)[2]] = True
        n += int((~seen).sum())
    return n


def tile_uniform(tile_id: str) -> float:
    """Deterministic U[0, 1) per tile id, identical across machines and Python versions."""
    return int(hashlib.md5(tile_id.encode()).hexdigest()[:8], 16) / 2 ** 32


def source_fingerprint(records: list[ImageRecord]) -> str:
    """Identifies the source images (ids, sizes) and their exact label contents."""
    h = hashlib.sha1()
    for r in records:
        h.update(f"{r.image_id}:{r.width}x{r.height}:".encode())
        h.update(r.label_path.read_bytes())
    return h.hexdigest()[:16]


def _tile_image(args) -> list[dict]:
    rec, ann, out_dir, p, num_classes = args
    img = read_image(rec.image_path)
    rows = []
    for win in tile_windows(rec.width, rec.height, p.size, p.overlap):
        x0, y0, x1, y1 = win
        tile_id = f"{rec.image_id}_{x0}_{y0}"
        boxes, cls, _, n_dropped = crop_annotation(ann, win, p.min_visibility)
        crop = img[y0:y1, x0:x1]
        nodata = float((crop.max(axis=2) <= 2).mean())
        if len(cls):
            status = "positive"
        elif nodata > p.max_nodata_frac:
            status = "neg_nodata"
        elif tile_uniform(tile_id) >= p.negative_frac:
            status = "neg_sampled_out"
        else:
            status = "negative"
        if status in KEPT:
            cv2.imwrite(str(out_dir / "images" / f"{tile_id}.png"), crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            (out_dir / "labels" / f"{tile_id}.txt").write_text(to_yolo_lines(boxes, cls, x1 - x0, y1 - y0))
        row = {"tile_id": tile_id, "image_id": rec.image_id, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
               "status": status, "n_objects": len(cls), "n_dropped": n_dropped, "nodata": round(nodata, 4)}
        row.update({f"n_cls{k}": int(v) for k, v in enumerate(np.bincount(cls, minlength=num_classes))})
        rows.append(row)
    return rows


def ensure_tile_cache(records: list[ImageRecord], annotations: dict[str, Annotation], cache_dir: str | Path,
                      params: TilingParams, num_classes: int, workers: int) -> TileCache:
    """Reuse the cache in `cache_dir` if it was built from identical parameters and sources, else build it.

    A complete cache with other parameters or sources, or an incomplete one, raises instead of being reused
    or overwritten (delete it deliberately).
    """
    cache_dir = Path(cache_dir)
    spec = {"format": FORMAT, "params": asdict(params), "source": source_fingerprint(records),
            "images": len(records)}
    marker = cache_dir / "tiling.json"
    if marker.exists():
        found = json.loads(marker.read_text())
        if found != spec:
            raise RuntimeError(f"{cache_dir} holds a different tile cache:\n found  {found}\n wanted {spec}")
        return TileCache(cache_dir, params, read_manifest(cache_dir), built=False)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise RuntimeError(f"{cache_dir} exists without tiling.json (interrupted build?); delete it and rebuild")

    (cache_dir / "images").mkdir(parents=True)
    (cache_dir / "labels").mkdir()
    jobs = [(r, annotations[r.image_id], cache_dir, params, num_classes) for r in records]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for image_rows in pool.map(_tile_image, jobs, chunksize=1) for row in image_rows]
    manifest = pd.DataFrame(rows)
    manifest.to_csv(cache_dir / "manifest.csv", index=False)
    marker.write_text(json.dumps(spec, indent=1))
    return TileCache(cache_dir, params, manifest, built=True)


def read_manifest(cache_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(cache_dir) / "manifest.csv", dtype={"tile_id": str, "image_id": str})


def read_tile(cache: TileCache, tile_id: str, num_classes: int) -> tuple[np.ndarray, Annotation]:
    """A cached tile as the detector sees it: BGR image + labels parsed from the label file."""
    img = read_image(cache.image_path(tile_id))
    return img, parse_yolo_labels(cache.label_path(tile_id), img.shape[1], img.shape[0], num_classes)


def verify_tile_labels(cache: TileCache, annotations: dict[str, Annotation], num_classes: int) -> pd.DataFrame:
    """Re-read every kept tile's label file and compare it with the source labels cropped to the window.

    One row per kept tile: object counts (file / expected), class agreement and the largest corner error
    in px (YOLO text rounding is ~1e-4 px at 1024 px).
    """
    rows = []
    for t in cache.kept.itertuples():
        win = (t.x0, t.y0, t.x1, t.y1)
        boxes, cls, _, _ = crop_annotation(annotations[t.image_id], win, cache.params.min_visibility)
        got = parse_yolo_labels(cache.label_path(t.tile_id), t.x1 - t.x0, t.y1 - t.y0, num_classes)
        same_n = len(got) == len(cls)
        rows.append({"tile_id": t.tile_id, "n_file": len(got), "n_expected": len(cls),
                     "classes_match": same_n and bool((got.cls == cls).all()),
                     "max_err_px": float(np.abs(got.boxes - boxes).max(initial=0)) if same_n else np.inf,
                     "image_exists": cache.image_path(t.tile_id).is_file()})
    return pd.DataFrame(rows)
