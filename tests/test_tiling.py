from pathlib import Path

import numpy as np
import pytest

from auric.data.dataset import ImageRecord
from auric.data.tiling import crop_annotation, grid_starts, tile_windows
from auric.evaluation.metrics import evaluate
from auric.model.predict import stitch_ground_truth
from auric.structures import Annotation


@pytest.mark.parametrize("L,size,ov", [(3000, 1024, 200), (1369, 1024, 200), (800, 1024, 200), (7204, 640, 128)])
def test_grid_covers_without_padding(L, size, ov):
    s = grid_starts(L, size, ov)
    assert s[0] == 0 and min(s[-1] + size, L) == L
    assert all(x + size <= L or L <= size for x in s)
    assert all(b - a <= size - ov for a, b in zip(s, s[1:]))


def _scene(seed, W=3100, H=2900, n=300, max_side=160):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, [W - max_side, H - max_side], (n, 2))
    boxes = np.concatenate([xy, xy + rng.uniform(4, max_side, (n, 2))], 1)
    return Annotation(boxes, rng.integers(0, 5, n)), W, H


def test_every_object_whole_in_some_tile():
    ann, W, H = _scene(0)
    seen = np.zeros(len(ann), bool)
    for win in tile_windows(W, H, 1024, 200):
        seen[crop_annotation(ann, win, min_visibility=1.0)[2]] = True
    assert seen.all()


@pytest.mark.parametrize("edge_margin", [0, 4])
def test_stitching_ground_truth_recovers_the_scene(edge_margin):
    """GT standing in for detector output: tile -> stitch -> evaluate must give AP 1 (pipeline ceiling)."""
    ann, W, H = _scene(1)
    rec = ImageRecord("s", Path("s.png"), Path("s.txt"), W, H)
    dets = stitch_ground_truth([rec], {"s": ann}, 1024, 200, min_visibility=1.0, merge_iou=0.7,
                               edge_margin=edge_margin)
    r = evaluate({"s": ann}, dets, [f"c{k}" for k in range(5)], bootstrap=0)
    assert r.summary["mAP50"] == pytest.approx(1.0)
    assert len(dets["s"]) == len(ann)
