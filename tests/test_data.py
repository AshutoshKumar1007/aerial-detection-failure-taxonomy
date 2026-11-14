import cv2
import numpy as np
import pytest

from auric.data.dataset import index_split, load_annotations, read_class_names, to_yolo_lines
from auric.data.splits import ensure_split
from auric.data.stats import image_table, object_table
from auric.data.tiling import TilingParams, ensure_tile_cache, read_tile, verify_tile_labels

NAMES = ["A", "B"]


def _dataset(root, n_images=6, W=700, H=500, seed=0):
    """Small images with random boxes; image 0 has no objects and a black (no-data) half."""
    rng = np.random.default_rng(seed)
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir()
    (root / "classmap.txt").write_text("0 A\n1 B\n")
    for i in range(n_images):
        img = rng.integers(20, 255, (H, W, 3), dtype=np.uint8)
        n = 0 if i == 0 else int(rng.integers(3, 12))
        if i == 0:
            img[:, : W // 2] = 0
        xy = rng.uniform(0, [W - 40, H - 40], (n, 2))
        boxes = np.concatenate([xy, xy + rng.uniform(6, 40, (n, 2))], 1)
        cv2.imwrite(str(root / "train" / "images" / f"{i}.png"), img)
        (root / "train" / "labels" / f"{i}.txt").write_text(to_yolo_lines(boxes, rng.integers(0, 2, n), W, H))
    recs = index_split(root, "train")
    return recs, load_annotations(recs, len(read_class_names(root)))


def test_tile_cache_status_reuse_and_labels(tmp_path):
    recs, anns = _dataset(tmp_path / "data")
    params = TilingParams(size=256, overlap=64, min_visibility=0.5, negative_frac=1.0, max_nodata_frac=0.5)
    cache = ensure_tile_cache(recs, anns, tmp_path / "tiles", params, len(NAMES), workers=1)
    m = cache.manifest
    assert cache.built and set(m.status) <= {"positive", "negative", "neg_nodata", "neg_sampled_out"}
    assert (m[m.status == "positive"].n_objects > 0).all() and (m[m.status != "positive"].n_objects == 0).all()
    assert (m.image_id == "0").any() and (m[m.image_id == "0"].status == "neg_nodata").any()
    assert "neg_sampled_out" not in set(m.status)            # negative_frac = 1 keeps every usable negative
    check = verify_tile_labels(cache, anns, len(NAMES))
    assert len(check) == len(cache.kept) and check.classes_match.all() and check.image_exists.all()
    assert check.max_err_px.max() < 1e-3
    img, ann = read_tile(cache, cache.kept.tile_id.iloc[0], len(NAMES))
    assert img.shape[:2] == (256, 256)

    again = ensure_tile_cache(recs, anns, tmp_path / "tiles", params, len(NAMES), workers=1)
    assert not again.built and len(again.manifest) == len(m)
    other = TilingParams(size=256, overlap=64, min_visibility=0.6, negative_frac=1.0, max_nodata_frac=0.5)
    with pytest.raises(RuntimeError, match="different tile cache"):
        ensure_tile_cache(recs, anns, tmp_path / "tiles", other, len(NAMES), workers=1)


def test_ensure_split_creates_then_loads(tmp_path):
    recs, anns = _dataset(tmp_path / "data", n_images=20)
    kw = dict(dev_frac=0.25, seed=0, n_trials=200, min_dev_instances={"B": 1}, dense_min_objects=10)
    split, created = ensure_split(tmp_path / "split.json", anns, NAMES, **kw)
    assert created and len(split["dev"]) == 5 and not set(split["dev"]) & set(split["train_core"])
    again, created = ensure_split(tmp_path / "split.json", anns, NAMES, **kw)
    assert not created and again["dev"] == split["dev"]
    with pytest.raises(ValueError, match="other parameters"):
        ensure_split(tmp_path / "split.json", anns, NAMES, **{**kw, "seed": 1})


def test_attribute_tables(tmp_path):
    recs, anns = _dataset(tmp_path / "data")
    obj = object_table(recs, anns, NAMES, [16, 32])
    img = image_table(recs, anns, NAMES)
    assert len(obj) == sum(len(a) for a in anns.values()) == img.n_objects.sum()
    assert (obj.aspect >= 1).all() and set(obj["class"]) <= set(NAMES)
    assert (img[["n A", "n B"]].sum(axis=1) == img.n_objects).all()
