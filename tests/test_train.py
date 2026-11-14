import json

import cv2
import numpy as np
import pytest

from auric.data.sampling import class_loss_weights
from auric.model.train import imbalance_trainer, train_yolo, write_yolo_scene_dataset
from auric.data.dataset import ImageRecord
from auric.structures import Annotation

pytest.importorskip("ultralytics")

NAMES = ["a", "b", "c", "d", "e"]
ARGS = dict(epochs=1, imgsz=128, batch=4, nbs=4, optimizer="SGD", amp=False, plots=False, val=False, mosaic=1.0,
            close_mosaic=0, cache=False, deterministic=True, patience=0)


def _scenes(root, n=16):
    """n 128 px scenes, class 0 in most of them, class 4 in one: <root>/train/{images,labels}/<id>.{png,txt}."""
    rng = np.random.default_rng(0)
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir()
    records, anns = [], {}
    for i in range(n):
        cls = [0, 0, 1][: 1 + i % 3] + ([4] if i == 3 else []) + ([2] if i % 5 == 0 else [])
        cx, cy = rng.uniform(25, 100, (2, len(cls)))
        img = rng.integers(0, 60, (128, 128, 3), dtype=np.uint8)
        lines, boxes = [], []
        for c, x, y in zip(cls, cx, cy):
            img[int(y) - 10:int(y) + 10, int(x) - 10:int(x) + 10] = 50 * c + 40
            lines.append(f"{c} {x / 128:.4f} {y / 128:.4f} {20 / 128:.4f} {20 / 128:.4f}")
            boxes.append([x - 10, y - 10, x + 10, y + 10])
        path = root / "train" / "images" / f"{i}.png"
        cv2.imwrite(str(path), img)
        (root / "train" / "labels" / f"{i}.txt").write_text("\n".join(lines) + "\n")
        records.append(ImageRecord(str(i), path, root / "train" / "labels" / f"{i}.txt", 128, 128))
        anns[str(i)] = Annotation(np.array(boxes), np.array(cls))
    return records, anns


def test_scene_dataset_lists_the_source_images(tmp_path):
    records, anns = _scenes(tmp_path / "raw")
    ids = [r.image_id for r in records]
    ds = write_yolo_scene_dataset(tmp_path / "data", records, anns, ids[:12], ids[12:], NAMES)
    assert ds.lists["train"].read_text().count("\n") == 12 and ds.lists["val"].read_text().count("\n") == 4
    assert ds.summary.set_index("list").loc["train", "objects"] == sum(len(anns[i]) for i in ids[:12])
    assert "/train/images/0.png" in ds.lists["train"].read_text()


def test_class_weights_and_class_aware_sampler_are_applied_and_recorded(tmp_path):
    records, anns = _scenes(tmp_path / "raw")
    ids = [r.image_id for r in records]
    ds = write_yolo_scene_dataset(tmp_path / "data", records, anns, ids, ids, NAMES)

    out = tmp_path / "run"
    res = train_yolo("yolo26n.yaml", ds.data_yaml, out, {**ARGS, "cls_pw": 0.5}, seed=0, device="cpu", workers=0,
                     sampler={"alpha": 0.0})
    rec = res.imbalance
    counts = np.array(rec["dataset_counts"])
    assert counts.tolist() == np.bincount(np.concatenate([a.cls for a in anns.values()]), minlength=5).tolist()
    per_class = {r["class"]: r for r in rec["sampler"]["per_class"]}
    # alpha 0: every class is expected equally often, up to the samples shared between classes
    assert per_class["e"]["share % (expected per epoch)"] > per_class["e"]["share % (dataset)"]
    assert per_class["a"]["share % (expected per epoch)"] < per_class["a"]["share % (dataset)"]
    # cls_pw acts on the expected counts of the draws, not on the label counts; weights have mean 1
    expected = np.array([per_class[n]["share % (expected per epoch)"] for n in NAMES])
    w, present = np.array(rec["class_weights"]), counts > 0           # class "d" has no label: Ultralytics gives it the mean
    assert not present[3] and w.mean() == pytest.approx(1.0)
    assert np.allclose(w[present] / w[0], (expected[0] / expected[present]) ** 0.5, rtol=1e-3)
    # the preview formula is the one Ultralytics applied
    assert np.allclose(w, class_loss_weights([per_class[n]["instances (expected per epoch)"] for n in NAMES], 0.5), rtol=1e-4)
    assert json.loads((out / "imbalance.json").read_text()) == rec


def test_the_trainer_replaces_the_shuffle_with_the_weighted_sampler(tmp_path):
    from torch.utils.data import WeightedRandomSampler
    from ultralytics import YOLO

    records, anns = _scenes(tmp_path / "raw")
    ids = [r.image_id for r in records]
    ds = write_yolo_scene_dataset(tmp_path / "data", records, anns, ids, ids, NAMES)
    seen = {}

    def on_train_start(trainer):
        seen["sampler"] = trainer.train_loader.batch_sampler.sampler.sampler
        seen["len"] = len(trainer.train_loader)

    model = YOLO("yolo26n.yaml")
    model.add_callback("on_train_start", on_train_start)
    model.train(trainer=imbalance_trainer({"alpha": 0.5}, 0), data=str(ds.data_yaml), project=str(tmp_path),
                name="direct", exist_ok=True, device="cpu", workers=0, seed=0, **ARGS)
    assert isinstance(seen["sampler"], WeightedRandomSampler)
    assert seen["sampler"].num_samples == len(ids) and seen["len"] == len(ids) // 4      # epoch length unchanged
