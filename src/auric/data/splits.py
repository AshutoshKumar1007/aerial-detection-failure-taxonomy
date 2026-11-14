"""Image-level dev split carved out of train.

Val has only 22 images, so every model-selection decision is made on `dev` and val is only reported.
The split is created once, written to a JSON file that ships with the code (configs/splits/) and is
only *loaded* afterwards: `ensure_split` creates it when the file does not exist, otherwise loads it and
checks that it was made with the requested parameters and partitions the train images.

Method: seeded random search over image subsets of size round(dev_frac * N). A candidate must meet
the per-class instance floors; among those, the one closest to `dev_frac` on
    * the per-class share of instances      (AP denominators)
    * the per-class share of images         (scene diversity per class)
    * the share of dense images             (crowded scenes are a distinct regime)
is kept (L1 distance, equal weights).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..structures import Annotation
from .dataset import ImageRecord


def make_dev_split(annotations: dict[str, Annotation], class_names: list[str], dev_frac: float, seed: int,
                   n_trials: int, min_dev_instances: dict[str, int], dense_min_objects: int) -> dict:
    ids = list(annotations)
    K = len(class_names)
    counts = np.stack([np.bincount(annotations[i].cls, minlength=K) for i in ids]).astype(float)   # (N, K)
    present = (counts > 0).astype(float)
    dense = (counts.sum(1) >= dense_min_objects).astype(float)
    unknown = set(min_dev_instances) - set(class_names)
    if unknown:
        raise ValueError(f"min_dev_instances names unknown classes: {sorted(unknown)}")
    floor = np.array([min_dev_instances.get(n, 0) for n in class_names], float)
    tot_c, tot_p, tot_d = counts.sum(0), present.sum(0), max(dense.sum(), 1.0)

    rng = np.random.default_rng(seed)
    k = int(round(dev_frac * len(ids)))
    best, best_score = None, np.inf
    for _ in range(n_trials):
        idx = rng.choice(len(ids), k, replace=False)
        c = counts[idx].sum(0)
        if (c < floor).any():
            continue
        score = (np.abs(c / tot_c - dev_frac).sum() + np.abs(present[idx].sum(0) / tot_p - dev_frac).sum()
                 + abs(dense[idx].sum() / tot_d - dev_frac))
        if score < best_score:
            best, best_score = idx, score
    if best is None:
        raise RuntimeError("no candidate met min_dev_instances; lower the floors or raise n_trials")

    in_dev = np.zeros(len(ids), bool)
    in_dev[best] = True
    dev_c = counts[in_dev].sum(0)
    return {
        "params": {"dev_frac": dev_frac, "seed": seed, "n_trials": n_trials,
                   "min_dev_instances": min_dev_instances, "dense_min_objects": dense_min_objects},
        "score": round(float(best_score), 5),
        "stats": {"dev_images": int(in_dev.sum()), "train_core_images": int((~in_dev).sum()),
                  "dev_instances": {n: int(v) for n, v in zip(class_names, dev_c)},
                  "dev_instance_share": {n: round(float(v), 4) for n, v in zip(class_names, dev_c / tot_c)},
                  "dev_dense_images": int(dense[in_dev].sum())},
        "train_core": [i for i, d in zip(ids, in_dev) if not d],
        "dev": [i for i, d in zip(ids, in_dev) if d],
    }


def save_split(split: dict, path: str | Path, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; splits are immutable once created (pass overwrite=True deliberately)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))


def load_split(path: str | Path, train_ids: list[str]) -> dict:
    """Load and check that train_core and dev partition exactly the given train image ids."""
    split = json.loads(Path(path).read_text())
    core, dev = set(split["train_core"]), set(split["dev"])
    if core & dev:
        raise ValueError(f"{path}: train_core and dev overlap")
    if core | dev != set(train_ids):
        raise ValueError(f"{path}: split does not partition the train images of this dataset")
    return split


def ensure_split(path: str | Path, annotations: dict[str, Annotation], class_names: list[str], dev_frac: float,
                 seed: int, n_trials: int, min_dev_instances: dict[str, int],
                 dense_min_objects: int) -> tuple[dict, bool]:
    """(split, created): load the split at `path`, or create and save it there if the file does not exist."""
    params = {"dev_frac": dev_frac, "seed": seed, "n_trials": n_trials, "min_dev_instances": min_dev_instances,
              "dense_min_objects": dense_min_objects}
    if Path(path).exists():
        split = load_split(path, list(annotations))
        if split["params"] != params:
            raise ValueError(f"{path} was made with other parameters:\n found  {split['params']}\n wanted {params}")
        return split, False
    split = make_dev_split(annotations, class_names, **params)
    save_split(split, path)
    return split, True


def subset(records: list[ImageRecord], ids: list[str]) -> list[ImageRecord]:
    """Records of the given image ids, in record order."""
    wanted = set(ids)
    out = [r for r in records if r.image_id in wanted]
    if len(out) != len(wanted):
        raise KeyError(f"{len(wanted) - len(out)} image ids not among the records")
    return out
