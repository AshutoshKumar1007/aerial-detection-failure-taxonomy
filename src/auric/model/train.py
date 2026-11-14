"""YOLO training on the tile cache (Ultralytics).

The run gets its own image lists and data yaml, so one tile cache serves many runs: absolute paths into the
cache (tiles formulation) or to the source images (scenes formulation, no cache). Every training argument comes
from the config (`train.args`), so the saved config fully specifies the run independently of Ultralytics defaults;
the arguments Ultralytics actually used are read back from its `args.yaml`.

Class imbalance has two independent levers, both recorded in `train/imbalance.json`:
    train.args.cls_pw   Ultralytics' class weights: (1 / instances) ** cls_pw, scaled to mean 1, multiply the
                        classification loss of each class. They are computed on the class counts the detector is
                        expected to see, i.e. after the sampler below, so the two levers do not count a class twice.
    train.sampler       class-aware draws of the training samples (`data/sampling.py`); absent: Ultralytics' shuffle.
"""
from __future__ import annotations

import json
import logging
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from ..data.dataset import ImageRecord
from ..data.sampling import class_aware_probs, describe_sampling, expected_counts
from ..data.tiling import TileCache
from ..structures import Annotation


@dataclass
class YoloDataset:
    data_yaml: Path
    lists: dict[str, Path]       # "train" / "val" -> text file with one tile image path per line
    summary: pd.DataFrame        # per list: source images, tiles, positive / negative tiles, objects


@dataclass
class TrainResult:
    last: Path
    best: Path | None            # best dev-tile fitness (Ultralytics); recorded, not used for evaluation
    results_csv: Path
    effective_args: dict         # arguments Ultralytics trained with (args.yaml)
    imbalance: dict | None       # imbalance.json: class counts, class weights, sampler summary


@dataclass
class CheckpointInfo:
    epochs_completed: int
    epochs_planned: int
    date: str | None             # when Ultralytics wrote the checkpoint
    ultralytics: str | None      # Ultralytics version that wrote it
    train_args: dict
    train_results: dict | None   # results.csv columns at the time of writing

    @property
    def finished(self) -> bool:
        return self.epochs_completed >= self.epochs_planned

    def summary(self) -> dict:
        return {"epochs_completed": self.epochs_completed, "epochs_planned": self.epochs_planned,
                "finished": self.finished, "date": self.date, "ultralytics": self.ultralytics}


def _write_yolo_dataset(out_dir: Path, paths: dict[str, list[Path]], rows: list[dict],
                        class_names: list[str]) -> YoloDataset:
    out_dir.mkdir(parents=True, exist_ok=True)
    lists = {}
    for name, images in paths.items():
        lists[name] = out_dir / f"{name}.txt"
        lists[name].write_text("".join(f"{p.resolve().as_posix()}\n" for p in images))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": out_dir.resolve().as_posix(), "train": lists["train"].resolve().as_posix(),
        "val": lists["val"].resolve().as_posix(), "names": dict(enumerate(class_names))}, sort_keys=False))
    return YoloDataset(data_yaml, lists, pd.DataFrame(rows))


def write_yolo_dataset(out_dir: str | Path, cache: TileCache, train_ids: list[str], val_ids: list[str],
                       class_names: list[str]) -> YoloDataset:
    """train.txt / val.txt with the kept tiles of the given source images, and data.yaml."""
    paths, rows = {}, []
    for name, ids in (("train", train_ids), ("val", val_ids)):
        tiles = cache.kept[cache.kept.image_id.isin(set(ids))]
        if tiles.empty:
            raise ValueError(f"no cached tiles for the {name} images")
        paths[name] = [cache.image_path(t) for t in tiles.tile_id]
        rows.append({"list": name, "source_images": len(set(ids)), "tiles": len(tiles),
                     "positive": int((tiles.status == "positive").sum()),
                     "negative": int((tiles.status == "negative").sum()), "objects": int(tiles.n_objects.sum())})
    return _write_yolo_dataset(Path(out_dir), paths, rows, class_names)


def write_yolo_scene_dataset(out_dir: str | Path, records: list[ImageRecord], annotations: dict[str, Annotation],
                             train_ids: list[str], val_ids: list[str], class_names: list[str]) -> YoloDataset:
    """Scenes formulation: train.txt / val.txt list the source images themselves (no tile cache), and data.yaml.

    Ultralytics finds a label file by swapping `/images/` for `/labels/` in the image path, which is the layout
    `index_split` requires. It also writes a `labels.cache` next to the labels when that folder is writable.
    """
    by_id = {r.image_id: r for r in records}
    paths, rows = {}, []
    for name, ids in (("train", train_ids), ("val", val_ids)):
        if not ids:
            raise ValueError(f"no images for the {name} list")
        paths[name] = [by_id[i].image_path for i in ids]
        n = [len(annotations[i]) for i in ids]
        rows.append({"list": name, "source_images": len(ids), "tiles": len(ids), "positive": sum(x > 0 for x in n),
                     "negative": sum(x == 0 for x in n), "objects": sum(n)})
    return _write_yolo_dataset(Path(out_dir), paths, rows, class_names)


def fetch_pretrained(name: str, dest_dir: str | Path) -> Path:
    """`dest_dir/name`, downloaded from the Ultralytics release assets if it is not there yet."""
    from ultralytics.utils.downloads import attempt_download_asset

    dest = Path(dest_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        attempt_download_asset(dest)
    if not dest.is_file():
        raise FileNotFoundError(f"could not obtain pretrained weights {name!r} at {dest}")
    return dest


class _Progress:
    """Ultralytics callbacks -> an epoch bar (losses, dev-tile mAP50) and a per-epoch batch bar."""

    def __init__(self):
        self.epochs = self.batches = None

    def _close_batches(self):
        if self.batches is not None:
            self.batches.close()
            self.batches = None

    def on_train_start(self, tr):
        self.epochs = tqdm(total=tr.epochs, initial=tr.start_epoch, desc="epochs", unit="epoch")

    def on_train_epoch_start(self, tr):
        self._close_batches()   # an epoch restarted after an OOM / NaN recovery
        self.batches = tqdm(total=len(tr.train_loader), desc=f"epoch {tr.epoch + 1}", unit="batch", leave=False)

    def on_train_batch_end(self, tr):
        self.batches.update(1)
        if tr.tloss:
            self.batches.set_postfix({k: f"{float(v):.3f}" for k, v in tr.tloss.items()}, refresh=False)

    def on_fit_epoch_end(self, tr):
        self._close_batches()
        if self.epochs is None or self.epochs.n >= self.epochs.total:
            return              # Ultralytics' final validation of best.pt after the last epoch
        post = {k: f"{float(v):.3f}" for k, v in (tr.tloss or {}).items()}
        post["dev mAP50"] = f"{tr.metrics.get('metrics/mAP50(B)', float('nan')):.3f}"
        self.epochs.set_postfix(post, refresh=False)
        self.epochs.update(1)

    def on_train_end(self, tr):
        self._close_batches()
        if self.epochs is not None:
            self.epochs.close()

    def attach(self, model):
        for event in ("on_train_start", "on_train_epoch_start", "on_train_batch_end", "on_fit_epoch_end",
                      "on_train_end"):
            model.add_callback(event, getattr(self, event))


@contextmanager
def _ultralytics_log_level(level: int):
    from ultralytics.utils import LOGGER

    old = LOGGER.level
    LOGGER.setLevel(level)
    try:
        yield
    finally:
        LOGGER.setLevel(old)


def imbalance_trainer(sampler: dict | None, seed: int):
    """An Ultralytics detection trainer that records the class balance and can draw the samples class-aware.

    With `sampler` ({"alpha": ...}) the training loader draws N samples per epoch (N = dataset size, so an epoch
    keeps its length) with the probabilities of `class_aware_probs`, from a generator seeded with `seed`. The sampler
    replaces the shuffling `RandomSampler` that Ultralytics 8.4.179 puts inside its loader; anything else raises.
    Mosaic partners are taken from the samples loaded most recently, so they follow the same draws.
    `cls_pw` then uses the expected class counts of the draws (without a sampler: the label counts, as Ultralytics does).
    Everything it decided is written to `<save_dir>/imbalance.json`.
    """
    import torch
    from torch.utils.data import RandomSampler, WeightedRandomSampler
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class ImbalanceTrainer(DetectionTrainer):
        sample_counts = None     # (samples, classes): instances of each class in each training sample
        sample_probs = None

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
            if mode != "train" or sampler is None:
                return loader
            batches = loader.batch_sampler.sampler           # _RepeatSampler -> torch BatchSampler -> RandomSampler
            if not isinstance(getattr(batches, "sampler", None), RandomSampler):
                raise RuntimeError("train.sampler needs single-GPU training with shuffling (no DDP, no rect)")
            nc = self.data["nc"]
            self.sample_counts = np.stack([np.bincount(lb["cls"].reshape(-1).astype(int), minlength=nc)
                                           for lb in loader.dataset.labels])
            self.sample_probs = class_aware_probs(self.sample_counts, sampler["alpha"])
            batches.sampler = WeightedRandomSampler(
                self.sample_probs, num_samples=len(self.sample_probs), replacement=True,
                generator=torch.Generator().manual_seed(seed))
            return loader

        def get_class_counts(self):
            if self.sample_probs is None:
                return super().get_class_counts()
            return expected_counts(self.sample_counts, self.sample_probs).astype(np.float32)

        def set_class_weights(self):
            super().set_class_weights()
            weights = getattr(unwrap_model(self.model), "class_weights", None)
            names = [self.data["names"][i] for i in range(self.data["nc"])]
            record = {"cls_pw": self.args.cls_pw, "class_names": names,
                      "dataset_counts": DetectionTrainer.get_class_counts(self).astype(int).tolist(),
                      "class_weights": None if weights is None else weights.cpu().numpy().tolist(), "sampler": None}
            if self.sample_probs is not None:
                per_class = describe_sampling(self.sample_counts, self.sample_probs, names).to_dict("records")
                record["sampler"] = {**sampler, "seed": seed, "samples": len(self.sample_probs), "per_class": per_class}
            (self.save_dir / "imbalance.json").write_text(json.dumps(record, indent=1))

    return ImbalanceTrainer


def train_yolo(weights: str | Path, data_yaml: str | Path, out_dir: str | Path, args: dict, seed: int, device,
               workers: int, resume: bool = False, verbose: bool = False, sampler: dict | None = None) -> TrainResult:
    """Train from `weights` into `out_dir`, or resume the interrupted training in `out_dir`.

    verbose=False shows tqdm progress bars and Ultralytics warnings only; True shows Ultralytics' own log.
    `sampler` is `train.sampler` of the config (None: Ultralytics' shuffling); a resumed run must be given the same.
    """
    from ultralytics import YOLO

    out_dir = Path(out_dir).resolve()
    last = out_dir / "weights" / "last.pt"
    if resume:
        if not last.exists():
            raise FileNotFoundError(f"nothing to resume: {last}")
        model, kwargs = YOLO(str(last)), {"resume": True}
    else:
        if last.exists():
            raise FileExistsError(f"{out_dir} already holds a trained model")
        model = YOLO(str(weights))
        kwargs = dict(data=str(data_yaml), project=str(out_dir.parent), name=out_dir.name, exist_ok=True,
                      seed=seed, device=device, workers=workers, **args)
    if not verbose:
        _Progress().attach(model)
    with _ultralytics_log_level(logging.INFO if verbose else logging.WARNING):
        model.train(trainer=imbalance_trainer(sampler, seed), **kwargs)

    best = out_dir / "weights" / "best.pt"
    record = out_dir / "imbalance.json"
    return TrainResult(last=last, best=best if best.exists() else None, results_csv=out_dir / "results.csv",
                       effective_args=yaml.safe_load((out_dir / "args.yaml").read_text()),
                       imbalance=json.loads(record.read_text()) if record.exists() else None)


def checkpoint_info(weights: str | Path) -> CheckpointInfo:
    """Training state stored in an Ultralytics checkpoint.

    Ultralytics stores the index of the last completed epoch, or -1 once training has finished.
    """
    from ultralytics.utils.patches import torch_load

    ckpt = torch_load(str(weights), map_location="cpu")
    args = ckpt.get("train_args") or {}
    planned = int(args["epochs"])
    epoch = ckpt.get("epoch", -1)
    return CheckpointInfo(epochs_completed=planned if epoch == -1 else int(epoch) + 1, epochs_planned=planned,
                          date=ckpt.get("date"), ultralytics=ckpt.get("version"), train_args=args,
                          train_results=ckpt.get("train_results"))


def import_checkpoint(weights: str | Path, train_dir: str | Path, train_args: dict, seed: int) -> CheckpointInfo:
    """Make a checkpoint kept from an earlier session the weights of a run (`train_dir/weights/last.pt`).

    The checkpoint must have been trained with `train_args` and `seed`. Its own arguments and training curves
    are written as `args.yaml` / `results.csv`, as Ultralytics would have. Raises if the run already has weights.
    """
    train_dir = Path(train_dir)
    last = train_dir / "weights" / "last.pt"
    if last.exists():
        raise FileExistsError(f"{train_dir} already holds a trained model")
    info = checkpoint_info(weights)
    expected = {**train_args, "seed": seed}
    diff = {k: (v, info.train_args.get(k)) for k, v in expected.items() if info.train_args.get(k) != v}
    if diff:
        raise ValueError(f"{weights} was trained with other arguments (config, checkpoint): {diff}")
    last.parent.mkdir(parents=True)
    shutil.copy2(weights, last)
    (train_dir / "args.yaml").write_text(yaml.safe_dump(info.train_args, sort_keys=False))
    if info.train_results:
        pd.DataFrame(info.train_results).to_csv(train_dir / "results.csv", index=False)
    return info
