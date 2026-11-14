"""Run directories: one per run id (`<experiment.name>_s<experiment.seed>`) under `<work_dir>/runs/`.

    <run>/config.yaml            resolved config + fingerprint, written when the run is created
    <run>/record.json            one entry per stage: time, software, hardware, git commit + stage facts
                                 (seed, weights and their hashes, effective Ultralytics args, ...)
    <run>/data/                  train/val lists (tiles, or whole scenes) + data.yaml for Ultralytics
                                 (with a `subset` section: only the tiles/scenes of the selected train_core images)
    <run>/train/                 Ultralytics output (args.yaml, weights/{last,best}.pt, results.csv, plots,
                                 imbalance.json: class counts, class weights, sampler summary)
    <run>/predictions/<split>.csv.gz
    <run>/eval/<split>/          metrics.json + tables

A run is identified by the config sections that determine its weights (RUN_SECTIONS). An existing run is
never replaced silently: `create_run` takes an explicit decision for that case.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from .config import fingerprint, read_config, save_config

RUN_SECTIONS = ("experiment", "split", "tiling", "model", "train", "formulation", "subset")
PACKAGES = ("ultralytics", "torch", "torchvision", "numpy", "pandas", "opencv-python-headless", "pyyaml")


@dataclass(frozen=True)
class Run:
    dir: Path

    @property
    def config_file(self) -> Path:
        return self.dir / "config.yaml"

    @property
    def record_file(self) -> Path:
        return self.dir / "record.json"

    @property
    def data_dir(self) -> Path:
        return self.dir / "data"

    @property
    def train_dir(self) -> Path:
        return self.dir / "train"

    @property
    def weights(self) -> Path:
        """The evaluated checkpoint: last epoch of the fixed schedule (no selection on any metric)."""
        return self.train_dir / "weights" / "last.pt"

    def predictions(self, split: str) -> Path:
        return self.dir / "predictions" / f"{split}.csv.gz"

    def eval_dir(self, split: str) -> Path:
        return self.dir / "eval" / split


def run_id(cfg: dict) -> str:
    return f"{cfg['experiment']['name']}_s{cfg['experiment']['seed']}"


def create_run(run_dir: str | Path, cfg: dict, if_exists: Literal["error", "resume", "overwrite"] = "error") -> Run:
    """Create the run directory for `cfg`, or deal with an existing one as explicitly requested.

    if_exists:
        error      raise (default: nothing is reused or replaced by accident)
        resume     continue the existing run; its stored config must have the same fingerprint
        overwrite  delete the existing run directory and start again
    """
    run = Run(Path(run_dir))
    fp = fingerprint(cfg, RUN_SECTIONS)
    if run.dir.exists() and any(run.dir.iterdir()):
        if if_exists == "error":
            raise FileExistsError(f"{run.dir} already exists. Choose if_exists='resume' or 'overwrite', "
                                  "or change experiment.name / experiment.seed")
        if if_exists == "resume":
            stored = read_config(run.config_file)["fingerprint"] if run.config_file.exists() else None
            if stored != fp:
                raise RuntimeError(f"{run.dir} was created with a different config ({', '.join(RUN_SECTIONS)})")
            return run
        if if_exists != "overwrite":
            raise ValueError(f"if_exists must be 'error', 'resume' or 'overwrite', got {if_exists!r}")
        shutil.rmtree(run.dir)
    run.dir.mkdir(parents=True, exist_ok=True)
    save_config({**cfg, "fingerprint": fp}, run.config_file)
    return run


def open_run(run_dir: str | Path, cfg: dict | None = None) -> Run:
    """An existing run (for inference/evaluation); with `cfg`, check that it belongs to that config."""
    run = Run(Path(run_dir))
    if not run.config_file.exists():
        raise FileNotFoundError(f"no run at {run.dir}")
    if cfg is not None and read_config(run.config_file)["fingerprint"] != fingerprint(cfg, RUN_SECTIONS):
        raise RuntimeError(f"{run.dir} was created with a different config ({', '.join(RUN_SECTIONS)})")
    return run


def _version(dist: str) -> str | None:
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _git_commit(cwd: Path) -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if rev.returncode == 0 else None


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    """Software and hardware of the current process."""
    import torch

    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "packages": {p: _version(p) for p in PACKAGES},
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "git_commit": _git_commit(Path(__file__).resolve().parent),
    }


def log_stage(run: Run, stage: str, info: dict | None = None) -> dict:
    """Append a stage entry (time, environment, `info`) to the run record and return it."""
    entry = {"stage": stage, "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "environment": environment(), **(info or {})}
    log = json.loads(run.record_file.read_text()) if run.record_file.exists() else []
    run.record_file.write_text(json.dumps(log + [entry], indent=1, default=str))
    return entry


def read_record(run: Run) -> list[dict]:
    return json.loads(run.record_file.read_text()) if run.record_file.exists() else []
