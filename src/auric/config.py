"""Experiment configuration.

    cfg = load_config("configs/baseline.yaml", "configs/env/kaggle.yaml", overrides=["train.args.epochs=1"])

What belongs in a config file:
    experiment file   what is computed: split, tiling, model, training, inference, evaluation parameters
    env file          where it runs: paths.data_root, paths.work_dir, runtime.device, runtime.workers
What does not: decisions taken when running (reuse vs. overwrite a run, resume, which samples to show);
those are explicit function arguments set in the notebook.

* Files are deep-merged left to right; `overrides` ("dotted.key=value", value parsed as YAML) apply last.
* A file may inherit from another with `_base_: <path relative to the file>`.
* Relative values under `paths:` resolve against the directory of the file that sets them
  (relative override values resolve against the current working directory). Nothing is searched for:
  required paths must be set explicitly, and they are checked here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

REQUIRED_PATHS = ("data_root", "work_dir", "split_file")
FORMULATIONS = ("tiles", "scenes")
SUBSET_METHODS = ("random", "coverage")      # auric.data.subsets reads it from here


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _resolve_paths(paths: dict | None, anchor: Path) -> None:
    for k, v in (paths or {}).items():
        if isinstance(v, str) and not Path(v).is_absolute():
            paths[k] = (anchor / v).resolve().as_posix()


def _load_file(path: Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _resolve_paths(cfg.get("paths"), path.parent)
    base = cfg.pop("_base_", None)
    return deep_merge(_load_file(path.parent / base), cfg) if base else cfg


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        *parents, leaf = key.strip().split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
        if parents == ["paths"]:
            _resolve_paths(node, Path.cwd())
    return cfg


def validate(cfg: dict) -> None:
    paths = cfg.get("paths") or {}
    missing = [k for k in REQUIRED_PATHS if not paths.get(k)]
    if missing:
        raise ValueError(f"paths not set: {missing} (set them in configs/env/*.yaml or as overrides)")
    root = Path(paths["data_root"])
    if not (root / "classmap.txt").is_file():
        raise FileNotFoundError(f"paths.data_root={root} does not contain classmap.txt")
    if cfg["train"]["args"]["imgsz"] % 32:
        raise ValueError("train.args.imgsz must be a multiple of 32")
    if cfg["tiling"]["overlap"] >= cfg["tiling"]["size"] or cfg["inference"]["overlap"] >= cfg["inference"]["tile"]:
        raise ValueError("tile overlap must be smaller than the tile size")
    if formulation(cfg) not in FORMULATIONS:
        raise ValueError(f"formulation.mode must be one of {FORMULATIONS}, got {formulation(cfg)!r}")
    if not 0 <= cfg["train"]["args"].get("cls_pw", 0.0) <= 1:
        raise ValueError("train.args.cls_pw must be in [0, 1]")
    if (sampler := cfg["train"].get("sampler")) is not None and not 0 <= sampler["alpha"] <= 1:
        raise ValueError("train.sampler.alpha must be in [0, 1]")
    if (sub := cfg.get("subset")) is not None:
        if sub.get("method") not in SUBSET_METHODS:
            raise ValueError(f"subset.method must be one of {SUBSET_METHODS}, got {sub.get('method')!r}")
        if not 0 < sub.get("budget_frac", 0) <= 1:
            raise ValueError("subset.budget_frac must be in (0, 1]")


def formulation(cfg: dict) -> str:
    """How the detector sees the data: "tiles" (training tile cache, tiled inference) or "scenes" (whole images).

    Configs without a `formulation` section are tile runs.
    """
    return (cfg.get("formulation") or {}).get("mode", "tiles")


def load_config(*files: str | Path, overrides: list[str] | None = None) -> dict:
    cfg: dict = {}
    for f in files:
        cfg = deep_merge(cfg, _load_file(Path(f)))
    cfg = apply_overrides(cfg, overrides)
    validate(cfg)
    return cfg


def fingerprint(cfg: dict, sections: tuple[str, ...]) -> str:
    """Hash of the given top-level sections that are present (e.g. those that determine the trained weights).

    A section a config does not have is left out, so adding an optional section to the list does not change the
    fingerprint of the configs (and runs) that predate it.
    """
    core = {k: cfg[k] for k in sections if k in cfg}
    return hashlib.sha1(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:12]


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def read_config(path: str | Path) -> dict:
    """Read a resolved config written by `save_config` (no inheritance or path resolution)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

