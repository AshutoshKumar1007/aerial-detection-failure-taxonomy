"""Aerial-domain initialisation: transfer a DOTA-pretrained backbone/neck into a detect model.

Why. Every run so far initialises from COCO. The open question (declared ablation) is whether the
COCO representation -- not the data, recipe, or architecture -- is what caps the detector. DOTA is an overhead
dataset with large-vehicle / small-vehicle classes, so its features are far closer to this task than COCO's.

Integrity. DOTA is a different dataset from xView (different providers/scenes), so a DOTA checkpoint has not
seen the val images: it is an *allowed, declared* initialisation, not the barred "contaminated upper bound"
that an xView checkpoint would be. Only the learned weights transfer; DOTA's labels are never used here.

Mechanism. Ultralytics publishes DOTA only as OBB (oriented-box) models, which cannot be loaded as a plain
detector. So we build a detect model from the architecture and transfer the matching layers (backbone + neck);
the detect head is left fresh and is re-sized to the task's class count at train time (single_cls -> nc=1 for
T1). The result is a single detect checkpoint the existing `train_yolo` consumes unchanged.
"""
from __future__ import annotations

from pathlib import Path

from .train import fetch_pretrained


def build_dota_transfer_init(pretrained_dir: str | Path, arch: str = "yolo11s.yaml",
                             obb_asset: str = "yolo11s-obb.pt", out_name: str = "yolo11s_dota.pt") -> Path:
    """Write `<pretrained_dir>/<out_name>`: a detect checkpoint with DOTA backbone/neck and a fresh head.

    `arch` scale must match `obb_asset` scale (s<->s) so backbone/neck shapes align. Idempotent: returns the
    existing file if already built. Needs internet once to fetch the OBB asset. Reference it from a config as
    `model.weights: <out_name>` -- `fetch_pretrained` then finds this local file by name (no asset download).
    """
    from ultralytics import YOLO

    out = Path(pretrained_dir) / out_name
    if out.is_file():
        return out
    obb = fetch_pretrained(obb_asset, pretrained_dir)   # DOTAv1 OBB checkpoint
    model = YOLO(arch)                                   # detect architecture, random init
    model.load(str(obb))                                # transfers matching layers (backbone + neck); OBB head skipped
    #   expect the log to say "Transferred <most>/<all> items"; the un-transferred items are the head.
    model.save(str(out))                                # a detect checkpoint carrying DOTA features
    return out
