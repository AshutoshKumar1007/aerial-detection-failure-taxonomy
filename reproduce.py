"""Inference and evaluation of the final models: validation images -> predictions -> reported metrics.

    python reproduce.py                                   # defaults from config.yaml: D1, multi-label, val
    python reproduce.py --model A1                        # the other final model
    python reproduce.py --model D1 --variant single --splits val dev
    python reproduce.py --runtime kaggle --dataset-root /kaggle/input/<dataset>/CV_Assignment_AAL
    python reproduce.py --from-predictions outputs/D1_multi_label/val/predictions.csv.gz --splits val   # metrics only

Every input comes from config.yaml or the command line; nothing is searched for. Outputs go to
<output_dir>/<model>_<variant>/<split>/: predictions.csv.gz, metrics.json (summary, per-class and per-size AP),
the diagnostic tables (confusion, error outcomes, PR curves, per image) and run.json (settings, checkpoint hash, versions).
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

SPLITS = ("val", "dev")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "config.yaml"), help="reproduction config (default: config.yaml next to this script)")
    p.add_argument("--runtime", help="local | kaggle (default: config `runtime`)")
    p.add_argument("--model", help="D1 | A1 (default: config `model`)")
    p.add_argument("--variant", help="multi_label | single (default: config `variant`)")
    p.add_argument("--splits", nargs="+", choices=SPLITS, help="val and/or dev (default: config `splits`)")
    p.add_argument("--dataset-root", help="folder holding classmap.txt, train/, val/ (overrides the runtime's dataset_root)")
    p.add_argument("--weights", help="checkpoint file (overrides <weights_dir>/<models.<model>.weights>)")
    p.add_argument("--output-dir", help="overrides the runtime's output_dir")
    p.add_argument("--device", help="CUDA index or cpu (overrides the runtime's device)")
    p.add_argument("--from-predictions", help="evaluate this predictions file instead of running the detector (one split)")
    p.add_argument("--limit", type=int, help="only the first N images of each split (wiring check; metrics are not comparable)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing output folder (default: stop)")
    return p.parse_args()


def fail(msg: str) -> None:
    sys.exit(f"\nERROR: {msg}\n")


def resolve(path: str | Path, base: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def check_dataset(root: Path, splits: list[str]) -> None:
    folders = sorted({"val" if s == "val" else "train" for s in splits})      # dev is a held-out part of train/
    needed = ["classmap.txt"] + [f"{f}/{d}" for f in folders for d in ("images", "labels")]
    missing = [n for n in needed if not (root / n).exists()]
    if missing:
        fail(f"dataset not found or incomplete at {root}\n"
             f"  missing: {', '.join(missing)}\n"
             f"  Put the provided dataset folder there, so that {root / 'classmap.txt'} and {root / 'val' / 'images'} exist\n"
             f"  (see Dataset/README.md), or point to it with --dataset-root or `dataset_root` in config.yaml.")


def versions() -> dict:
    import numpy
    import torch
    import ultralytics
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "numpy": numpy.__version__, "gpu": gpu, "platform": platform.platform()}


def headline(rep, reported: float | None) -> dict:
    s = rep.summary
    ci = s["mAP50_ci95"] or [float("nan")] * 2
    row = {"images": s["n_images"], "objects": s["n_gt"], "detections": s["n_det"], "mAP50": round(s["mAP50"], 4),
           "mAP50 CI95": f"[{ci[0]:.3f}, {ci[1]:.3f}]", "AP50 class-agnostic": round(s["AP50_class_agnostic"], 4),
           "mAP50-95": round(s["mAP50_95"], 4), "mAP75": round(s["mAP75"], 4)}
    if reported is not None:
        row["reported mAP50"] = reported
    return row


def main() -> None:
    args = parse_args()
    cfg_file = Path(args.config).resolve()
    if not cfg_file.is_file():
        fail(f"config not found: {cfg_file}")
    rc = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    base = cfg_file.parent

    runtime = args.runtime or rc["runtime"]
    if runtime not in rc["runtimes"]:
        fail(f"runtime {runtime!r} is not defined in {cfg_file} (have: {', '.join(rc['runtimes'])})")
    rt = rc["runtimes"][runtime]
    model = args.model or rc["model"]
    if model not in rc["models"]:
        fail(f"model {model!r} is not defined in {cfg_file} (have: {', '.join(rc['models'])})")
    mc = rc["models"][model]
    variant = args.variant or rc["variant"]
    if variant not in rc["variants"]:
        fail(f"variant {variant!r} is not defined in {cfg_file} (have: {', '.join(rc['variants'])})")
    splits = list(dict.fromkeys(args.splits or rc["splits"]))
    if any(s not in SPLITS for s in splits):
        fail(f"splits must be among {SPLITS}, got {splits}")

    dataset_root = resolve(args.dataset_root or rt["dataset_root"], Path.cwd() if args.dataset_root else base)
    output_dir = resolve(args.output_dir or rt["output_dir"], Path.cwd() if args.output_dir else base)
    device = str(args.device if args.device is not None else rt["device"])
    device = int(device) if device.isdigit() else device
    check_dataset(dataset_root, splits)

    weights = None
    if args.from_predictions:
        pred_file = resolve(args.from_predictions, Path.cwd())
        if not pred_file.is_file():
            fail(f"predictions file not found: {pred_file}")
        if len(splits) != 1:
            fail("--from-predictions evaluates one split; pass --splits val or --splits dev")
    else:
        weights = resolve(args.weights, Path.cwd()) if args.weights else resolve(rt["weights_dir"], base) / mc["weights"]
        if not weights.is_file():
            fail(f"checkpoint not found: {weights}\n"
                 f"  Put the {model} checkpoint there (see weights/README.md), or pass --weights PATH.")

    # heavy imports only after the inputs are known to exist
    import torch

    from auric.config import load_config
    from auric.data.dataset import index_split, load_annotations, read_class_names
    from auric.data.splits import ensure_split, subset
    from auric.evaluation import evaluate
    from auric.model.predict import TiledPredictor
    from auric.runs import sha256
    from auric.structures import load_detections, save_detections

    if weights is not None and device != "cpu" and not torch.cuda.is_available():
        fail(f"device {device!r} requested but CUDA is not available; pass --device cpu (much slower) or use a GPU machine")

    exp_file = ROOT / "src" / "auric" / "configs" / mc["experiment"]
    cfg = load_config(exp_file, overrides=[f"paths.data_root={dataset_root.as_posix()}", f"paths.work_dir={output_dir.as_posix()}"])
    ic = {**cfg["inference"], **rc["variants"][variant]}
    if device == "cpu":
        ic["fp16"] = False           # half precision is a GPU setting
    ec = cfg["evaluation"]

    names = read_class_names(dataset_root)
    eval_sets = {}
    for part in splits:
        folder = "val" if part == "val" else "train"
        recs = index_split(dataset_root, folder)
        part_anns = load_annotations(recs, len(names))
        if part == "dev":
            split, _ = ensure_split(cfg["paths"]["split_file"], part_anns, names, **cfg["split"])   # the shipped split file
            recs = subset(recs, split["dev"])
        eval_sets[part] = (recs, part_anns)
    if args.limit:
        eval_sets = {s: (recs[:args.limit], a) for s, (recs, a) in eval_sets.items()}

    weights_sha = None
    if weights is not None:
        weights_sha = sha256(weights)
        if mc.get("sha256") and mc["sha256"] != weights_sha:
            print(f"WARNING: {weights.name} has sha256 {weights_sha[:12]}..., not the reported checkpoint "
                  f"{mc['sha256'][:12]}...; the metrics will not match REPORT.md")

    tag = f"{model}_{variant}" + ("_from_predictions" if args.from_predictions else "") + (f"_limit{args.limit}" if args.limit else "")
    run_dir = output_dir / tag
    print(f"model {model} ({mc['experiment']}) | variant {variant} | splits {splits} | runtime {runtime}\n"
          f"dataset {dataset_root}\n" + (f"weights {weights} (sha256 {weights_sha[:12]}...)\n" if weights else
                                          f"predictions {pred_file}\n") + f"output  {run_dir}")

    predictor = None if args.from_predictions else TiledPredictor(weights, ic, device)
    env = versions()
    results = {}
    for part, (recs, part_anns) in eval_sets.items():
        out = run_dir / part
        if out.exists() and any(out.iterdir()):
            if not args.overwrite:
                fail(f"{out} already exists; pass --overwrite to replace it")
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        gts = {r.image_id: part_anns[r.image_id] for r in recs}

        t0 = time.perf_counter()
        if predictor is not None:
            dets = predictor.predict(recs, desc=f"{model}/{variant} {part}")
            save_detections(dets, out / "predictions.csv.gz")
        else:
            dets = load_detections(pred_file)
            unknown = set(dets) - set(gts)
            if unknown:
                fail(f"{pred_file} has detections for {len(unknown)} images that are not in the {part} split")
        t_pred = time.perf_counter() - t0
        rep = evaluate(gts, dets, names, ec["size_bins"], ec["bootstrap"], ec["seed"])
        rep.summary["settings"] = {"model": model, "experiment": mc["experiment"], "variant": variant, "inference": ic,
                                   "evaluation": ec, "weights": str(weights) if weights else None, "weights_sha256": weights_sha,
                                   "predictions_file": str(pred_file) if args.from_predictions else None}
        rep.save(out)
        (out / "run.json").write_text(json.dumps({**rep.summary["settings"], "split": part, "images": [r.image_id for r in recs],
                                                  "limit": args.limit, "predict_seconds": round(t_pred, 1),
                                                  "environment": env}, indent=1, default=str))
        results[part] = rep

    import pandas as pd
    pd.set_option("display.width", 200)
    reported = (mc.get("reported") or {}).get(variant, {})
    print("\n" + pd.DataFrame({p: headline(r, None if args.limit else reported.get(p)) for p, r in results.items()}).to_string())
    for part, rep in results.items():
        pc = rep.per_class.set_index("class")[["n_gt", "AP50", "AP50_lo", "AP50_hi", "recall_max"]]
        print(f"\n{part}: per-class AP50 (95% image-bootstrap interval)\n" + pc.round(3).to_string())
    print(f"\nsaved under {run_dir}")


if __name__ == "__main__":
    main()
