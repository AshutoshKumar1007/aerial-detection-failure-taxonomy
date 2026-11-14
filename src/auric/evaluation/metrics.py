"""Full-image detection metrics.

Primary metric: mAP@0.5 with COCO 101-point interpolation (identical to pycocotools), averaged over the
classes that have ground truth. Also reported:
    mAP@[.5:.95], mAP@.75                 localisation quality
    class-agnostic AP@.5                   detection ability with class identity removed
    mAP@.5 with Ultralytics interpolation  comparability with numbers printed by Ultralytics
    AP@.5 per size bucket                  COCO area-range semantics (other sizes ignored, not penalised)
    image-bootstrap 95% CIs                objects within an image are correlated, so images are resampled
    operating-point diagnostics            see diagnostics.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..boxes import size_buckets
from ..structures import Annotation, Detections
from . import diagnostics
from .matching import Matcher, Matching

REC_THRS = np.linspace(0, 1, 101)
IOU_THRS = np.round(np.arange(0.5, 0.951, 0.05), 2)
TABLES = ("per_class", "per_size", "pr_curves", "confusion", "fn_breakdown", "fp_breakdown", "per_image",
          "gt_outcomes", "det_outcomes")


def average_precision(tp: np.ndarray, fp: np.ndarray, n_gt: float, method: str = "coco101"):
    """AP from per-detection TP/FP weights sorted by descending score.

    Returns (ap, precision at the 101 recall points). `tp`/`fp` may be weights (bootstrap).
    """
    if n_gt <= 0:
        return float("nan"), np.full(len(REC_THRS), np.nan)
    tpc, fpc = np.cumsum(tp, dtype=float), np.cumsum(fp, dtype=float)
    rc = tpc / n_gt
    pr = tpc / (tpc + fpc + np.spacing(1))
    if method == "coco101":
        env = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
        inds = np.searchsorted(rc, REC_THRS, side="left")
        q = np.zeros(len(REC_THRS))
        valid = inds < len(env)
        q[valid] = env[inds[valid]]
        return float(q.mean()), q
    if method == "ultralytics":
        mrec = np.concatenate([[0.0], rc, [1.0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate([[1.0], pr, [0.0]]))))
        q = np.interp(REC_THRS, mrec, mpre)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        return float(trapezoid(q, REC_THRS)), q
    raise ValueError(f"unknown AP method {method!r}")


def _sorted_class(m: Matching, c: int):
    sel = (m.det_cls == c) & ~m.det_ignore
    order = np.argsort(-m.det_score[sel], kind="stable")
    return m.det_img[sel][order], m.det_tp[sel][order]


def class_aps(m: Matching, method: str = "coco101") -> tuple[np.ndarray, np.ndarray]:
    """(K,) AP and (K, 101) precision curves."""
    K = m.gt_count.shape[1]
    ap, curves = np.full(K, np.nan), np.full((K, len(REC_THRS)), np.nan)
    for c in range(K):
        _, tp = _sorted_class(m, c)
        ap[c], curves[c] = average_precision(tp, ~tp, m.gt_count[:, c].sum(), method)
    return ap, curves


def bootstrap_aps(m: Matching, n: int, seed: int) -> np.ndarray:
    """(n, K) AP@IoU(m) over image resamples; a resampled image enters with its multiplicity as weight."""
    I, K = m.gt_count.shape
    rng = np.random.default_rng(seed)
    per_class = [_sorted_class(m, c) for c in range(K)]
    out = np.full((n, K), np.nan)
    for b in range(n):
        w = np.bincount(rng.integers(0, I, I), minlength=I).astype(float)
        for c, (img, tp) in enumerate(per_class):
            wd = w[img]
            keep = wd > 0
            out[b, c] = average_precision((wd * tp)[keep], (wd * ~tp)[keep], float(w @ m.gt_count[:, c]))[0]
    return out


@dataclass
class EvaluationReport:
    summary: dict
    per_class: pd.DataFrame
    per_size: pd.DataFrame
    pr_curves: pd.DataFrame
    confusion: pd.DataFrame
    fn_breakdown: pd.DataFrame
    fp_breakdown: pd.DataFrame
    per_image: pd.DataFrame
    gt_outcomes: pd.DataFrame
    det_outcomes: pd.DataFrame

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(
            {"summary": self.summary, "per_class": self.per_class.to_dict("records"),
             "per_size": self.per_size.to_dict("records")}, indent=1, default=_json_default))
        for name in TABLES:
            getattr(self, name).to_csv(out_dir / f"{name}.csv", index=name == "confusion", float_format="%.6g")

    @classmethod
    def load(cls, out_dir: str | Path) -> "EvaluationReport":
        out_dir = Path(out_dir)
        summary = json.loads((out_dir / "metrics.json").read_text())["summary"]
        tables = {n: pd.read_csv(out_dir / f"{n}.csv", index_col=0 if n == "confusion" else None,
                                 dtype={"image_id": str}) for n in TABLES}
        return cls(summary, **tables)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 6)
    raise TypeError(type(o))


def _nanmean(a) -> float:
    a = np.asarray(a, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def evaluate(gts: dict[str, Annotation], dets: dict[str, Detections], class_names: list[str],
             size_bins=(16, 32), bootstrap: int = 1000, seed: int = 0) -> EvaluationReport:
    K = len(class_names)
    matcher = Matcher(gts, dets, K)
    by_iou = {t: matcher.match(t) for t in IOU_THRS}
    m50 = by_iou[0.5]

    ap = np.stack([class_aps(by_iou[t])[0] for t in IOU_THRS])          # (T, K)
    ap50, curves = ap[0], class_aps(m50)[1]
    ap50_ultra = class_aps(m50, "ultralytics")[0]
    agn = Matcher(gts, dets, K, class_agnostic=True).match(0.5)
    ap50_agnostic = class_aps(agn)[0][0]

    ap50_ci, maps = np.full((2, K), np.nan), None
    if bootstrap:
        boot = bootstrap_aps(m50, bootstrap, seed)
        has_gt = np.isfinite(boot).any(0)
        ap50_ci[:, has_gt] = np.nanpercentile(boot[:, has_gt], [2.5, 97.5], axis=0)
        maps = np.nanmean(boot[:, has_gt], 1)

    # size buckets (COCO area-range semantics)
    size_rows = []
    for label, lo, hi in size_buckets(list(size_bins)):
        ms = matcher.match(0.5, (lo ** 2, hi ** 2))
        ap_s = class_aps(ms)[0]
        size_rows.append({"bucket": label, "n_gt": int(ms.gt_count.sum()), "AP50": _nanmean(ap_s),
                          **{f"AP50 {n}": a for n, a in zip(class_names, ap_s)}})

    # operating point and diagnostics
    op = diagnostics.operating_point(m50)
    diag = diagnostics.analyse(gts, dets, m50, op["conf"], class_names, list(size_bins))

    gt_count = m50.gt_count.sum(0)
    n_img = (m50.gt_count > 0).sum(0)
    rec_max = np.array([m50.det_tp[m50.det_cls == c].sum() for c in range(K)]) / np.maximum(gt_count, 1)
    per_class = pd.DataFrame({
        "class": class_names, "n_images": n_img, "n_gt": gt_count,
        "n_det": [int((m50.det_cls == c).sum()) for c in range(K)],
        "AP50": ap50,
        "AP50_lo": ap50_ci[0], "AP50_hi": ap50_ci[1],
        "AP75": ap[list(IOU_THRS).index(0.75)], "AP50_95": ap.mean(0),   # nan only for classes without GT
        "AP50_ultralytics": ap50_ultra, "recall_max": rec_max,
    }).merge(diag["per_class_op"], on="class", how="left")
    per_size = pd.DataFrame(size_rows).merge(diag["recall_by_size"], on="bucket", how="left")

    summary = {
        "n_images": len(gts), "n_gt": int(gt_count.sum()), "n_det": int(len(m50.det_score)),
        "mAP50": _nanmean(ap50),
        "mAP50_ci95": [float(x) for x in np.nanpercentile(maps, [2.5, 97.5])] if bootstrap else None,
        "mAP50_boot_std": float(np.nanstd(maps)) if bootstrap else None,
        "mAP50_95": _nanmean(ap.mean(0)), "mAP75": _nanmean(ap[list(IOU_THRS).index(0.75)]),
        "AP50_class_agnostic": float(ap50_agnostic), "mAP50_ultralytics": _nanmean(ap50_ultra),
        "AP50_by_size": {r["bucket"]: r["AP50"] for r in size_rows},
        "operating_point": op, "bootstrap": bootstrap, "seed": seed,
    }
    pr = pd.DataFrame([{"class": n, "recall": r, "precision": p}
                       for n, q in zip(class_names, curves) for r, p in zip(REC_THRS, q)])
    return EvaluationReport(summary, per_class, per_size, pr, diag["confusion"], diag["fn_breakdown"],
                            diag["fp_breakdown"], diag["per_image"], diag["gt_outcomes"], diag["det_outcomes"])


def compare(gts: dict[str, Annotation], dets_a: dict[str, Detections], dets_b: dict[str, Detections],
            class_names: list[str], bootstrap: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Paired comparison of two detection sets on the same images: AP50 of B minus A with a paired bootstrap CI.

    Both sets are evaluated on the same image resamples (same seed, same image order), so the interval is that
    of the *difference* and is much narrower than the two intervals of `evaluate` taken separately.
    Rows: every class with ground truth, `mAP50` (mean over those classes) and `AP50 class-agnostic`.
    Columns: A, B, diff (B - A), lo / hi (95% interval of the difference), p_b_better (share of resamples with diff > 0).
    """
    K = len(class_names)
    series = {}
    for label, agnostic in (("classes", False), ("agnostic", True)):
        ma, mb = (Matcher(gts, d, K, class_agnostic=agnostic).match(0.5) for d in (dets_a, dets_b))
        aa, ab = class_aps(ma)[0], class_aps(mb)[0]
        ba, bb = bootstrap_aps(ma, bootstrap, seed), bootstrap_aps(mb, bootstrap, seed)
        if agnostic:
            series["AP50 class-agnostic"] = (aa[0], ab[0], bb[:, 0] - ba[:, 0])
            continue
        has = np.isfinite(aa)
        series.update({class_names[c]: (aa[c], ab[c], bb[:, c] - ba[:, c]) for c in np.flatnonzero(has)})
        series["mAP50"] = (_nanmean(aa), _nanmean(ab), np.nanmean(bb[:, has], 1) - np.nanmean(ba[:, has], 1))
    rows = []
    for name in [*(n for n in series if n not in ("mAP50", "AP50 class-agnostic")), "mAP50", "AP50 class-agnostic"]:
        a, b, d = series[name]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        rows.append({"metric": name, "A": a, "B": b, "diff": b - a, "lo": lo, "hi": hi,
                     "p_b_better": float(np.nanmean(d > 0))})
    return pd.DataFrame(rows).set_index("metric")
