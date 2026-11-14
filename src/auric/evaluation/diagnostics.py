"""Error diagnostics at one operating point (IoU 0.5, one global confidence threshold).

The threshold maximises micro-F1 over all classes. At that threshold every ground-truth object and
every kept detection receives exactly one outcome (TIDE-style definitions, background IoU 0.1):

  ground truth                                        detections
    TP           matched by a same-class detection      TP    matched to a GT (COCO matching)
    FN_class     a detection of another class has        dup   IoU >= .5 with a same-class GT already taken
                 IoU >= .5 (classification error)         cls   IoU >= .5 with a GT of another class
    FN_loc       a same-class detection has .1<=IoU<.5    loc   same class, .1 <= IoU < .5
    FN_lowscore  matched only by detections below conf    both  other class, .1 <= IoU < .5
    FN_nobox     no same-class detection at any score     bkg   IoU < .1 with every GT

The confusion matrix uses class-agnostic matching of the kept detections (IoU >= .5), so it shows which
class the detector assigns to objects it has localised, plus missed objects and background detections.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..boxes import iou_matrix, size_bucket_labels, size_buckets
from ..structures import Annotation, Detections
from .matching import Matching

FG_IOU, BG_IOU = 0.5, 0.1
GT_OUTCOMES = ["TP", "FN_class", "FN_loc", "FN_lowscore", "FN_nobox"]
DET_OUTCOMES = ["TP", "dup", "cls", "loc", "both", "bkg"]


def operating_point(m: Matching) -> dict:
    """Confidence threshold (detections with score >= conf are kept) maximising micro-F1."""
    keep = ~m.det_ignore
    order = np.argsort(-m.det_score[keep], kind="stable")
    s, tp = m.det_score[keep][order], m.det_tp[keep][order]
    n_gt = int(m.gt_count.sum())
    if len(s) == 0 or n_gt == 0:
        return {"conf": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": n_gt}
    tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
    p, r = tpc / (tpc + fpc), tpc / n_gt
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    last_of_score = np.r_[s[1:] != s[:-1], True]          # a threshold keeps all detections with equal score
    i = int(np.argmax(np.where(last_of_score, f1, -1)))
    return {"conf": float(s[i]), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
            "tp": int(tpc[i]), "fp": int(fpc[i]), "fn": int(n_gt - tpc[i])}


def _greedy_agnostic(iou: np.ndarray) -> np.ndarray:
    """Rows (detections) already in descending score order -> matched column per row or -1."""
    match = np.full(iou.shape[0], -1)
    taken = np.zeros(iou.shape[1], bool)
    for r in range(iou.shape[0]):
        cand = np.where(taken, -1.0, iou[r])
        if len(cand) and cand.max() >= FG_IOU:
            j = int(np.argmax(cand))
            taken[j], match[r] = True, j
    return match


def analyse(gts: dict[str, Annotation], dets: dict[str, Detections], m50: Matching, conf: float,
            class_names: list[str], size_bins: list[float]) -> dict[str, pd.DataFrame]:
    K = len(class_names)
    ids = list(gts)
    buckets = size_buckets(size_bins)
    confusion = np.zeros((K + 1, K + 1), np.int64)        # rows: GT class | background, cols: pred class | missed
    gt_rows, det_rows, img_rows = [], [], []
    empty = Detections.empty()

    for i, image_id in enumerate(ids):
        g, d = gts[image_id], dets.get(image_id, empty)
        sel = m50.det_img == i
        det_tp = np.zeros(len(d), bool)
        det_tp[m50.det_idx[sel]] = m50.det_tp[sel]
        gt_score = np.full(len(g), np.nan)                 # score of the class-aware match at any confidence
        matched = sel & m50.det_tp
        gt_score[m50.det_gt[matched]] = m50.det_score[matched]

        kept = np.flatnonzero(d.scores >= conf)
        kept = kept[np.argsort(-d.scores[kept], kind="stable")]
        iou = iou_matrix(d.boxes[kept], g.boxes)           # (Dk, G)
        same = d.cls[kept][:, None] == g.cls[None, :]

        # ground-truth outcomes
        gt_out = np.empty(len(g), object)
        confused_as = np.full(len(g), -1)
        for j in range(len(g)):
            col, s_col = iou[:, j], same[:, j]
            if gt_score[j] >= conf:
                gt_out[j] = "TP"
            elif ((col >= FG_IOU) & ~s_col).any():
                gt_out[j] = "FN_class"
                confused_as[j] = d.cls[kept][np.argmax(np.where(~s_col, col, -1))]
            elif ((col >= BG_IOU) & s_col).any():
                gt_out[j] = "FN_loc"
            elif np.isfinite(gt_score[j]):
                gt_out[j] = "FN_lowscore"
            else:
                gt_out[j] = "FN_nobox"

        # detection outcomes
        iou_s = np.where(same, iou, 0).max(1) if len(g) else np.zeros(len(kept))
        iou_o = np.where(~same, iou, 0).max(1) if len(g) else np.zeros(len(kept))
        det_out = np.select([det_tp[kept], iou_s >= FG_IOU, iou_o >= FG_IOU, iou_s >= BG_IOU, iou_o >= BG_IOU],
                            DET_OUTCOMES[:5], default="bkg")

        # class-agnostic confusion
        match = _greedy_agnostic(iou)
        for r, j in enumerate(match):
            confusion[g.cls[j] if j >= 0 else K, d.cls[kept[r]]] += 1
        missed = np.setdiff1d(np.arange(len(g)), match[match >= 0])
        np.add.at(confusion, (g.cls[missed], K), 1)

        wh = g.boxes[:, 2:] - g.boxes[:, :2]
        side = np.sqrt(wh.prod(1))
        gt_rows.append(pd.DataFrame({
            "image_id": image_id, "gt_idx": np.arange(len(g)), "cls": g.cls, "x1": g.boxes[:, 0], "y1": g.boxes[:, 1],
            "x2": g.boxes[:, 2], "y2": g.boxes[:, 3], "w": wh[:, 0], "h": wh[:, 1], "sqrt_area": side,
            "size": size_bucket_labels(side, size_bins),
            "outcome": gt_out, "match_score": gt_score, "confused_as": confused_as}))
        det_rows.append(pd.DataFrame({
            "image_id": image_id, "det_idx": kept, "cls": d.cls[kept], "score": d.scores[kept],
            "x1": d.boxes[kept, 0], "y1": d.boxes[kept, 1], "x2": d.boxes[kept, 2], "y2": d.boxes[kept, 3],
            "outcome": det_out, "iou_same": iou_s, "iou_other": iou_o}))
        n_tp = int((gt_out == "TP").sum())
        img_rows.append({"image_id": image_id, "n_gt": len(g), "n_det": len(kept), "tp": n_tp,
                         "fp": int(len(kept) - det_tp[kept].sum()), "fn": len(g) - n_tp})

    gt_df = pd.concat(gt_rows, ignore_index=True)
    det_df = pd.concat(det_rows, ignore_index=True)
    for df in (gt_df, det_df):
        df.insert(df.columns.get_loc("cls") + 1, "class", np.asarray(class_names, object)[df["cls"].to_numpy()])
    per_image = pd.DataFrame(img_rows)
    per_image["precision"] = per_image.tp / per_image.n_det.where(per_image.n_det > 0)
    per_image["recall"] = per_image.tp / per_image.n_gt.where(per_image.n_gt > 0)

    fn = pd.crosstab(gt_df["class"], gt_df["outcome"]).reindex(index=class_names, columns=GT_OUTCOMES, fill_value=0)
    fp = pd.crosstab(det_df["class"], det_df["outcome"]).reindex(index=class_names, columns=DET_OUTCOMES, fill_value=0)
    fn.loc["all"], fp.loc["all"] = fn.sum(), fp.sum()
    fn.insert(0, "n_gt", fn.sum(axis=1))
    fn["recall"] = fn.TP / fn.n_gt.where(fn.n_gt > 0)
    fp.insert(0, "n_det", fp.sum(axis=1))
    fp["precision"] = fp.TP / fp.n_det.where(fp.n_det > 0)

    per_class_op = pd.DataFrame({"class": class_names, "P_op": fp.precision[class_names].to_numpy(),
                                 "R_op": fn.recall[class_names].to_numpy()})
    per_class_op["F1_op"] = 2 * per_class_op.P_op * per_class_op.R_op / (per_class_op.P_op + per_class_op.R_op)

    is_tp = gt_df.outcome == "TP"
    rec_size = is_tp.groupby(gt_df["size"]).mean().rename("R_op").to_frame()
    rec_size = rec_size.join(is_tp.groupby([gt_df["size"], gt_df["class"]]).mean().unstack()
                             .reindex(columns=class_names).add_prefix("R_op "))
    rec_size = rec_size.reindex([b[0] for b in buckets]).rename_axis("bucket").reset_index()

    labels = list(class_names)
    return {
        "confusion": pd.DataFrame(confusion, index=labels + ["background"], columns=labels + ["missed"]),
        "fn_breakdown": fn.rename_axis("class").reset_index(),
        "fp_breakdown": fp.rename_axis("class").reset_index(),
        "per_class_op": per_class_op, "recall_by_size": rec_size, "per_image": per_image,
        "gt_outcomes": gt_df, "det_outcomes": det_df,
    }
