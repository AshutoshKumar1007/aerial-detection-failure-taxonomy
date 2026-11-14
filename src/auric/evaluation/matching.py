"""COCO (pycocotools) matching of detections to ground truth, vectorised across images.

Per image and class, detections are visited in descending score order; each takes the unmatched,
non-ignored GT with the highest IoU >= threshold, otherwise an unmatched *ignored* GT (the detection is
then ignored too). For size-restricted evaluation, GT outside the size range is ignored, and unmatched
detections outside the range are ignored. No crowd regions exist in this dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..boxes import box_area, iou_matrix
from ..structures import Annotation, Detections


@dataclass
class Matching:
    """Matching of all detections of a split at one IoU threshold and size range (concatenated arrays)."""
    det_img: np.ndarray      # (D,) image index
    det_idx: np.ndarray      # (D,) index into the image's detections
    det_cls: np.ndarray      # (D,)
    det_score: np.ndarray    # (D,)
    det_tp: np.ndarray       # (D,) bool
    det_ignore: np.ndarray   # (D,) bool
    det_gt: np.ndarray       # (D,) index into the image's GT, -1 if unmatched
    gt_count: np.ndarray     # (I, K) non-ignored GT per image and class


class Matcher:
    """Pre-computes per-(image, class) IoU tables once; `match` is then cheap for any threshold/size range."""

    def __init__(self, gts: dict[str, Annotation], dets: dict[str, Detections], num_classes: int,
                 class_agnostic: bool = False):
        unknown = set(dets) - set(gts)
        if unknown:
            raise KeyError(f"detections for images without ground truth: {sorted(unknown)[:5]}")
        self.image_ids = list(gts)
        self.K = 1 if class_agnostic else num_classes
        self.groups = []    # (img, cls, det_idx sorted by score, gt_idx, ious, det_area, gt_area)
        empty = Detections.empty()
        for i, image_id in enumerate(self.image_ids):
            g, d = gts[image_id], dets.get(image_id, empty)
            gcls = np.zeros(len(g), np.int64) if class_agnostic else g.cls
            dcls = np.zeros(len(d), np.int64) if class_agnostic else d.cls
            for c in np.unique(np.concatenate([gcls, dcls])):
                di = np.flatnonzero(dcls == c)
                di = di[np.argsort(-d.scores[di], kind="stable")]
                gi = np.flatnonzero(gcls == c)
                self.groups.append((i, int(c), di, gi, iou_matrix(d.boxes[di], g.boxes[gi]),
                                    box_area(d.boxes[di]), box_area(g.boxes[gi])))
        self.dets = dets
        self.class_agnostic = class_agnostic

    def match(self, iou_thr: float, area_range: tuple[float, float] | None = None) -> Matching:
        thr = min(iou_thr, 1 - 1e-10)
        img, idx, cls, score, tp, ign, gt = [], [], [], [], [], [], []
        gt_count = np.zeros((len(self.image_ids), self.K), np.int64)
        for i, c, di, gi, ious, d_area, g_area in self.groups:
            g_ign = np.zeros(len(gi), bool) if area_range is None else \
                (g_area < area_range[0]) | (g_area > area_range[1])
            gt_count[i, c] += int((~g_ign).sum())
            m_gt = np.full(len(di), -1)
            d_ign = np.zeros(len(di), bool)
            taken = np.zeros(len(gi), bool)
            for r in range(len(di)):
                ok = ~taken & (ious[r] >= thr)
                for allowed in (ok & ~g_ign, ok & g_ign):     # non-ignored GT first, then ignored GT
                    if allowed.any():
                        j = int(np.argmax(np.where(allowed, ious[r], -1.0)))
                        taken[j], m_gt[r], d_ign[r] = True, gi[j], g_ign[j]
                        break
            if area_range is not None:
                d_ign |= (m_gt < 0) & ((d_area < area_range[0]) | (d_area > area_range[1]))
            d = self.dets.get(self.image_ids[i])
            img.append(np.full(len(di), i))
            idx.append(di)
            cls.append(np.full(len(di), c))
            score.append(d.scores[di] if len(di) else np.zeros(0))
            tp.append((m_gt >= 0) & ~d_ign)
            ign.append(d_ign)
            gt.append(m_gt)
        cat = (lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt))
        return Matching(cat(img, np.int64), cat(idx, np.int64), cat(cls, np.int64), cat(score, np.float64),
                        cat(tp, bool), cat(ign, bool), cat(gt, np.int64), gt_count)
