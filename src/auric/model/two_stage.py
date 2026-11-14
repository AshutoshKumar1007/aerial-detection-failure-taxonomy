"""Two-stage assembly: class-agnostic detector (stage 1) x crop classifier (stage 2) -> final Detections.

The whole point of the pipeline. Stage 1 is a single-class detector (the T1 config: `single_cls`), run at
inference with `merge_agnostic` so NMS never splits one truck's confidence across classes -- it answers only
"is there a truck here, and where". Stage 2 re-labels each surviving box. Because the output is an ordinary
`Detections` dict, the existing `evaluate` (COCO-101 mAP50 + image bootstrap) scores it with no changes.

What it measures, for free (assignment 5.1): run `evaluate` on the stage-1 detections with every class set to a
single id -> that is the detection ceiling (class-agnostic AP50). Run it on the assembled output -> that is the
full mAP50. The gap between them is exactly the classification cost, now attributable to a real classifier
rather than the fused YOLO head.

Score fusion. For ranking (all mAP cares about) the final score is det_score ** alpha * p(class) ** (1 - alpha).
alpha=1 ignores the classifier's confidence (rank by detector only); alpha=0 ranks by class prob only; the
default geometric blend (0.5) let both speak. `emit_topk` > 1 emits the k most likely classes per box as
separate detections (the MX finding: emitting the 2nd-best class was worth +8-14% dev mAP because the metric
can only credit a class it sees) -- stage-2 makes that hedge principled instead of a post-hoc NMS trick.
"""
from __future__ import annotations

import numpy as np

from ..data.dataset import ImageRecord
from ..structures import Annotation, Detections
from .classifier import CropClassifier


def assemble(stage1: dict[str, Detections], records: list[ImageRecord], clf: CropClassifier,
             *, alpha: float = 0.5, emit_topk: int = 1, batch: int = 256, workers: int = 4) -> dict[str, Detections]:
    """stage-1 agnostic detections + trained classifier -> final multi-class Detections."""
    boxes_by_image = {r.image_id: stage1.get(r.image_id, Detections.empty()).boxes for r in records}
    proba = clf.predict_proba(records, boxes_by_image, batch=batch, workers=workers)
    out = {}
    for r in records:
        iid = r.image_id
        d = stage1.get(iid, Detections.empty())
        if len(d) == 0:
            out[iid] = Detections.empty(); continue
        p = proba[iid]                                             # (n, K)
        obj = d.scores                                             # stage-1 objectness
        k = min(emit_topk, p.shape[1])
        top = np.argsort(-p, axis=1)[:, :k]                        # (n, k) best classes per box
        bx, sc, cl = [], [], []
        for j in range(k):
            cls_j = top[:, j]
            pj = p[np.arange(len(p)), cls_j]
            score = (obj ** alpha) * (pj ** (1.0 - alpha))
            bx.append(d.boxes); sc.append(score); cl.append(cls_j)
        out[iid] = Detections(np.concatenate(bx), np.concatenate(sc), np.concatenate(cl).astype(np.int64))
    return out


def gt_as_detections(gts: dict[str, Annotation]) -> dict[str, Detections]:
    """Perfect locations (assignment 5.1): every ground-truth box as a stage-1 detection with objectness 1.

    `assemble` then lets the classifier alone decide label and ranking, so the evaluator's mAP50 is the score of the
    five-way problem once localisation, missed objects and background detections are taken out."""
    return {i: Detections(a.boxes.astype(np.float64), np.ones(len(a)), a.cls.astype(np.int64)) for i, a in gts.items()}


def as_single_class(dets: dict[str, Detections]) -> dict[str, Detections]:
    """Collapse detection classes to id 0, keeping boxes/scores: the detector side of the ceiling eval."""
    return {i: Detections(d.boxes, d.scores, np.zeros(len(d), np.int64)) for i, d in dets.items()}


def gt_single_class(gts: dict[str, Annotation]) -> dict[str, Annotation]:
    """Collapse GT classes to id 0: the ground-truth side of the ceiling eval (Annotation has no scores)."""
    return {i: Annotation(a.boxes, np.zeros(len(a), np.int64)) for i, a in gts.items()}


def detection_ceiling_names(class_names: list[str]) -> list[str]:
    """Single-element class list to pass to `evaluate` for the detection-only ceiling."""
    return ["truck"]
