"""Full-image inference by tiling: slice -> batched detector -> shift to image coords -> merge.

The tile grid is the same function used to build the training cache. Stitching is separate from the
detector (`stitch`), so it can be checked with ground truth standing in for predictions
(`stitch_ground_truth`: the best AP that tiling and merging allow).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from ..boxes import class_nms
from ..data.dataset import ImageRecord, read_image
from ..data.tiling import Window, crop_annotation, tile_windows
from ..structures import Annotation, Detections


def interior_edge_mask(boxes: np.ndarray, win: Window, W: int, H: int, margin: float) -> np.ndarray:
    """True for tile-local boxes within `margin` px of a tile border that lies inside the image.

    With overlap > object size, such boxes are truncated views of objects seen whole in a neighbouring tile.
    """
    x0, y0, x1, y1 = win
    m = np.zeros(len(boxes), bool)
    if margin <= 0:
        return m
    if x0 > 0:
        m |= boxes[:, 0] < margin
    if y0 > 0:
        m |= boxes[:, 1] < margin
    if x1 < W:
        m |= boxes[:, 2] > (x1 - x0) - margin
    if y1 < H:
        m |= boxes[:, 3] > (y1 - y0) - margin
    return m


def stitch(tile_dets: list[tuple[Window, Detections]], W: int, H: int, merge_iou: float,
           edge_margin: float = 0, max_det: int | None = None, agnostic: bool = False) -> Detections:
    """Tile-local detections -> image detections (edge filter, shift, NMS across tiles, top-k).

    The merge is class-aware (a box survives next to an overlapping box of another class) unless `agnostic`.
    """
    parts = []
    for win, d in tile_dets:
        if len(d) == 0:
            continue
        d = d.select(~interior_edge_mask(d.boxes, win, W, H, edge_margin))
        parts.append(Detections(d.boxes + np.array([win[0], win[1], win[0], win[1]], float), d.scores, d.cls))
    dets = Detections.concat(parts)
    keep = class_nms(dets.boxes, dets.scores, dets.cls, merge_iou, agnostic) if len(tile_dets) > 1 else \
        np.argsort(-dets.scores, kind="stable")
    return dets.select(keep[:max_det] if max_det else keep)


def stitch_ground_truth(records: list[ImageRecord], annotations: dict[str, Annotation], tile: int, overlap: int,
                        min_visibility: float, merge_iou: float, edge_margin: float = 0,
                        max_det: int | None = None) -> dict[str, Detections]:
    """Ground truth pushed through tiling and stitching as if it were detector output.

    Each tile "detects" its objects exactly as they appear in the training labels (clipped boxes with
    visibility >= min_visibility), with one consistent score per object.
    """
    out = {}
    for r in records:
        g = annotations[r.image_id]
        tiles = []
        for win in tile_windows(r.width, r.height, tile, overlap):
            b, c, idx, _ = crop_annotation(g, win, min_visibility)
            tiles.append((win, Detections(b, 1.0 - idx / (10 * max(len(g), 1)), c)))
        out[r.image_id] = stitch(tiles, r.width, r.height, merge_iou, edge_margin, max_det)
    return out


def multi_label_predictor():
    """An Ultralytics detection predictor whose NMS keeps every class above `conf` for a box, not only the best.

    `model.predict` uses best-class-only NMS, whereas the validator Ultralytics trains with (and COCO-style
    evaluation) emits every class above the threshold. When two classes are plausible for one object the
    second hypothesis is a true positive for that class's AP and a low-scoring false positive otherwise.
    The body mirrors `DetectionPredictor.postprocess` of the pinned Ultralytics (8.4.179) with `multi_label=True`.
    """
    from ultralytics.models.yolo.detect import DetectionPredictor
    from ultralytics.utils import nms, ops

    class MultiLabelPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            preds = nms.non_max_suppression(
                preds, self.args.conf, kwargs.pop("iou", self.args.iou), self.args.classes, self.args.agnostic_nms,
                max_det=self.args.max_det, nc=0 if self.args.task == "detect" else len(self.model.names),
                multi_label=True, end2end=getattr(self.model, "end2end", False), rotated=self.args.task == "obb")
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
            return self.construct_results(preds, img, orig_imgs, **kwargs)

    return MultiLabelPredictor


class TiledPredictor:
    """Ultralytics YOLO weights -> full-image `Detections`.

    Optional `icfg` keys (all default to the behaviour of the YOLO11 runs):
        multi_label    False: each box keeps its best class; True: every class above `conf` (see `multi_label_predictor`)
        agnostic_nms   False: NMS inside a tile suppresses per class; True: across classes
        merge_agnostic False: the merge across tiles is class-aware (`stitch`); True: class-agnostic
        head           YOLO26 only: "one2many" decodes the dense head and applies NMS, "one2one" the NMS-free head
                       (top-`max_det_tile` of box x class scores, `iou` and `multi_label` do not apply);
                       absent: the Ultralytics default, which for YOLO26 is the one-to-many head with NMS
    Ultralytics keeps at most 30,000 candidate boxes per tile before NMS (best scores first), which can only
    matter with multi-label output at a very low `conf`.
    """

    def __init__(self, weights, icfg: dict, device=0):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.cfg = icfg
        head = icfg.get("head")
        if head not in (None, "one2many", "one2one"):
            raise ValueError(f"inference.head must be 'one2many' or 'one2one', got {head!r}")
        if head == "one2one":
            if getattr(self.model.model.model[-1], "one2one_cv2", None) is None:
                raise ValueError(f"{weights} has no one-to-one head")
            if icfg.get("multi_label", False):
                raise ValueError("multi_label applies to NMS output; the one-to-one head has no NMS")
        self.predictor = multi_label_predictor() if icfg.get("multi_label", False) else None
        self.kwargs = dict(imgsz=icfg["imgsz"], conf=icfg["conf"], iou=icfg["iou"], max_det=icfg["max_det_tile"],
                           agnostic_nms=icfg.get("agnostic_nms", False), quantize=16 if icfg["fp16"] else None,
                           device=device, verbose=False)
        if head == "one2one":
            self.kwargs["nms"] = False       # Ultralytics: nms=False selects the NMS-free head, otherwise external NMS

    def predict_tiles(self, tiles: list[np.ndarray]) -> list[Detections]:
        out = []
        for r in self.model.predict(tiles, predictor=self.predictor, **self.kwargs):
            b = r.boxes
            out.append(Detections(b.xyxy.cpu().numpy().astype(np.float64), b.conf.cpu().numpy().astype(np.float64),
                                  b.cls.cpu().numpy().astype(np.int64)))
        return out

    def predict_image(self, img: np.ndarray) -> Detections:
        c = self.cfg
        H, W = img.shape[:2]
        wins = tile_windows(W, H, c["tile"], c["overlap"])
        tile_dets = []
        for k in range(0, len(wins), c["batch"]):
            chunk = wins[k:k + c["batch"]]
            tile_dets += zip(chunk, self.predict_tiles([img[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk]))
        return stitch(tile_dets, W, H, c["merge_iou"], c["edge_margin"], c["max_det_image"],
                      c.get("merge_agnostic", False))

    def predict(self, records: list[ImageRecord], desc: str = "images") -> dict[str, Detections]:
        """All images; the next image is decoded in a background thread while the current one runs."""
        out = {}
        with ThreadPoolExecutor(max_workers=1) as io:
            pending = io.submit(read_image, records[0].image_path) if records else None
            for n, rec in enumerate(tqdm(records, desc=desc, unit="image")):
                img = pending.result()
                if n + 1 < len(records):
                    pending = io.submit(read_image, records[n + 1].image_path)
                out[rec.image_id] = self.predict_image(img)
        return out
