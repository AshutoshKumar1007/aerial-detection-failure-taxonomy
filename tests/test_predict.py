import numpy as np
import pytest

from auric.model.predict import TiledPredictor

pytest.importorskip("ultralytics")

ICFG = dict(tile=256, overlap=32, imgsz=256, conf=0.001, iou=0.6, merge_iou=0.6, edge_margin=0, max_det_tile=1000,
            max_det_image=5000, batch=2, fp16=False)


def _tile_detections(multi_label):
    import torch

    # an untrained detector whose class biases give every class a different, mid-range score on every box
    p = TiledPredictor("yolo11n.yaml", {**ICFG, "multi_label": multi_label}, device="cpu")
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for branch in p.model.model.model[-1].cv3:
            branch[-1].bias.copy_(torch.randn(branch[-1].bias.shape, generator=g) - 2.0)
    img = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    return p.predict_tiles([img])[0]


def test_multi_label_keeps_secondary_classes():
    best, multi = _tile_detections(False), _tile_detections(True)
    assert len(multi) > len(best)                      # secondary class hypotheses are emitted
    assert len(np.unique(multi.cls)) >= len(np.unique(best.cls))
    assert multi.scores.max() == pytest.approx(best.scores.max(), abs=1e-6)      # the top candidate survives both


def _yolo26(head, **extra):
    import torch

    p = TiledPredictor("yolo26n.yaml", {**ICFG, "multi_label": False, "head": head, **extra}, device="cpu")
    g = torch.Generator().manual_seed(0)
    detect = p.model.model.model[-1]
    with torch.no_grad():
        for branches in (detect.cv3, detect.one2one_cv3):
            for branch in branches:
                branch[-1].bias.copy_(torch.randn(branch[-1].bias.shape, generator=g) - 2.0)
    return p


def test_yolo26_heads_are_selected_by_the_config():
    img = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    many, one = _yolo26("one2many").predict_tiles([img])[0], _yolo26("one2one", max_det_tile=50).predict_tiles([img])[0]
    assert len(many) > 0 and 0 < len(one) <= 50
    assert len(many) != len(one)                       # different heads, different decoding


def test_tile_nms_can_be_class_agnostic():
    img = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    aware = _yolo26("one2many").predict_tiles([img])[0]
    agnostic = _yolo26("one2many", agnostic_nms=True).predict_tiles([img])[0]
    assert 0 < len(agnostic) < len(aware)


def test_head_choice_is_checked():
    with pytest.raises(ValueError, match="no one-to-one head"):
        TiledPredictor("yolo11n.yaml", {**ICFG, "head": "one2one"}, device="cpu")
    with pytest.raises(ValueError, match="multi_label"):
        TiledPredictor("yolo26n.yaml", {**ICFG, "head": "one2one", "multi_label": True}, device="cpu")
    with pytest.raises(ValueError, match="inference.head"):
        TiledPredictor("yolo26n.yaml", {**ICFG, "head": "both"}, device="cpu")


def test_merge_across_tiles_is_class_aware_unless_agnostic():
    from auric.model.predict import stitch
    from auric.structures import Detections

    # the same object seen in the overlap of two tiles, called class 0 by one and class 1 by the other
    tiles = [((0, 0, 100, 100), Detections(np.array([[60., 10, 80, 30]]), np.array([0.9]), np.array([0]))),
             ((50, 0, 150, 100), Detections(np.array([[10., 10, 30, 30]]), np.array([0.8]), np.array([1])))]
    assert len(stitch(tiles, 150, 100, merge_iou=0.6)) == 2
    kept = stitch(tiles, 150, 100, merge_iou=0.6, agnostic=True)
    assert len(kept) == 1 and kept.cls[0] == 0


def test_whole_scene_inference_is_one_window_without_a_merge():
    from auric.data.tiling import tile_windows

    assert tile_windows(300, 200, 8192, 0) == [(0, 0, 300, 200)]
    p = _yolo26("one2many", tile=8192, overlap=0, imgsz=320, batch=1)
    img = np.random.default_rng(1).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    whole, tile = p.predict_image(img), p.predict_tiles([img])[0]
    assert len(whole) == len(tile) > 0 and np.allclose(np.sort(whole.scores), np.sort(tile.scores))
