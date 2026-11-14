import numpy as np
import pytest
import torch

from auric.evaluation import evaluate
from auric.evaluation.classification import classification_report, matched_pairs
from auric.model.classifier import CropClassifier, CropSpec, build_backbone
from auric.model.two_stage import assemble, gt_as_detections
from auric.structures import Annotation

NAMES = ["a", "b", "c"]


def _gts():
    return {"i0": Annotation(np.array([[0, 0, 20, 20], [40, 40, 60, 60]], float), np.array([0, 1])),
            "i1": Annotation(np.array([[10, 10, 30, 30]], float), np.array([2]))}


class _Oracle:
    """Stands in for a trained classifier: puts mass `p` on the true class of each box (looked up by box)."""

    def __init__(self, gts, p):
        self.gts, self.p = gts, p

    def predict_proba(self, records, boxes_by_image, batch=256, workers=0):
        out = {}
        for iid, bx in boxes_by_image.items():
            prob = np.full((len(bx), 3), (1 - self.p) / 2)
            prob[np.arange(len(bx)), self.gts[iid].cls] = self.p
            out[iid] = prob
        return out


def test_perfect_boxes_with_a_perfect_classifier_score_one_and_a_wrong_one_scores_less():
    gts = _gts()
    final = assemble(gt_as_detections(gts), [type("R", (), {"image_id": i}) for i in gts], _Oracle(gts, 0.9), emit_topk=1)
    assert evaluate(gts, final, NAMES, bootstrap=0).summary["mAP50"] == pytest.approx(1.0)

    class Wrong(_Oracle):
        def predict_proba(self, records, boxes_by_image, **kw):
            return {i: np.roll(p, 1, axis=1) for i, p in super().predict_proba(records, boxes_by_image).items()}

    wrong = assemble(gt_as_detections(gts), [type("R", (), {"image_id": i}) for i in gts], Wrong(gts, 0.9), emit_topk=1)
    assert evaluate(gts, wrong, NAMES, bootstrap=0).summary["mAP50"] < 0.1
    # emitting the second-best class cannot lower the score: the metric only credits a class it sees
    hedged = assemble(gt_as_detections(gts), [type("R", (), {"image_id": i}) for i in gts], Wrong(gts, 0.9), emit_topk=3)
    assert evaluate(gts, hedged, NAMES, bootstrap=0).summary["mAP50"] > evaluate(gts, wrong, NAMES, bootstrap=0).summary["mAP50"]


def test_classification_report_counts_and_chance_levels():
    y = np.array([0, 0, 0, 0, 1, 1, 2])
    proba = np.eye(3)[[0, 0, 0, 1, 1, 0, 2]]
    r = classification_report(y, proba, NAMES)
    assert r["n"] == 7 and r["accuracy"] == pytest.approx(5 / 7)
    assert r["balanced_accuracy"] == pytest.approx(np.mean([3 / 4, 1 / 2, 1.0]))
    assert r["chance_majority"] == pytest.approx(4 / 7) and r["chance_balanced"] == pytest.approx(1 / 3)
    assert r["confusion"].loc["a", "b"] == 1 and r["confusion"].loc["b", "a"] == 1 and r["confusion"].values.sum() == 7
    assert r["per_class"].loc["a", "precision"] == pytest.approx(3 / 4)


def test_matched_pairs_take_the_best_scoring_detection_per_object_and_skip_background():
    gt = _gts()["i0"]
    boxes = np.array([[1, 1, 21, 21], [100, 100, 120, 120], [40, 40, 60, 60], [0, 0, 20, 21], [50, 50, 90, 90]], float)
    scores = np.array([0.4, 0.9, 0.5, 0.8, 0.9])
    # object 0 is found by detections 0 and 3 (3 scores higher); object 1 by detection 2; 1 and 4 touch nothing
    assert matched_pairs(boxes, scores, gt).tolist() == [[3, 0], [2, 1]]
    assert matched_pairs(np.zeros((0, 4)), np.zeros(0), gt).shape == (0, 2)


def test_classifier_save_and_load_roundtrip(tmp_path):
    spec = CropSpec(context=2.0, min_px=16, out_px=32)
    with torch.random.fork_rng():                    # random init must not shift the global RNG other tests rely on
        torch.manual_seed(0)
        model = build_backbone("convnext_tiny", 3, pretrained=False)
        x = torch.randn(2, 3, 32, 32)
    clf = CropClassifier(model, spec, "cpu", NAMES, [{"epoch": 1, "loss": 1.0, "train_acc": 0.5, "lr": 1e-3}])
    clf.save(tmp_path / "c.pt")
    back = CropClassifier.load(tmp_path / "c.pt", device="cpu")
    assert back.spec == spec and back.class_names == NAMES and back.history == clf.history and back.backbone == "convnext_tiny"
    with torch.no_grad():
        assert torch.allclose(clf.model.eval()(x), back.model(x), atol=1e-5)
