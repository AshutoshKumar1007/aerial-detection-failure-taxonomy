import numpy as np
import pytest

from auric.evaluation.matching import Matcher
from auric.evaluation.metrics import IOU_THRS, average_precision, bootstrap_aps, class_aps, compare, evaluate
from auric.structures import Annotation, Detections

K = 5
NAMES = [f"c{k}" for k in range(K)]


def _box(x, y, s=20):
    return [x, y, x + s, y + s]


def _synthetic(seed, n_img=12):
    """Jittered GT (some with wrong class) plus random false positives, sizes 6-60 px."""
    rng = np.random.default_rng(seed)
    gts, dets = {}, {}
    for i in range(n_img):
        n = int(rng.integers(0, 40))
        xy = rng.uniform(0, 900, (n, 2))
        gb = np.concatenate([xy, xy + rng.uniform(6, 60, (n, 2))], 1)
        gc = rng.integers(0, K, n)
        keep = rng.random(n) < 0.8
        pb = gb[keep] + rng.normal(0, 3, (keep.sum(), 4))
        pb = np.concatenate([np.minimum(pb[:, :2], pb[:, 2:]), np.maximum(pb[:, :2], pb[:, 2:]) + 1e-3], 1)
        pc = np.where(rng.random(keep.sum()) < 0.8, gc[keep], rng.integers(0, K, keep.sum()))
        nf = int(rng.integers(0, 15))
        fxy = rng.uniform(0, 900, (nf, 2))
        pb = np.concatenate([pb, np.concatenate([fxy, fxy + rng.uniform(6, 60, (nf, 2))], 1)])
        pc = np.concatenate([pc, rng.integers(0, K, nf)])
        gts[str(i)] = Annotation(gb, gc)
        dets[str(i)] = Detections(pb, rng.random(len(pb)), pc)
    return gts, dets


def _pycoco(gts, dets, iou_thrs, area_rng=(0, 1e10)):
    """pycocotools precision -> (T, K) AP."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, res, aid = [], [], [], 1
    for k, key in enumerate(gts):
        images.append({"id": k, "width": 1000, "height": 1000})
        for b, c in zip(gts[key].boxes, gts[key].cls):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append({"id": aid, "image_id": k, "category_id": int(c), "bbox": [b[0], b[1], w, h],
                         "area": w * h, "iscrowd": 0})
            aid += 1
        d = dets[key]
        for b, s, c in zip(d.boxes, d.scores, d.cls):
            res.append({"image_id": k, "category_id": int(c), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                        "score": float(s)})
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in range(K)]}
    coco.createIndex()
    E = COCOeval(coco, coco.loadRes(res), "bbox")
    E.params.iouThrs = np.asarray(iou_thrs, float)
    E.params.maxDets = [10000]
    E.params.areaRng, E.params.areaRngLbl = [list(area_rng)], ["r"]
    E.evaluate()
    E.accumulate()
    prec = E.eval["precision"][:, :, :, 0, 0]          # (T, R, K)
    return np.where((prec > -1).any(1), np.where(prec > -1, prec, 0).mean(1), np.nan)


def test_hand_computed_ap():
    # 2 GT; detections by score: TP, FP, TP -> precision envelope 1 up to recall .5, then 2/3
    tp = np.array([True, False, True])
    assert average_precision(tp, ~tp, 2)[0] == pytest.approx((51 + 50 * 2 / 3) / 101)


def test_perfect_detections():
    gts = {"a": Annotation(np.array([_box(0, 0), _box(50, 50)], float), np.array([0, 1])),
           "b": Annotation(np.array([_box(10, 10)], float), np.array([4]))}
    dets = {k: Detections(v.boxes, np.ones(len(v)), v.cls) for k, v in gts.items()}
    r = evaluate(gts, dets, NAMES, bootstrap=20)
    assert r.summary["mAP50"] == pytest.approx(1.0) and r.summary["mAP50_95"] == pytest.approx(1.0)
    assert np.isnan(r.per_class.AP50[2])                 # classes without GT are excluded, not zero
    assert r.summary["operating_point"]["f1"] == 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ap_matches_pycocotools(seed):
    pytest.importorskip("pycocotools")
    gts, dets = _synthetic(seed)
    m = Matcher(gts, dets, K)
    ours = np.stack([class_aps(m.match(t))[0] for t in IOU_THRS])
    np.testing.assert_allclose(ours, _pycoco(gts, dets, IOU_THRS), atol=1e-6)


@pytest.mark.parametrize("area_rng", [(0, 16 ** 2), (16 ** 2, 32 ** 2), (32 ** 2, np.inf)])
def test_size_range_ap_matches_pycocotools(area_rng):
    pytest.importorskip("pycocotools")
    gts, dets = _synthetic(4, n_img=20)
    ours = class_aps(Matcher(gts, dets, K).match(0.5, area_rng))[0]
    ref = _pycoco(gts, dets, [0.5], (area_rng[0], min(area_rng[1], 1e10)))[0]
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_bootstrap_with_unit_weights_reproduces_point_estimate():
    gts, dets = _synthetic(5)
    m = Matcher(gts, dets, K).match(0.5)
    boot = bootstrap_aps(m, 200, seed=0)
    ap = class_aps(m)[0]
    lo, hi = np.nanpercentile(boot, [0, 100], axis=0)
    assert np.all((lo <= ap + 1e-9) & (ap <= hi + 1e-9))


def test_outcomes_on_a_constructed_scene():
    easy = [_box(700 + 50 * k, 0) for k in range(4)]            # 4 clean class-1 hits fix the threshold at 0.5
    g = Annotation(np.array([_box(0, 0), _box(100, 100), _box(200, 200), _box(300, 300), _box(400, 400), *easy],
                            float), np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]))
    d = Detections(
        np.array([_box(0, 0), _box(1, 1), _box(100, 100), _box(208, 208), _box(600, 600), _box(400, 400), *easy,
                  *[_box(900, 900 + 30 * k) for k in range(6)]], float),
        np.array([0.9, 0.8, 0.85, 0.7, 0.6, 0.0005, 0.5, 0.5, 0.5, 0.5, *[0.001] * 6]),
        np.array([0, 0, 2, 0, 3, 1, 1, 1, 1, 1, *[4] * 6]))
    r = evaluate({"x": g}, {"x": d}, NAMES, bootstrap=0)
    assert r.summary["operating_point"]["conf"] == 0.5
    gt = dict(zip(r.gt_outcomes.gt_idx, r.gt_outcomes.outcome))
    assert [gt[j] for j in range(5)] == ["TP", "FN_class", "FN_loc", "FN_nobox", "FN_lowscore"]
    det = dict(zip(r.det_outcomes.det_idx, r.det_outcomes.outcome))
    assert det == {0: "TP", 1: "dup", 2: "cls", 3: "loc", 4: "bkg", 6: "TP", 7: "TP", 8: "TP", 9: "TP"}
    assert r.confusion.loc["c0", "c0"] == 1 and r.confusion.loc["c0", "c2"] == 1
    assert r.confusion.loc["c0", "missed"] == 2 and r.confusion.loc["background", "c0"] == 2


def test_paired_comparison():
    gts, dets = _synthetic(3, n_img=30)
    same = compare(gts, dets, dets, NAMES, bootstrap=50, seed=0)
    assert (same["diff"].abs() < 1e-12).all() and (same["lo"] == 0).all() and (same["hi"] == 0).all()
    assert same.loc["mAP50", "A"] == pytest.approx(evaluate(gts, dets, NAMES, bootstrap=0).summary["mAP50"])

    # B = A without the lowest-scoring half of the detections: AP cannot rise, and the interval is paired
    worse = {i: d.select(d.scores >= np.median(d.scores)) if len(d) else d for i, d in dets.items()}
    cmp = compare(gts, dets, worse, NAMES, bootstrap=100, seed=0)
    assert cmp.loc["mAP50", "diff"] < 0 and cmp.loc["mAP50", "hi"] <= 0.02
    assert cmp.loc["mAP50", "p_b_better"] < 0.1
    # reversing the roles negates the point estimates
    back = compare(gts, worse, dets, NAMES, bootstrap=100, seed=0)
    assert back.loc["mAP50", "diff"] == pytest.approx(-cmp.loc["mAP50", "diff"])
