import numpy as np
import pytest

from auric.data.subsets import cell_counts, describe_subset, select_subset
from auric.structures import Annotation

NAMES = ["a", "b", "c"]
BINS = [16, 32]


def _scene(n, cls, side):
    boxes = np.tile([0.0, 0.0, side, side], (n, 1)) + np.arange(n)[:, None] * [50, 0, 50, 0]
    return Annotation(boxes, np.full(n, cls))


def _annotations():
    """40 scenes: 30 hold 5 small class-0 trucks, 8 hold 3 class-1 trucks, 2 hold one large class-2 truck; plus one empty scene."""
    anns = {f"a{i}": _scene(5, 0, 12.0) for i in range(30)}
    anns.update({f"b{i}": _scene(3, 1, 24.0) for i in range(8)})
    anns.update({f"c{i}": _scene(1, 2, 40.0) for i in range(2)})
    anns["empty"] = Annotation(np.zeros((0, 4)), np.zeros(0, np.int64))
    return anns


def test_cells_are_class_by_size_bucket():
    a = _annotations()
    cells = cell_counts(a, ["a0", "b0", "c0"], 3, BINS)
    assert cells.shape == (3, 9)
    assert cells[0, 0 * 3 + 0] == 5 and cells[1, 1 * 3 + 1] == 3 and cells[2, 2 * 3 + 2] == 1 and cells.sum() == 9


@pytest.mark.parametrize("method", ["random", "coverage"])
def test_selection_is_deterministic_inside_the_budget_and_skips_empty_scenes(method):
    a = _annotations()
    ids = list(a)
    r1 = select_subset(a, ids, NAMES, BINS, method, 0.4, seed=3)
    assert r1 == select_subset(a, ids, NAMES, BINS, method, 0.4, seed=3)
    assert r1["instances"] <= r1["budget"] and r1["instances"] >= 0.9 * r1["budget"]
    assert "empty" not in r1["ids"] and set(r1["ids"]) <= set(ids)
    assert sum(len(a[i]) for i in r1["ids"]) == r1["instances"]


def test_random_depends_on_the_seed_and_coverage_does_not_need_one_to_cover_the_rare_cells():
    a = _annotations()
    ids = list(a)
    assert select_subset(a, ids, NAMES, BINS, "random", 0.3, seed=0)["ids"] != select_subset(a, ids, NAMES, BINS, "random", 0.3, seed=1)["ids"]
    cov = select_subset(a, ids, NAMES, BINS, "coverage", 0.3, min_cell=3)
    d = describe_subset(a, ids, cov["ids"], NAMES, BINS)["classes"]
    assert (d["instances (subset)"] > 0).all()                       # every class is present, including the 2 rare class-2 scenes
    # random subsets of the same budget miss class 2 more often than coverage does
    missed = sum(describe_subset(a, ids, select_subset(a, ids, NAMES, BINS, "random", 0.3, seed=s)["ids"], NAMES, BINS)
                 ["classes"]["instances (subset)"].iloc[2] == 0 for s in range(30))
    assert missed >= 5


def test_full_budget_keeps_every_labelled_image():
    a = _annotations()
    for method in ("random", "coverage"):
        r = select_subset(a, list(a), NAMES, BINS, method, 1.0)
        assert len(r["ids"]) == len(a) - 1 and r["instances"] == r["budget"]


def test_describe_subset_reports_what_is_kept():
    a = _annotations()
    ids = list(a)
    r = select_subset(a, ids, NAMES, BINS, "coverage", 0.3, min_cell=3)
    d = describe_subset(a, ids, r["ids"], NAMES, BINS)
    assert d["classes"]["instances (all)"].tolist() == [150, 24, 2]
    assert d["classes"]["instances (subset)"].sum() == r["instances"]
    assert d["images"].loc["subset", "images"] == len(r["ids"]) and d["images"].loc["all", "images"] == len(ids)


def test_bad_arguments():
    a = _annotations()
    with pytest.raises(ValueError, match="method"):
        select_subset(a, list(a), NAMES, BINS, "hard", 0.5)
    with pytest.raises(ValueError, match="budget_frac"):
        select_subset(a, list(a), NAMES, BINS, "random", 0.0)


def test_efficiency_arithmetic():
    from auric.evaluation.efficiency import gain_for_extra_labels, log2_slope, recovery_threshold

    r = recovery_threshold([0.10, 0.09, 0.11], 0.9)
    assert r["reference"] == pytest.approx(0.10) and r["threshold"] == pytest.approx(0.09) and (r["min"], r["max"]) == (0.09, 0.11)
    fit = log2_slope([100, 200, 400, 800], [0.10, 0.14, 0.18, 0.22])
    assert fit["slope"] == pytest.approx(0.04) and fit["se"] == pytest.approx(0.0, abs=1e-9)
    assert gain_for_extra_labels(0.04, 6525, 500) == pytest.approx(0.04 * np.log2(7025 / 6525))
    with pytest.raises(ValueError):
        log2_slope([100], [0.1])
    with pytest.raises(ValueError):
        recovery_threshold([])
