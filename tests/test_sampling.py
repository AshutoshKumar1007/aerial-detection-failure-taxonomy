import numpy as np
import pytest

from auric.data.sampling import class_aware_probs, describe_sampling, expected_counts

# 6 samples x 3 classes: two background samples, class 0 dominant, class 2 rare and only together with class 1
COUNTS = np.array([[8, 0, 0], [5, 0, 0], [6, 1, 0], [0, 3, 1], [0, 0, 0], [0, 0, 0]])


def test_probabilities_are_a_distribution_and_keep_the_background_share():
    for alpha in (0.0, 0.5, 1.0):
        p = class_aware_probs(COUNTS, alpha)
        assert p.sum() == pytest.approx(1.0) and (p >= 0).all()
        assert p[4:].sum() == pytest.approx(2 / 6)       # object-free samples keep the 2 of 6 draws they had


def test_alpha_zero_is_uniform_over_classes_and_alpha_one_follows_the_instances():
    names = ["a", "b", "c"]
    flat = describe_sampling(COUNTS, class_aware_probs(COUNTS, 0.0), names)
    plain = describe_sampling(COUNTS, np.full(6, 1 / 6), names)
    assert flat["draws per epoch"].iloc[2] > plain["draws per epoch"].iloc[2]       # the rare class is drawn more often
    # class draw probabilities q_c: alpha 0 -> 1/3 each; the draws containing class c add up to at least q_c
    assert flat["draws per epoch"].iloc[2] >= 6 * (1 / 3) * (4 / 6)
    one = class_aware_probs(COUNTS, 1.0)
    assert one[0] > one[3]                                # frequent-class samples are favoured when alpha = 1


def test_rare_class_share_rises_monotonically_as_alpha_falls():
    share = []
    for alpha in (1.0, 0.5, 0.0):
        c = expected_counts(COUNTS, class_aware_probs(COUNTS, alpha))
        share.append(c[2] / c.sum())
    assert share[0] < share[1] < share[2]


def test_expected_counts_of_uniform_draws_are_the_dataset_counts():
    assert expected_counts(COUNTS, np.full(6, 1 / 6)) == pytest.approx(COUNTS.sum(axis=0))


def test_bad_inputs():
    with pytest.raises(ValueError):
        class_aware_probs(COUNTS, 1.5)
    with pytest.raises(ValueError):
        class_aware_probs(np.zeros((3, 2), int), 0.5)
