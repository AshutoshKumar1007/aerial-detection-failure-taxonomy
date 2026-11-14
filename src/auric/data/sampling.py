"""Class-aware sampling of training samples (tiles or whole scenes).

`counts` is an (N, K) matrix: instances of each class in each training sample, as the detector will see them
(the label files of the run's train list). Plain shuffling draws a sample with probability 1/N, so a class is seen in
proportion to how often it occurs; `class_aware_probs` changes the draw probabilities, not the data:

    1. object-free samples keep their share of the draws (the choice of how many of them to keep is `tiling.negative_frac`);
    2. the remaining draws are spread over the classes, class c with probability q_c proportional to n_c ** alpha
       (n_c = instances of c; alpha 0: every class equally often, alpha 1: as often as it occurs);
    3. inside a class, the draw is uniform over the samples that contain it.

A sample holding several classes is reachable through each of them. The realised instance shares therefore differ from
q_c, which is why `describe_sampling` reports the expected shares instead of assuming them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def class_aware_probs(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Draw probability of each sample (sums to 1) under the class-aware scheme described in the module docstring."""
    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError("counts must be (samples, classes)")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    present = counts > 0
    positive = present.any(axis=1)
    if not positive.any():
        raise ValueError("no sample contains an object")
    instances = counts.sum(axis=0).astype(float)
    q = np.where(instances > 0, instances, 1.0) ** alpha * (instances > 0)
    q /= q.sum()
    samples_per_class = present.sum(axis=0)
    per_class = np.where(samples_per_class > 0, q / np.maximum(samples_per_class, 1), 0.0)
    probs = np.zeros(len(counts))
    background = float((~positive).mean())
    probs[positive] = (1 - background) * (present[positive] * per_class).sum(axis=1)
    if background > 0:
        probs[~positive] = background / int((~positive).sum())
    return probs / probs.sum()


def expected_counts(counts: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Instances of each class in one epoch of N draws with the given probabilities (N = number of samples)."""
    return len(counts) * (np.asarray(probs) @ np.asarray(counts))


def describe_sampling(counts: np.ndarray, probs: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Per class: what the dataset holds and what one epoch of draws is expected to show."""
    counts = np.asarray(counts)
    seen = expected_counts(counts, probs)
    own = counts.sum(axis=0)
    containing = (counts > 0).sum(axis=0)
    draws = len(counts) * (np.asarray(probs)[:, None] * (counts > 0)).sum(axis=0)
    return pd.DataFrame({
        "class": class_names, "instances": own, "samples containing": containing,
        "share % (dataset)": 100 * own / own.sum(), "instances (expected per epoch)": seen,
        "share % (expected per epoch)": 100 * seen / seen.sum(), "draws per epoch": draws, "draws per containing sample": draws / np.maximum(containing, 1)})


def class_loss_weights(counts: np.ndarray, pw: float) -> np.ndarray:
    """Ultralytics' `cls_pw` class weights for (expected) class counts: (1 / n_c) ** pw scaled to mean 1.

    A class without instances gets the mean weight. This is the formula of `DetectionTrainer.compute_class_weights` in the pinned
    Ultralytics (8.4.179), for previews before a run; the run itself records the weights Ultralytics used (`imbalance.json`).
    """
    counts = np.asarray(counts, dtype=float)
    present = counts > 0
    w = (1.0 / np.where(present, counts, 1.0)) ** pw
    w = np.where(present, w, w[present].mean())
    return w / w.mean()
