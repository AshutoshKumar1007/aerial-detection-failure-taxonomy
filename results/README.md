# Saved predictions of the Kaggle runs

The analysis notebooks for 5.2–5.4 (`notebooks/03_resist_learning.ipynb`, `notebooks/04_subset_efficiency.ipynb`) do
not run any model. They re-evaluate the predictions that the training runs saved on Kaggle, read from:

```
results/runs/<arm>/predictions/dev.csv[.gz]
results/runs/<arm>/predictions/val.csv[.gz]
```

One row per detection: `image_id, cls, score, x1, y1, x2, y2` (source-image pixels, conf ≥ 0.001), as written by
`auric.structures.save_detections`.

| Arm | What it is | Used by | Present |
|---|---|---|---|
| `D1_s0` | final model D1 (single-label inference) | 03, 04 | yes |
| `A1_s0` | final model A1 (single-label inference) | 03, 04 | yes |
| `D1_s1` | D1 with training seed 1 (noise check) | 03, 04 | yes |
| `SR50_k0`, `SR25_k0`, `SR12_k0` | D1 recipe on random 50 / 25 / 12.5% subsets | 04 | yes |
| `SC25`, `SC12` | D1 recipe on coverage-selected 25 / 12.5% subsets | 04 | yes |

Either `.csv` or `.csv.gz` is read. A missing arm is reported by the notebook and left out of its tables.

`reproduce.py` writes its own predictions and metrics to `outputs/` (see the top-level README). It does not use this folder.
