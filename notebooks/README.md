# Notebooks

Two kinds of notebook are included:
* **Protocol notebooks** (this folder, outputs cleared): the code to re-run. Each has one setup cell with
  `RUNTIME = "local" | "kaggle"` and the input paths.
* **Executed runs** (`executed/`): the Kaggle notebooks exactly as they ran, with all outputs. They are the evidence for the
  numbers in `REPORT.md`. Their setup cells show the Kaggle dataset paths of that run.

## Protocol notebooks

| Notebook | Purpose | Assignment | Inputs |
|---|---|---|---|
| `01_train_and_evaluate.ipynb` | One experiment end to end: dataset audit, dev split, tile cache, training, full-image evaluation with diagnostics and galleries. `EXPERIMENT` selects `src/auric/configs/<EXPERIMENT>.yaml` (`D1`, `A1`, `baseline`, `T1`, `DT1`, `Y1`…`Y6`, `SR`, `SC`); `OVERRIDES` holds the run's dotted overrides | 2, 3, 4 | dataset; GPU |
| `02_two_stage_and_perfect_locations.ipynb` | TS1 two-stage pipeline, then the 5.1 perfect-location experiment with the same classifier | 4, 5.1 | dataset; T1 weights (`work/runs/T1_s0/...`, from 01 with `EXPERIMENT="T1"`) |
| `03_resist_learning.ipynb` | Objects no full-data model solves: populations, detector typing, predict-then-test | 5.2 | dataset; `results/runs/{D1_s0,D1_s1,A1_s0}/predictions/` |
| `04_subset_efficiency.ipynb` | Subset composition, 90% recovery, coverage vs random, learning curve and +500 labels | 5.3, 5.4 | dataset; `results/runs/<arm>/predictions/` |

Paths: with `RUNTIME = "local"` the notebooks read `src/auric/configs/env/local.yaml` (dataset at
`Dataset/CV_Assignment_AAL`, work folder `work/`), and they must be started from this folder. With
`RUNTIME = "kaggle"`, set the `SRC_INPUT` / `SRC_DIR` path in the setup cell to the Kaggle dataset that holds `src/auric`,
and `data_root` in `src/auric/configs/env/kaggle.yaml` to the attached dataset.

The final validation numbers do not need any notebook: `python reproduce.py` (see the top-level README).

## Executed runs (evidence)

| Notebook | Experiment | Report section |
|---|---|---|
| `01_B0_baseline.ipynb` | B0: baseline | §2 |
| `02_D1_more_weight_updates_FINAL.ipynb` | D1: final model (training and single-label evaluation; diagnostics and galleries) | §3, §4 |
| `03_D1_seed1_noise_check.ipynb` | D1 with seed 1 | §1.2 |
| `04_A1_2x_upsampling_FINAL.ipynb` | A1: final model | §3 |
| `05_MX_multilabel_inference_D1_A1.ipynb` | MX: multi-label inference on D1 and A1 (final setting) and D1 vs A1 | §3 |
| `06_T1_single_class_detector.ipynb` | T1 | §5.1 |
| `07_DT1_dota_initialisation.ipynb` | DT1 | §5.1 |
| `08_TS1_two_stage_and_5.1_perfect_locations.ipynb` | TS1 and 5.1 | §5.1, §6.1 |
| `09_Y1_yolo26s.ipynb` | Y1 | §5 |
| `10_Y2_whole_scenes.ipynb` | Y2 | §5 |
| `11_Y6_imbalance_levers.ipynb` | Y6 | §5 |
| `12_RL_5.2_resist_learning.ipynb` | 5.2 | §6.2 |
| `13_S_5.3_5.4_subset_analysis.ipynb` | 5.3 and 5.4 | §6.3, §6.4 |
| `14_SR50_random_50pct.ipynb` | SR50_k0: D1 recipe on a random 50% subset | §6.3, §6.4 |
| `15_SR25_random_25pct.ipynb` | SR25_k0: random 25% subset | §6.3, §6.4 |
| `16_SC25_coverage_25pct.ipynb` | SC25: coverage-selected 25% subset | §6.4 |
| `17_SC12_coverage_12pct.ipynb` | SC12: coverage-selected 12.5% subset | §6.4 |

The training notebook of SR12_k0 (random 12.5%) is not included; its saved predictions are in `results/runs/SR12_k0/`.
