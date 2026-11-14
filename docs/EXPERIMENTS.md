# Experiment records

One record per experiment, in the order the questions arose. Each record uses the assignment's fields: **Observation →
Hypothesis → Change vs. previous run → Results → Conclusion → Next**.

**Conventions.**
* mAP50 is the COCO 101-point AP at IoU 0.5 on full images, averaged over the five classes, with 95% image-bootstrap intervals.
* "dev" is the held-out 15% of train used for every decision; "val" (22 images) is reported only.
* The checkpoint is always `last.pt`; nothing is selected on a metric.
* Training runs used Kaggle Tesla T4 GPUs, Ultralytics 8.4.179 and seed 0 unless stated.
* `Run` names the configuration and the notebook overrides exactly as executed. `Evidence` is the executed notebook in
  `notebooks/executed/`.
* Decision rules marked *pre-declared* were written into the config or notebook before the run.

---

## B0: reference baseline
* **Observation.** We need a documented reference: a COCO-initialised detector at native resolution, evaluated on full images.
  Objects are tiny (median 18–31 px) and scenes are large (up to 7204 px), so the detector is trained on tiles.
* **Hypothesis.** None quantified. B0 is the reference that every later change is measured against.
* **Change.** `baseline.yaml`:
  * YOLO11s, COCO initialisation;
  * tiles 1024 px with overlap 200;
  * 50 epochs, SGD, Ultralytics default augmentation plus vertical flips.
  * Run: `EXPERIMENT="baseline"`, `OVERRIDES=["train.args.batch=16"]` (nbs 64, i.e. ~50 weight updates per epoch).
* **Results.**
  * mAP50: dev **0.089** [0.053, 0.136], val **0.072** [0.047, 0.107].
  * class-agnostic AP50: 0.265 / 0.277.
  * val per-class AP50: Cargo 0.120, Box 0.157, Flatbed 0.079, Tractor 0.005, Liquid 0.000. Liquid gets almost no detections.
  * Pipeline ceiling: the ground truth pushed through the same tiling and stitching scores 0.983 / 0.993, so the pipeline
    does not cap the score.
  * An earlier session that stopped at epoch ~10 gave dev 0.099 / val 0.074: the score is reached early.
* **Conclusion.** The baseline is far from the target. Class-agnostic AP (0.27) is about three times the class-aware
  score, so both finding and typing are weak.
* **Next.** Is the model under-trained? B0 makes only ~50 updates per epoch while the T4 has memory to spare → D1.
* **Evidence.** `01_B0_baseline.ipynb`.

## D1: more weight updates (final model)
* **Observation.** B0 accumulates gradients to nbs 64 (one update per 4 batches). The dev loss is already flat.
* **Hypothesis.** B0 learns slowly because it updates rarely; 5x the updates should raise mAP50.
* **Change vs B0.** nbs 64 → 12 with batch 16: every batch is an update, ~201 per epoch. Weight decay is scaled by
  Ultralytics to 6.7e-4. Run: `EXPERIMENT="D1"`, `OVERRIDES=["train.args.batch=16"]` (equal to `D1.yaml`).
* **Results.** 1.49 h on a T4, ≤ 11.9 GB.
  * mAP50: dev **0.101** [0.068, 0.147], val **0.088** [0.053, 0.137].
  * class-agnostic AP50: 0.299 / 0.297.
  * val per-class AP50: Cargo 0.110, Box 0.155, Flatbed 0.131, Tractor 0.043, Liquid 0.000. **No Liquid box at any
    confidence.**
  * val AP50 by size: xs 0.044, s 0.073, m 0.106.
  * Training dynamics: train loss keeps falling (box 2.21 → 1.78, cls 4.97 → 2.2) while the dev loss is flat from epoch
    ~15–20. Dev-tile mAP50 reaches ≈ 0.10 by epoch 30.
  * val outcomes at conf\*: TP 29%, never boxed 28%, low score 23%, wrong class 16%. 48% of the kept detections are
    background.
* **Conclusion.** **Not supported**: +0.012 dev, inside overlapping intervals. Optimisation is not the limit. D1 is kept as
  the reference recipe and is one of the two final models (REPORT §3).
* **Next.** Is it resolution? 23% of objects are < 16 px → A1.
* **Evidence.** `02_D1_more_weight_updates_FINAL.ipynb`.

## D1 seed 1: noise check
* **Observation.** Every comparison rests on one training seed; the size of seed-to-seed noise is unknown.
* **Hypothesis.** None: this measures noise.
* **Change vs D1.** `OVERRIDES=["experiment.seed=1", "train.args.batch=16"]`.
* **Results.** dev 0.104 [0.067, 0.146], val 0.081 [0.053, 0.121]; class-agnostic 0.295 / 0.283. Difference to seed 0:
  +0.003 dev, −0.007 val.
* **Conclusion.** Differences of ≤ 0.01 between single runs are noise. Used as the noise scale for every later comparison,
  and as the second full-data reference in 5.2–5.4.
* **Evidence.** `03_D1_seed1_noise_check.ipynb`.

## A1: 2x up-sampling (final model)
* **Observation.** 23% of train objects are < 16 px (Cargo 35%). A 16 px box covers ~2 cells of the stride-8 feature map.
* **Hypothesis.** Missed small objects are the dominant error. Up-sampling 512 px tiles to the 1024 px input gives a 22 px
  truck 44 px (4x the cells), which should lift AP for the xs bucket.
* **Change vs B0.** Tiles 512 (overlap 200) enlarged to imgsz 1024 in training and inference. Object-free tile share 0.25 →
  0.07, because a 512 grid has ~37,000 empty tiles. Otherwise B0 (batch 8, nbs 64). Run: `EXPERIMENT="A1"`, no overrides.
* **Results.** ~3.9 h on a T4.
  * mAP50: dev **0.100** [0.067, 0.152], val **0.131** [0.082, 0.207].
  * class-agnostic AP50: 0.252 / 0.303.
  * val per-class AP50: Cargo 0.148, Box 0.255, Flatbed 0.060, Tractor 0.036, Liquid **0.158** (3 of 20 objects).
  * xs AP: dev 0.026 → 0.015, val 0.044 → 0.073 (no consistent direction).
  * Paired comparison with D1 on dev: mAP50 equal; class-agnostic AP50 **0.047 [0.013, 0.070] in D1's favour**.
* **Conclusion.** The predicted mechanism (better xs AP) did not appear, and A1 is not better on dev. The val gain is made
  of three Liquid objects and Box, and is inside the noise of 22 images. Resolution is not the main limit. A1 is kept as
  the second final model because it is the best on val and equal on dev (REPORT §3).
* **Unresolved.** A1 also sees each object ~1.5x as often per epoch, which confounds the up-sampling effect.
* **Next.** The analyses of the D1/A1 predictions (analyses P1–P5 below) showed that objects are found but not separated,
  and that the class step costs a factor ≈ 2.3. That raised two questions: does the inference throw away correct class
  hypotheses (→ MX)? And does the fused class score cost detection (→ T1)?
* **Evidence.** `04_A1_2x_upsampling_FINAL.ipynb`.

## MX: emit every plausible class (inference only)
* **Observation.** `model.predict` keeps one class per box, whereas Ultralytics' own validator keeps every class above the
  threshold. For an object that is ambiguous between Cargo and Box, the second hypothesis can only score if it is emitted.
* **Hypothesis.** Multi-label output raises mAP50, most for Box, Flatbed and Tractor; the gain will be well below a third
  of the mAP (the upper estimate declared before the run). *Pre-declared rule:* keep it if the paired dev interval excludes 0.
* **Change.** Same D1 and A1 weights; `inference.multi_label: true`, image cap 2,000 → 5,000 boxes.
* **Results.**

  | | dev mAP50 single → multi (paired diff) | val mAP50 single → multi (paired diff) |
  |---|---|---|
  | D1 | 0.101 → **0.115** (+0.014 [0.009, 0.024]) | 0.088 → **0.106** (+0.018 [0.010, 0.031]) |
  | A1 | 0.100 → **0.108** (+0.008 [0.005, 0.012]) | 0.131 → **0.142** (+0.010 [0.005, 0.015]) |

  * Per class (dev, D1/A1): Box +0.026/+0.018, Flatbed +0.023/+0.017, Liquid +0.018/+0.007, Cargo ≈ 0, Tractor ≈ 0.
  * Class-agnostic AP50 falls 0.007–0.012.
  * D1 vs A1 with multi-label: dev +0.007 [−0.012, 0.019] for D1; val −0.036 [−0.101, 0.013] for D1, i.e. A1 ahead.
* **Conclusion.** **Kept**, the final inference setting. The gain is real and small, largest on the ambiguous classes, with
  no gain in detection: the signature of hedging an ambiguous type. Tractor gains nothing, because it is limited by recall.
* **Unresolved.** The cap change rides along. 32% of val images reached the old 2,000 cap against 3% of dev, so the dev gain
  is the multi-label effect while the val gain cannot be attributed to it alone.
* **Evidence.** `05_MX_multilabel_inference_D1_A1.ipynb`; reproduced locally for D1 with `reproduce.py`.

## T1: single-class detector
* **Observation.** YOLO11 heads have no objectness branch, so the class score is both "is there a truck" and "which
  truck". With ambiguous labels, the confidence for an obvious truck is split across classes. Class-agnostic AP50 is only
  0.25–0.30.
* **Hypothesis.** The fused objective costs detection; one class pooling all 7,618 boxes would find trucks clearly better.
  *Pre-declared rule:* dev class-agnostic AP50 ≥ 0.40 supports, ≤ 0.32 does not, in between is inconclusive.
* **Change vs D1.** `train.args.single_cls: true` only. Run: `EXPERIMENT="T1"`, `OVERRIDES=["train.args.batch=16"]`.
* **Results.** 1.50 h. Class-agnostic AP50 **0.326 dev / 0.324 val** (D1 0.299 / 0.297). The class-aware columns are
  meaningless for a one-class model.
* **Conclusion.** **Inconclusive** by the rule: +0.027, against the +0.10 margin declared. The fused objective is not what
  pins detection near 0.3. A seed repeat was the declared next step and was not run.
* **Next.** Is the COCO representation the detection limit? → DT1. Can typing be done better on these boxes? → TS1.
* **Evidence.** `06_T1_single_class_detector.ipynb`.

## DT1: aerial (DOTA) initialisation
* **Observation.** Every run starts from COCO features, but the images are overhead. The xView-pretrained route is excluded
  (integrity); DOTA is an allowed aerial source.
* **Hypothesis.** An aerial backbone represents small overhead vehicles better, which would raise single-class detection.
* **Change vs T1.** Initial weights only: the backbone and neck of the Ultralytics YOLO11s-OBB DOTAv1 checkpoint in a detect
  model with a fresh head (`auric.model.pretrain.build_dota_transfer_init`; 493 of 499 tensors transferred).
  `OVERRIDES=["train.args.batch=16"]`.
* **Results.** 1.58 h. Class-agnostic AP50 **0.275 dev / 0.277 val** (T1: 0.326 / 0.324).
* **Conclusion.** **Not better**: the point estimate is 0.05 lower. One seed each, so "worse" is not claimed. The pretrained
  representation is not the detection limit.
* **Evidence.** `07_DT1_dota_initialisation.ipynb`.

## TS1: two stages, T1 boxes and a crop classifier
* **Observation.** On the detectors' boxes, typing costs about two-thirds of the AP: class-agnostic 0.30 against mAP50 0.10.
  A decoupled, stronger classifier therefore looked like the one direction with headroom.
* **Hypothesis.** A two-stage pipeline (best single-class boxes, then a stronger classifier with more context) beats the
  fused head. *Pre-declared rule (`TS1.yaml`):* dev mAP50 ≥ D1 + its CI width (≈ 0.13); inside the noise band means
  "no better".
* **Change.**
  * Stage 1: the T1 weights, class-agnostic merge.
  * Stage 2: ConvNeXt-tiny (ImageNet), crop side 3x the object at 128 px, 20 epochs, lr 3e-4, class-balanced loss,
    trained on train_core ground-truth crops.
  * Final score = √(det score · p(class)); the best two classes are emitted per box.
  * Run: `EXPERIMENT="TS1"`, `OVERRIDES=["two_stage.classifier.batch=256"]`.
* **Results.**
  * Detection ceiling 0.326 / 0.324.
  * **Two-stage mAP50 0.080 [0.053, 0.122] dev / 0.098 [0.064, 0.153] val**; classification cost 0.245 / 0.225.
  * val per-class AP50: Cargo 0.136, Box 0.190, Flatbed 0.109, Tractor 0.007, Liquid 0.050.
  * Classifier: loss 1.42 → 0.42, in-sample accuracy 0.61 → 0.986. Training took 4.5 h on the T4, because every crop
    decodes its whole scene.
  * An earlier run with batch 128 gave the same dev score (0.080) and val 0.076. Its notebook is not in the package.
* **Conclusion.** **Rejected**: below D1 on dev (0.080 vs 0.101). Decoupling plus a stronger classifier does not beat the
  fused head, and the typing loss stays at 0.22–0.25.
* **Next.** Is it the classifier or the boxes? Feed the same classifier the true boxes → PL (assignment 5.1).
* **Evidence.** `08_TS1_two_stage_and_5.1_perfect_locations.ipynb`.

## PL: if object locations were perfect (assignment 5.1)
* **Observation.** TS1 cannot say whether its typing loss comes from imperfect boxes or from the classifier and labels.
* **Hypothesis.** *Pre-declared:* with perfect boxes, A ≥ 0.38. A > 0.60 would mean typing is not the limit;
  A ≤ 0.45 would mean the five-way problem stays far from 0.75. Emitting two labels raises A. Cargo and Box score best.
* **Change.** The ground-truth boxes, typed by the TS1 classifier (trained on train_core only), are scored by the same
  evaluator with 1, 2 or 5 labels per box.
* **Results.**
  * mAP50 with perfect boxes, 1 / 2 / 5 labels: dev 0.233 / **0.292** / 0.361, val 0.291 / **0.358** / 0.421.
  * Held-out typing accuracy 0.569 / 0.581 (always answering Cargo: 0.488 / 0.515); top-2 accuracy 0.726 / 0.764.
  * The rare types are absorbed by Cargo and Box (Liquid → Cargo/Box 0.90 dev). The detector's box and the true box give
    the same type for 87% of objects.
* **Conclusion.**
  * Expectations: A ≤ 0.45 met; A ≥ 0.38 not met; two labels help (+0.06–0.07, met); Cargo/Box best mostly met
    (Tractor is above Flatbed).
  * **64–71% of the AP is still missing when localisation is removed.** Typing under these labels is the main limit.
  * The limits of the experiment are listed in REPORT §6.1.
* **Next.** Separate label ambiguity from scarcity: 5.2 (where do models fail repeatedly), 5.3 (does more data help) and
  5.4 (which data matters).
* **Evidence.** `08_TS1_two_stage_and_5.1_perfect_locations.ipynb`.

## Y1: YOLO26s (architecture)
* **Observation.** A newer architecture might be better at small objects, although the evidence so far does not point at
  capacity.
* **Hypothesis.** Only "Y1 vs D1 isolates the architecture"; it also serves as the control for Y2–Y6.
* **Change vs D1.** `model.weights: yolo26s.pt`, one-to-many head with NMS at inference. `OVERRIDES=["train.args.batch=16"]`.
* **Results.** 1.85 h, ≤ 13.7 GB. dev 0.106 [0.069, 0.153], val 0.088 [0.057, 0.131]; class-agnostic 0.293 / 0.283. No
  Liquid detection at any confidence.
* **Conclusion.** **Same as YOLO11s.** The architecture is not the limit.
* **Evidence.** `09_Y1_yolo26s.ipynb`.

## Y2: whole scenes instead of tiles
* **Observation.** The scenes are delivered whole; the question is whether tiling is needed.
* **Hypothesis.** *Stated in `Y2.yaml` before the run:* worse. Letterboxing to imgsz 3008 shrinks large scenes, e.g. a 20 px
  truck in the largest val scene drops to ~8 px.
* **Change vs Y1.** `formulation.mode: scenes`, imgsz 3008, one window at inference. `OVERRIDES=["train.args.batch=12"]`
  (nbs 2). The effective batch could not be confirmed from the log (one scene per iteration).
* **Results.** 3.40 h. dev **0.054** [0.036, 0.080], val **0.035** [0.026, 0.050]; class-agnostic 0.195 / 0.187.
* **Conclusion.** **As expected, the worst formulation**: about half the tile runs' score. Native-scale tiles are needed.
* **Evidence.** `10_Y2_whole_scenes.ipynb`.

## Y6: imbalance levers
* **Observation.** Class imbalance is strong (Liquid 2.5% of train) and mAP weights every class equally. D1's dev recall is
  1.3% for Tractor and 0% for Liquid.
* **Hypothesis.** Inverse-frequency class weights plus a class-aware sampler raise rare-class recall and AP.
* **Change vs Y1, as executed.** Four changes at once, so no single lever can be attributed:
  * `cls_pw` 0.5 (class weights 0.58 / 0.60 / 1.17 / 0.84 / 1.81);
  * class-aware sampler (alpha 0);
  * classification gain `cls` 0.5 → 1.5;
  * batch 12.
  * Run: `OVERRIDES=["train.args.batch=12", "train.args.cls=1.5"]`. The single-lever configs Y3–Y5 exist but were not run.
* **Results.** 1.92 h.
  * mAP50: dev 0.075 [0.044, 0.113], val 0.081 [0.045, 0.115].
  * Maximum same-class recall fell for Flatbed (dev 0.32 → 0.17) and Tractor (0.33 → 0.20).
  * Liquid boxes are now emitted (134 dev), but none of them match an object.
  * Dev-tile mAP peaks at epoch 20, then falls.
* **Conclusion.** **Not supported**: the rare classes got no better. Measurement of the tile cache shows why sampling cannot
  balance classes: rare-class tiles are crowded with Cargo and Box, so even a fully class-balanced draw moves Liquid only
  from 2.3% to 3.7% of instances.
* **Evidence.** `11_Y6_imbalance_levers.ipynb`.

## RL: examples that resist learning (assignment 5.2)
* **Observation.** All full-data models score ~0.1 and share their blind spots (11% of objects are missed by D1 and A1 at
  any confidence, against 3% if their errors were independent).
* **Hypothesis.** *Pre-declared:* resistance concentrates in the rare types and the smallest objects; Cargo ↔ Box appears in
  the typing; scene conditions matter little; attributes predict failure (held-out AUC 0.65–0.75).
  *Rule:* val AUC lower bound > 0.5 and top-fifth lift ≥ 1.3.
* **Change.** No training. The saved predictions of D1 s0, D1 s1 and A1 are used: an object is `R_miss` if no model solves it
  at its dev operating point. Attributes are class, size, crowding, border contact, scene density, brightness and
  sharpness. A logistic model is fitted on dev and tested on val.
* **Results.**
  * `R_miss` 62% dev / 67% val. The two D1 seeds disagree on 9% of objects.
  * Populations: xs objects (not found 0.81 / 0.75), rare types (Liquid 0.97 / 0.85, Tractor 0.92 / 0.95), Cargo ↔ Box by
    size.
  * Prediction test: val AUC **0.646 [0.531, 0.752]**, lift **1.32 [1.14, 1.55]**: **supported, narrowly**; size alone
    gives 0.638.
* **Conclusion.** Several populations with different mechanisms (REPORT §6.2). The analysis uses held-out objects, not
  training objects (no train-set predictions were saved).
* **Evidence.** `12_RL_5.2_resist_learning.ipynb`; protocol `notebooks/03_resist_learning.ipynb`.

## SR / SC: data efficiency (assignments 5.3 and 5.4)
* **Observation.** It is unknown whether more labels would help, or which training examples carry the score.
* **Hypothesis / declared protocol (before any subset run).**
  * The D1 recipe is trained on 50%, 25% and 12.5% of the train_core instances, on random image subsets (seed 0) and on
    a coverage selection (greedy over class × size cells).
  * Success means ≥ 90% of the full-data mAP50 (mean of the two D1 seeds) on val and on dev.
  * Coverage is compared with random at the same budget by a paired bootstrap.
  * "Coverage better than random" is only claimed against a range of random draws, which a single draw per budget does
    not provide.
* **Change.** `EXPERIMENT="SR"` with `["train.args.batch=16", "experiment.name=SR25_k0", "experiment.seed=0",
  "subset.method=random", "subset.budget_frac=0.25", "subset.seed=0"]` (and 0.50, 0.125); `EXPERIMENT="SC"` with
  `["train.args.batch=16", "experiment.name=SC25", "experiment.seed=0", "subset.method=coverage",
  "subset.budget_frac=0.25", "subset.seed=0"]` (and 0.125).
* **Results.**
  * No tested subset passes: the best is random 50% (77% val / 82% dev of full).
  * Coverage is no better at 25% and worse at 12.5% (−0.025 [−0.041, −0.013] dev).
  * Learning curve: +0.016 mAP50 per doubling (dev), so +500 labels ≈ +0.002. Per-class slopes: Box 0.042, Cargo 0.016,
    Flatbed 0.016, Tractor 0.004, Liquid 0.
* **Conclusion.** REPORT §6.3–6.4.
  * 500 labels would not matter.
  * Dense common-class scenes are the useful data.
  * Rare-class coverage is not the bottleneck.
* **Evidence.** `13_S_5.3_5.4_subset_analysis.ipynb`. The five subset training runs were executed on Kaggle with
  `notebooks/01_train_and_evaluate.ipynb`: `14_SR50_random_50pct`, `15_SR25_random_25pct`, `16_SC25_coverage_25pct`,
  `17_SC12_coverage_12pct` (the SR12_k0 training notebook is not included). Their predictions are the analysis input
  (`results/runs/`).

---

## Supporting analyses (no training)

These analyses were run locally on the saved D1/A1 predictions and on the dataset. They are exploratory evidence behind the
diagnosis. They are not part of the reproducible `reproduce.py` path.

| ID | Question | Result | Limits |
|---|---|---|---|
| CK | Is the pipeline correct? | The evaluator equals pycocotools to 1e-3; ground truth through tiling and stitching scores 0.983 / 0.993; a 16-tile overfit run reaches 0.995; tile labels read back exactly | — |
| P1 | Are objects found? | 81–85% of objects get a box at IoU ≥ 0.5 at any confidence (90–92% at IoU 0.1); only 15–18% never receive a box. Class-agnostic AP 0.357 → 0.303 from IoU 0.1 to 0.5 | D1 and A1 only |
| P2 | Are false positives noise? | D1's confident background boxes are found again by A1 at 0.60 / 0.77 (dev / val), as often as its true positives | — |
| P3 | Shared blind spots? | 11% of objects are missed by both models at any confidence vs 3% if independent; the union of A1 and D1 adds +0.005 | — |
| P4 | Are two coarse val scenes the cause? | Removing val 2384 and 2460 moves mAP50 by +0.004 (D1) / +0.006 (A1) | — |
| AU | Are background detections real errors? | Blind audit of 200 crops: 42–45% of confident unmatched detections are clearly unlabelled trucks; controls behave as expected | one rater; dev only (val crops too dark); intervals 29–58% |
| IM | Can sampling balance the classes? | No: rare-class tiles carry Cargo/Box, so a fully balanced draw moves Liquid from 2.3% to 3.7% of instances | — |

**Not run (declared gaps).**
* Seed repeats other than D1; a T1 seed repeat.
* The single imbalance levers (Y3, Y4, Y5).
* A blinded relabelling of val.
* Train-set predictions (needed for a training-set version of 5.2).
* The NMS / one-to-one head variants on YOLO26.
