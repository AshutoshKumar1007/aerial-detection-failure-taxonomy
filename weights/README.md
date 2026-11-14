# Final model weights

The two proposed final models are YOLO11s detectors trained on Kaggle (Tesla T4) with `notebooks/01_train_and_evaluate.ipynb`.
Each checkpoint is the `last.pt` after the fixed 50-epoch schedule: no checkpoint was selected on any metric.

| File (put it here) | Model | Training config | Source (Kaggle run output) | sha256 |
|---|---|---|---|---|
| `D1_last.pt` | D1: tiles 1024, batch 16 / nbs 12 | `src/auric/configs/D1.yaml` | `work/runs/D1_s0/train/weights/last.pt` of the D1 run (`notebooks/executed/02_D1_more_weight_updates_FINAL.ipynb`) | `884dd241ae76cc9a7af40e3509398ae6e209970e1f694606e49b5a15de89b563` |
| `A1_last.pt` | A1: 512 px tiles up-sampled 2x to 1024 | `src/auric/configs/A1.yaml` | `work/runs/A1_s0/train/weights/last.pt` of the A1 run (`notebooks/executed/04_A1_2x_upsampling_FINAL.ipynb`) | `a7e35b6cb65bf8e401e20dd0061a9afb1541bbff71bb693e40ba6c08f6200d6c` |

`reproduce.py` loads `weights/<file>` (set in `config.yaml`, `models.<name>.weights`), prints the file's sha256 and warns
if it differs from the one recorded in `config.yaml`. `--weights <path>` points to a checkpoint stored elsewhere.

**Check of the checkpoints.** Run locally (RTX 4050) through `reproduce.py`:
* D1 reproduces the Kaggle evaluation of the D1 run on val: mAP50 0.0877, the same confidence interval and the same
  per-class AP50 (25,249 against 25,253 detections, i.e. GPU-level numerical differences); multi-label 0.1057 val.
* A1 gives multi-label val mAP50 0.1418 [0.093, 0.217], equal to the reported 0.142 and its interval.

**Downloaded as a folder?** A `.pt` file is a zip archive, and a browser download can unpack it into a folder
(`data.pkl`, `data/0`, `data/1`, ..., `version`). Such a folder is not loadable as it is: re-pack it with torch's own
writer, in torch's record order (`data.pkl`, `.format_version`, `.storage_alignment`, `byteorder`, `data/<n>` in numeric
order), into a file named `last.pt`, then rename it. The A1 file here was rebuilt this way. Its records are identical to the
download, but torch writes a fresh `.data/serialization_id`, so its sha256 is not the one of the file on Kaggle. The sha256
above is the one of this file.

The pretrained initialisation (`yolo11s.pt`, COCO) is downloaded by Ultralytics on first use. It is only needed for
training, not for `reproduce.py`.
