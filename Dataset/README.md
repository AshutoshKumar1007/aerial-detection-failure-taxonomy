# Dataset Setup & External Source

The dataset contains high-resolution aerial imagery (~6.3 GB) across 5 vehicle classes (Cargo, Box, Flatbed, Tractor, Liquid). Due to size limits, raw image files are hosted externally.

### 📥 Download from Kaggle

The dataset is publicly hosted on Kaggle:
👉 **[Kaggle Dataset: shriramprajapati/auricdataset](https://www.kaggle.com/datasets/shriramprajapati/auricdataset)**

#### Option 1: Using Kaggle CLI
```bash
# Download archive to Dataset/
cd Dataset
kaggle datasets download -d shriramprajapati/auricdataset

# Unzip and arrange contents
unzip auricdataset.zip
# Ensure CV_Assignment_AAL is at: Dataset/CV_Assignment_AAL/
```

#### Option 2: Manual Download
1. Visit [https://www.kaggle.com/datasets/shriramprajapati/auricdataset](https://www.kaggle.com/datasets/shriramprajapati/auricdataset).
2. Click **Download (6.3 GB)** and extract the archive.
3. Place `CV_Assignment_AAL` into this directory:

```
Dataset/
└── CV_Assignment_AAL/
    ├── classmap.txt            # 0: Cargo, 1: Box, 2: Flatbed, 3: Tractor, 4: Liquid
    ├── train/
    │   ├── images/             # 443 PNG scenes (1369x1334 to 7204x5458 px)
    │   └── labels/             # YOLO-format bounding boxes (.txt per image)
    └── val/
        ├── images/             # 22 PNG scenes
        └── labels/
```

---

### ⚙️ Dataset Usage & Configuration

* **Validation Reproduction:** Only `val/` is strictly required to reproduce the validation metrics and failure analysis tables via `reproduce.py`.
* **Training & Dev Split:** `train/` is required for training runs and for evaluating on the fixed stratified `dev` split (`src/auric/configs/splits/dev_s0.json`).
* **Path Resolution:** `reproduce.py` reads `runtimes.local.dataset_root` from `config.yaml`, and protocol notebooks read `src/auric/configs/env/local.yaml`. You can also override the dataset path via CLI:
  ```bash
  python reproduce.py --dataset-root /path/to/CV_Assignment_AAL
  ```
