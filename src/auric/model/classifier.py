"""Stage-2 crop classifier: the 5-way truck typing, decoupled from detection.

Why this exists. Every end-to-end run (B0, D1, A1, Y1, Y6) sits at class-agnostic AP50 0.25-0.30 and
per-class mAP50 ~0.10, so typing costs about two-thirds of the AP. This module is a strong
ImageNet-pretrained backbone on a configurable context window, so the two-stage pipeline can measure how much of the lost AP is classification and how much a better classifier buys.

Interfaces reused unchanged: `read_image` (BGR HxWxC uint8), `Annotation`/`Detections` (abs-px xyxy).
Train on train_core GT crops only; dev and val are held-out *images*, so applying the trained model to them
has no leakage.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..data.dataset import ImageRecord, read_image
from ..structures import Annotation

# ImageNet stats, applied to RGB crops (read_image is BGR -> we flip once in _crop).
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def context_window(box: np.ndarray, W: int, H: int, context: float, min_px: int) -> tuple[int, int, int, int]:
    """Square crop around a box: side = context * max(bw, bh, min_px), centred, clipped to the image.

    `context` > 1 adds the surroundings (a truck is identified by its bed/trailer,
    which often falls outside a tight box). min_px stops a 10 px box producing a degenerate crop.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = context * max(x2 - x1, y2 - y1, float(min_px))
    h = side / 2
    ix0, iy0 = int(round(cx - h)), int(round(cy - h))
    ix1, iy1 = int(round(cx + h)), int(round(cy + h))
    # clip but keep it square where possible
    ix0, iy0 = max(0, ix0), max(0, iy0)
    ix1, iy1 = min(W, ix1), min(H, iy1)
    return ix0, iy0, max(ix0 + 1, ix1), max(iy0 + 1, iy1)


def _crop(img_bgr: np.ndarray, win, out_px: int) -> np.ndarray:
    x0, y0, x1, y1 = win
    c = img_bgr[y0:y1, x0:x1, ::-1]                       # -> RGB
    t = torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1).float() / 255.0
    t = F.interpolate(t[None], size=(out_px, out_px), mode="bilinear", align_corners=False)[0]
    return ((t - _MEAN) / _STD).numpy()


@dataclass
class CropSpec:
    context: float = 3.0      # crop side = context * object size
    min_px: int = 24          # floor so tiny boxes still yield a usable crop
    out_px: int = 128         # classifier input


class BoxCropDataset(Dataset):
    """GT boxes (training) or detection boxes (inference) -> normalised crops, grouped by image for one decode."""

    def __init__(self, records: list[ImageRecord], boxes_by_image: dict[str, np.ndarray],
                 labels_by_image: dict[str, np.ndarray] | None, spec: CropSpec, train: bool):
        self.spec, self.train = spec, train
        self.path = {r.image_id: r.image_path for r in records}
        self.items: list[tuple[str, int]] = []
        self.boxes, self.labels = boxes_by_image, labels_by_image
        for iid, bx in boxes_by_image.items():
            self.items += [(iid, j) for j in range(len(bx))]
        self._cache_id, self._cache_img = None, None

    def _image(self, iid: str) -> np.ndarray:
        if iid != self._cache_id:
            self._cache_id, self._cache_img = iid, read_image(self.path[iid])
        return self._cache_img

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        iid, j = self.items[i]
        img = self._image(iid)
        H, W = img.shape[:2]
        win = context_window(self.boxes[iid][j], W, H, self.spec.context, self.spec.min_px)
        x = torch.from_numpy(_crop(img, win, self.spec.out_px))
        if self.train:
            if torch.rand(1) < 0.5: x = x.flip(2)       # overhead imagery: both flips are label-preserving
            if torch.rand(1) < 0.5: x = x.flip(1)
            if torch.rand(1) < 0.5: x = torch.rot90(x, int(torch.randint(1, 4, (1,))), (1, 2))
        y = -1 if self.labels is None else int(self.labels[iid][j])
        return x, y


def build_backbone(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """torchvision ImageNet backbones (no extra dependency). convnext_tiny default; resnet50 fallback.

    pretrained=False builds the bare architecture (the weights then come from a saved classifier)."""
    import torchvision.models as M
    if name == "convnext_tiny":
        m = M.convnext_tiny(weights=M.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "resnet50":
        m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone {name!r}")
    return m


@dataclass
class CropClassifier:
    """Trained stage-2 model + its crop spec. `predict_proba` returns (N, K) softmax over the detection boxes."""
    model: nn.Module
    spec: CropSpec
    device: object
    class_names: list[str]
    history: list[dict] | None = None   # per-epoch {epoch, loss, train_acc, lr} from train_classifier
    backbone: str = "convnext_tiny"

    def save(self, path: str | Path) -> None:
        """Weights plus what is needed to rebuild the classifier: spec, class names, backbone, training history."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.model.state_dict(), "spec": asdict(self.spec), "class_names": self.class_names,
                    "backbone": self.backbone, "history": self.history}, path)

    @classmethod
    def load(cls, path: str | Path, device=0) -> "CropClassifier":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = build_backbone(blob["backbone"], len(blob["class_names"]), pretrained=False)
        model.load_state_dict(blob["state"])
        return cls(model.to(device).eval(), CropSpec(**blob["spec"]), device, blob["class_names"], blob["history"],
                   blob["backbone"])

    @torch.no_grad()
    def predict_proba(self, records, boxes_by_image: dict[str, np.ndarray], batch: int = 256,
                      workers: int = 4) -> dict[str, np.ndarray]:
        self.model.eval()
        ds = BoxCropDataset(records, boxes_by_image, None, self.spec, train=False)
        if len(ds) == 0:
            return {iid: np.zeros((0, len(self.class_names))) for iid in boxes_by_image}
        dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
        probs = []
        for x, _ in tqdm(dl, desc="stage2 classify", unit="batch", leave=False):
            p = F.softmax(self.model(x.to(self.device)).float(), 1).cpu().numpy()
            probs.append(p)
        probs = np.concatenate(probs)
        out, k = {}, 0
        for iid, bx in boxes_by_image.items():
            out[iid] = probs[k:k + len(bx)]
            k += len(bx)
        return out


def train_classifier(records: list[ImageRecord], anns: dict[str, Annotation], class_names: list[str],
                     spec: CropSpec, *, backbone: str = "convnext_tiny", epochs: int = 20, lr: float = 3e-4,
                     batch: int = 128, workers: int = 4, class_balanced: bool = True, device=0,
                     seed: int = 0) -> CropClassifier:
    """Train on GT crops from `records` (use train_core only). Returns a CropClassifier ready for predict_proba."""
    torch.manual_seed(seed)
    K = len(class_names)
    boxes = {r.image_id: anns[r.image_id].boxes for r in records if len(anns[r.image_id])}
    labels = {r.image_id: anns[r.image_id].cls for r in records if len(anns[r.image_id])}
    ds = BoxCropDataset(records, boxes, labels, spec, train=True)

    counts = np.bincount(np.concatenate(list(labels.values())).astype(int), minlength=K)
    # inverse-frequency CE weight; the sampler cannot rebalance (IM finding), so weight the loss instead.
    w = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** (0.5 if class_balanced else 0.0), dtype=torch.float32)
    w = (w / w.mean()).to(device)

    model = build_backbone(backbone, K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True)
    model.train()
    history: list[dict] = []
    print(f"classifier {backbone}: {len(ds)} crops, {len(dl)} batches of {batch} per epoch, {epochs} epochs, lr {lr}", flush=True)
    t0 = time.time()
    for ep in range(epochs):
        lr_ep = opt.param_groups[0]["lr"]
        tot = correct = n = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=w, label_smoothing=0.05)
            loss.backward()
            opt.step()
            bl = loss.detach().item()                         # .detach() avoids the requires_grad->scalar warning
            tot += bl * len(y); n += len(y)
            correct += int((logits.detach().argmax(1) == y).sum())
        sched.step()
        ep_loss, ep_acc = tot / max(n, 1), correct / max(n, 1)
        history.append({"epoch": ep + 1, "loss": ep_loss, "train_acc": ep_acc, "lr": lr_ep})
        print(f"epoch {ep + 1:>2}/{epochs}  loss {ep_loss:.4f}  train_acc {ep_acc:.3f}  lr {lr_ep:.2e}  "
              f"{time.time() - t0:6.0f}s elapsed", flush=True)
    return CropClassifier(model, spec, device, class_names, history, backbone)
