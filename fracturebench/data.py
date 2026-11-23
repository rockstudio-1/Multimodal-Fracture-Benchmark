"""Dataset root, manifests, and per-patch arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REQUIRED = (
    "images",
    "masks",
    "confidence",
    "ids",
    "depth",
    "depth_valid",
    "geom",
)
OPTIONAL = ("vectors",)

_SUFFIX = {
    "images": "_rgb.png",
    "masks": "_mask.png",
    "confidence": "_conf.png",
    "ids": "_id.png",
    "depth": "_depth.png",
    "depth_valid": "_dvalid.png",
    "geom": "_geom.npy",
    "vectors": "_vec.npy",
}


@dataclass(frozen=True)
class Patch:
    stem: str
    station: str
    split: str
    has_crack: bool
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    fov_deg: float
    width: int
    height: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_root(path: str | Path | None = None) -> Path:
    if path is None:
        root = repo_root() / "dataset"
    else:
        root = Path(path)
        if not root.is_absolute():
            root = repo_root() / root
    return root


def load_index(root: str | Path | None = None) -> list[Patch]:
    root = dataset_root(root)
    path = root / "index.json"
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    patches = []
    for item in document["patches"]:
        patches.append(
            Patch(
                stem=item["stem"],
                station=item["station"],
                split=item["split"],
                has_crack=bool(item["has_crack"]),
                yaw_deg=float(item["yaw_deg"]),
                pitch_deg=float(item["pitch_deg"]),
                roll_deg=float(item["roll_deg"]),
                fov_deg=float(item["fov_deg"]),
                width=int(item["width"]),
                height=int(item["height"]),
            )
        )
    return patches


def select_patches(patches: list[Patch], split: str | None = None, station: str | None = None) -> list[Patch]:
    chosen = patches
    if split is not None and split != "all":
        chosen = [item for item in chosen if item.split == split]
    if station is not None:
        chosen = [item for item in chosen if item.station == station]
    return chosen


def modality_path(root: Path, patch: Patch, modality: str) -> Path:
    return root / modality / f"{patch.stem}{_SUFFIX[modality]}"


def _read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image)


def class_label(mask: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Map a patch to {0 background, 1 visible crack, 2 occluded crack}."""
    crack = mask > 0
    label = np.zeros(mask.shape, dtype=np.int64)
    label[crack & (confidence >= 128)] = 1
    label[crack & (confidence > 0) & (confidence < 128)] = 2
    return label


def load_patch(root: str | Path, patch: Patch) -> dict:
    """Load every modality stored for one patch.

    Color images are standard RGB PNGs. Depth is uint16 millimeters.
    Orientation is ``(cos 2θ, sin 2θ)`` and is stored only for patches that
    contain a crack; missing files are returned as zeros.
    """
    root = dataset_root(root)
    rgb = _read_image(modality_path(root, patch, "images"))
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    mask = _read_image(modality_path(root, patch, "masks"))
    confidence = _read_image(modality_path(root, patch, "confidence"))
    instance = _read_image(modality_path(root, patch, "ids")).astype(np.int32)
    depth_mm = _read_image(modality_path(root, patch, "depth")).astype(np.float32)
    valid = (_read_image(modality_path(root, patch, "depth_valid")) > 0).astype(np.uint8)
    geom = np.load(modality_path(root, patch, "geom")).astype(np.float32)
    vector_path = modality_path(root, patch, "vectors")
    if vector_path.is_file():
        orientation = np.load(vector_path).astype(np.float32)
    else:
        orientation = np.zeros(rgb.shape[:2] + (2,), dtype=np.float32)
    return {
        "stem": patch.stem,
        "station": patch.station,
        "split": patch.split,
        "rgb": rgb,
        "mask": mask.astype(np.uint8),
        "confidence": confidence.astype(np.uint8),
        "label": class_label(mask, confidence),
        "instance": instance,
        "depth_mm": depth_mm,
        "depth_valid": valid,
        "geom": geom,
        "orientation": orientation,
    }
