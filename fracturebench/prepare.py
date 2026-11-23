"""Export the benchmark into the folder layout each upstream baseline expects."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from fracturebench.data import dataset_root, load_index, load_patch, repo_root, select_patches

MODELS = ("deepcrack", "swinunet", "cracksegformer", "bgcrack")


def _config() -> dict:
    with (repo_root() / "configs" / "benchmark.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _resize(array: np.ndarray, size: int, nearest: bool) -> np.ndarray:
    image = Image.fromarray(array)
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return np.array(image.resize((size, size), resample))


def _binary(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, np.uint8(255), np.uint8(0))


def _edge(mask: np.ndarray) -> np.ndarray:
    """4-neighborhood morphological gradient of a binary mask."""
    binary = mask > 0
    padded = np.pad(binary, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    dilated = center | up | down | left | right
    eroded = center & up & down & left & right
    gradient = dilated & np.logical_not(eroded)
    return np.where(gradient, np.uint8(255), np.uint8(0))


def _gray(rgb: np.ndarray) -> np.ndarray:
    red = rgb[..., 0].astype(np.float32)
    green = rgb[..., 1].astype(np.float32)
    blue = rgb[..., 2].astype(np.float32)
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _iter_split(dataset: Path):
    patches = load_index(dataset)
    for split in ("train", "val"):
        for patch in select_patches(patches, split=split):
            yield split, patch, load_patch(dataset, patch)


def prepare_deepcrack(dataset: Path, output: Path) -> None:
    for split, _patch, sample in _iter_split(dataset):
        folder = output / f"{split}_img"
        label = output / f"{split}_lab"
        name = sample["stem"] + ".png"
        _save_png(folder / name, sample["rgb"])
        _save_png(label / name, _binary(sample["mask"]))
    print(f"[deepcrack] {output}")


def prepare_cracksegformer(dataset: Path, output: Path) -> None:
    for split, _patch, sample in _iter_split(dataset):
        name = sample["stem"] + ".png"
        _save_png(output / split / "images" / name, sample["rgb"])
        _save_png(output / split / "masks" / name, _binary(sample["mask"]))
    print(f"[cracksegformer] {output}")


def prepare_swinunet(dataset: Path, output: Path, size: int) -> None:
    lists = output / "lists"
    arrays = output / "train_npz"
    arrays.mkdir(parents=True, exist_ok=True)
    lists.mkdir(parents=True, exist_ok=True)
    handles = {
        "train": (lists / "train.txt").open("w", encoding="utf-8"),
        "val": (lists / "val.txt").open("w", encoding="utf-8"),
    }
    try:
        for split, _patch, sample in _iter_split(dataset):
            gray = _resize(_gray(sample["rgb"]).astype(np.uint8), size, nearest=False)
            label = _resize(_binary(sample["mask"]), size, nearest=True)
            image = gray.astype(np.float32) / 255.0
            classes = (label > 127).astype(np.uint8)
            np.savez(arrays / f"{sample['stem']}.npz", image=image, label=classes)
            handles[split].write(sample["stem"] + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    print(f"[swinunet] {output}")


def prepare_bgcrack(dataset: Path, output: Path, size: int) -> None:
    names = {"train": "Train", "val": "Validation"}
    for split, _patch, sample in _iter_split(dataset):
        phase = names[split]
        root = output / "Dataset" / "Steelcrack" / phase
        name = sample["stem"] + ".png"
        rgb = _resize(sample["rgb"], size, nearest=False)
        mask = _resize(_binary(sample["mask"]), size, nearest=True)
        _save_png(root / "images" / name, rgb)
        _save_png(root / "masks" / name, mask)
        _save_png(root / "edges" / name, _edge(mask))
    print(f"[bgcrack] {output}")


def prepare(model: str, dataset: Path, output: Path, config: dict | None = None) -> Path:
    config = config or _config()
    destination = output / model
    if model == "deepcrack":
        prepare_deepcrack(dataset, destination)
    elif model == "cracksegformer":
        prepare_cracksegformer(dataset, destination)
    elif model == "swinunet":
        prepare_swinunet(dataset, destination, int(config["swinunet"]["img_size"]))
    elif model == "bgcrack":
        prepare_bgcrack(dataset, destination, int(config["bgcrack"]["img_size"]))
    else:
        raise SystemExit(f"Unknown model {model!r}. Choose from {', '.join(MODELS)} or all.")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare baseline training folders.")
    parser.add_argument("--model", default="all", choices=MODELS + ("all",))
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    config = _config()
    dataset = dataset_root(args.dataset)
    output = Path(args.out) if args.out else repo_root() / config["prepared"]
    models = MODELS if args.model == "all" else (args.model,)
    for name in models:
        prepare(name, dataset, output, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
