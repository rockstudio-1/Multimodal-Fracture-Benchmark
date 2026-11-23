"""Score prediction PNGs against the benchmark masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fracturebench.data import dataset_root, load_index, modality_path, select_patches
from fracturebench.metrics import MetricAccumulator


def _read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        array = np.array(image.convert("L"))
    if array.shape != size:
        resized = Image.fromarray(array).resize(size[::-1], Image.Resampling.NEAREST)
        array = np.array(resized)
    return array


def _find_prediction(directory: Path, stem: str) -> Path | None:
    for name in (f"{stem}.png", f"{stem}_rgb.png", f"{stem}_mask.png"):
        path = directory / name
        if path.is_file():
            return path
    return None


def evaluate_directory(
    dataset: Path,
    prediction_dir: Path,
    split: str = "val",
    tolerance_radius: int = 2,
    skeleton_iterations: int = 5,
) -> dict:
    patches = select_patches(load_index(dataset), split=split)
    if not patches:
        raise SystemExit(f"No patches for split {split!r} under {dataset}")
    missing = []
    accumulator = MetricAccumulator(tolerance_radius, skeleton_iterations)
    for patch in patches:
        prediction = _find_prediction(prediction_dir, patch.stem)
        if prediction is None:
            missing.append(patch.stem)
            continue
        ground_truth = np.array(Image.open(modality_path(dataset, patch, "masks")))
        predicted = _read_mask(prediction, ground_truth.shape)
        accumulator.update(predicted, ground_truth)
    if missing:
        preview = ", ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f" ... ({len(missing)} total)"
        raise SystemExit(f"Missing predictions for: {preview}{extra}")
    report = accumulator.compute()
    report["split"] = split
    report["aggregation"] = "micro"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate binary crack predictions.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "all"))
    parser.add_argument("--output", default="")
    parser.add_argument("--tolerance-radius", type=int, default=2)
    parser.add_argument("--skeleton-iterations", type=int, default=5)
    args = parser.parse_args(argv)
    report = evaluate_directory(
        dataset_root(args.dataset),
        Path(args.pred_dir),
        split=args.split,
        tolerance_radius=args.tolerance_radius,
        skeleton_iterations=args.skeleton_iterations,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
