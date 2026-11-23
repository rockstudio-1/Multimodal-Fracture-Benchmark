"""Check that the on-disk dataset matches the manifest."""

from __future__ import annotations

import argparse
from collections import Counter

from fracturebench.data import REQUIRED, dataset_root, load_index, load_patch, modality_path


def check(root=None) -> dict:
    root = dataset_root(root)
    patches = load_index(root)
    missing = []
    for patch in patches:
        for modality in REQUIRED:
            path = modality_path(root, patch, modality)
            if not path.is_file():
                missing.append(str(path))
        if patch.has_crack:
            vector = modality_path(root, patch, "vectors")
            if not vector.is_file():
                missing.append(str(vector))
    if missing:
        preview = "\n".join(missing[:12])
        raise SystemExit(f"Missing {len(missing)} files. First entries:\n{preview}")

    counts = Counter((patch.split, patch.station) for patch in patches)
    sample_ok = 0
    for station in sorted({patch.station for patch in patches}):
        patch = next(item for item in patches if item.station == station)
        sample = load_patch(root, patch)
        if sample["rgb"].shape != (patch.height, patch.width, 3):
            raise SystemExit(f"{patch.stem} rgb shape {sample['rgb'].shape}")
        if sample["depth_mm"].shape != (patch.height, patch.width):
            raise SystemExit(f"{patch.stem} depth shape {sample['depth_mm'].shape}")
        if sample["geom"].shape != (patch.height, patch.width):
            raise SystemExit(f"{patch.stem} geom shape {sample['geom'].shape}")
        if sample["orientation"].shape != (patch.height, patch.width, 2):
            raise SystemExit(f"{patch.stem} orientation shape {sample['orientation'].shape}")
        if sample["label"].shape != (patch.height, patch.width):
            raise SystemExit(f"{patch.stem} label shape {sample['label'].shape}")
        sample_ok += 1
    cracked = sum(patch.has_crack for patch in patches)
    report = {
        "patches": len(patches),
        "cracked": cracked,
        "stations_checked": sample_ok,
        "by_split": {split: sum(count for (name, _station), count in counts.items() if name == split) for split in ("train", "val")},
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the dataset tree.")
    parser.add_argument("--dataset", default="dataset")
    args = parser.parse_args(argv)
    report = check(args.dataset)
    print(
        f"patches={report['patches']} cracked={report['cracked']} "
        f"train={report['by_split']['train']} val={report['by_split']['val']} "
        f"stations_checked={report['stations_checked']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
