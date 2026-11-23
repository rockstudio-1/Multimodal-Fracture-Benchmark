"""Run a trained baseline and write full-resolution binary masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from fracturebench.data import dataset_root, load_index, load_patch, repo_root, select_patches
from fracturebench.prepare import MODELS
from fracturebench.train import _repo


def _config() -> dict:
    with (repo_root() / "configs" / "benchmark.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _save(directory: Path, stem: str, mask: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.where(mask > 0, np.uint8(255), np.uint8(0)))
    image.save(directory / f"{stem}.png")


def _upsample(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8))
    return np.array(image.resize((width, height), Image.Resampling.NEAREST))


def _load_state(path: Path):
    import torch

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model", "state_dict", "netG"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
        if any(name.startswith("module.") for name in state):
            state = {name.replace("module.", "", 1): value for name, value in state.items()}
    return state


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_deepcrack(ckpt: Path, samples: list[dict], output: Path, config: dict) -> None:
    import torch

    repo = _repo("deepcrack")
    sys.path.insert(0, str(repo))
    from models.deepcrack_networks import DeepCrackNet

    options = config["deepcrack"]
    model = DeepCrackNet(
        in_nc=3,
        num_classes=int(options["num_classes"]),
        ngf=64,
        norm=str(options["norm"]),
    )
    model.load_state_dict(_load_state(ckpt))
    device = _device()
    model.to(device).eval()
    size = int(options["load_width"])
    with torch.no_grad():
        for sample in samples:
            image = Image.fromarray(sample["rgb"]).resize((size, size), Image.Resampling.BICUBIC)
            tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
            tensor = (tensor - 0.5) / 0.5
            outputs = model(tensor.unsqueeze(0).to(device))
            fused = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
            probability = torch.sigmoid(fused)[0, 0].cpu().numpy()
            mask = _upsample((probability >= 0.5).astype(np.uint8) * 255, *sample["rgb"].shape[:2])
            _save(output, sample["stem"], mask)


def infer_swinunet(ckpt: Path, samples: list[dict], output: Path, config: dict) -> None:
    import torch

    repo = _repo("swinunet")
    sys.path.insert(0, str(repo))
    from config import get_config
    from networks.vision_transformer import SwinUnet as ViTSeg

    options = config["swinunet"]
    args = argparse.Namespace(
        cfg=str(repo / options["cfg"]),
        opts=["MODEL.PRETRAIN_CKPT", ""],
        batch_size=1,
        zip=False,
        cache_mode="part",
        resume="",
        accumulation_steps=None,
        use_checkpoint=False,
        amp_opt_level="O1",
        tag=None,
        eval=False,
        throughput=False,
    )
    model_config = get_config(args)
    model = ViTSeg(model_config, img_size=int(options["img_size"]), num_classes=int(options["num_classes"]))
    model.load_state_dict(_load_state(ckpt), strict=False)
    device = _device()
    model.to(device).eval()
    size = int(options["img_size"])
    with torch.no_grad():
        for sample in samples:
            gray = np.array(Image.fromarray(sample["rgb"]).convert("L").resize((size, size), Image.Resampling.BILINEAR))
            tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            logits = model(tensor.to(device))
            classes = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            mask = _upsample((classes == 1).astype(np.uint8) * 255, *sample["rgb"].shape[:2])
            _save(output, sample["stem"], mask)


def infer_cracksegformer(ckpt: Path, samples: list[dict], output: Path, config: dict) -> None:
    import torch
    from torchvision import transforms

    repo = _repo("cracksegformer")
    sys.path.insert(0, str(repo))
    from models.segformer.segformer import SegFormer

    options = config["cracksegformer"]
    model = SegFormer(num_classes=int(options["num_classes"]) + 1, phi=str(options["phi"]), pretrained=False)
    model.load_state_dict(_load_state(ckpt))
    device = _device()
    model.to(device).eval()
    size = int(options["img_size"])
    normalize = transforms.Normalize((0.473, 0.493, 0.504), (0.100, 0.100, 0.099))
    with torch.no_grad():
        for sample in samples:
            image = Image.fromarray(sample["rgb"]).resize((size, size), Image.Resampling.BILINEAR)
            tensor = transforms.ToTensor()(image)
            logits = model(normalize(tensor).unsqueeze(0).to(device))["out"]
            classes = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            mask = _upsample((classes == 1).astype(np.uint8) * 255, *sample["rgb"].shape[:2])
            _save(output, sample["stem"], mask)


def infer_bgcrack(ckpt: Path, samples: list[dict], output: Path, config: dict) -> None:
    import torch
    from torchvision import transforms

    repo = _repo("bgcrack")
    sys.path.insert(0, str(repo))
    from Model.BGCrack import BGCrack

    model = BGCrack()
    model.load_state_dict(_load_state(ckpt))
    device = _device()
    model.to(device).eval()
    size = int(config["bgcrack"]["img_size"])
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    with torch.no_grad():
        for sample in samples:
            image = Image.fromarray(sample["rgb"]).resize((size, size), Image.Resampling.BILINEAR)
            tensor = normalize(transforms.ToTensor()(image)).unsqueeze(0).to(device)
            prediction = model(tensor)
            score = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
            probability = score[0, 0].detach().cpu().numpy()
            mask = _upsample((probability >= 0.5).astype(np.uint8) * 255, *sample["rgb"].shape[:2])
            _save(output, sample["stem"], mask)


INFER = {
    "deepcrack": infer_deepcrack,
    "swinunet": infer_swinunet,
    "cracksegformer": infer_cracksegformer,
    "bgcrack": infer_bgcrack,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export binary predictions for evaluation.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--split", default="val", choices=("train", "val", "all"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    root = dataset_root(args.dataset)
    patches = select_patches(load_index(root), split=args.split)
    samples = [load_patch(root, patch) for patch in patches]
    INFER[args.model](Path(args.ckpt), samples, Path(args.out), _config())
    print(f"Wrote {len(samples)} masks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
