"""Launch upstream training entry points on the prepared benchmark split."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from fracturebench.data import repo_root
from fracturebench.prepare import MODELS, prepare

REPOS = {
    "deepcrack": "DeepSegmentor",
    "swinunet": "Swin-Unet",
    "cracksegformer": "CrackSegFormer",
    "bgcrack": "BGCrack",
}


def _config() -> dict:
    with (repo_root() / "configs" / "benchmark.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _repo(name: str) -> Path:
    path = repo_root() / "third_party" / REPOS[name]
    if not path.is_dir():
        raise SystemExit(
            f"Missing {path}. Initialize submodules with:\n"
            "  git submodule update --init --recursive"
        )
    return path


def _run(command: list[str], cwd: Path, extra_env: dict | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _link_directory(link: Path, target: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise SystemExit(
            f"{link} already exists and is not a link. Move it aside so training "
            f"can point it at {target}."
        )
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True)


def train_deepcrack(config: dict, prepared: Path, runs: Path) -> None:
    options = config["deepcrack"]
    repo = _repo("deepcrack")
    data = prepared / "deepcrack"
    output = runs / "deepcrack"
    output.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable, "train.py",
            "--dataroot", str(data),
            "--name", "deepcrack",
            "--model", "deepcrack",
            "--dataset_mode", "deepcrack",
            "--gpu_ids", "0",
            "--niter", str(options["niter"]),
            "--niter_decay", str(options["niter_decay"]),
            "--batch_size", str(options["batch_size"]),
            "--num_classes", str(options["num_classes"]),
            "--norm", str(options["norm"]),
            "--lr", str(options["lr"]),
            "--lr_policy", "step",
            "--lr_decay_iters", "40",
            "--load_width", str(options["load_width"]),
            "--load_height", str(options["load_height"]),
            "--loss_mode", str(options["loss_mode"]),
            "--display_id", "0",
            "--checkpoints_dir", str(output),
        ],
        cwd=repo,
    )


def train_swinunet(config: dict, prepared: Path, runs: Path) -> None:
    options = config["swinunet"]
    repo = _repo("swinunet")
    data = prepared / "swinunet"
    list_dir = repo / "lists" / "Synapse"
    list_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.txt", "val.txt"):
        shutil.copyfile(data / "lists" / name, list_dir / name)
    output = runs / "swinunet"
    output.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable, "train.py",
            "--dataset", "Synapse",
            "--cfg", str(options["cfg"]),
            "--root_path", str(data),
            "--output_dir", str(output),
            "--img_size", str(options["img_size"]),
            "--max_epochs", str(options["max_epochs"]),
            "--base_lr", str(options["base_lr"]),
            "--batch_size", str(options["batch_size"]),
            "--n_class", str(options["num_classes"]),
        ],
        cwd=repo,
    )


def train_cracksegformer(config: dict, prepared: Path, runs: Path) -> None:
    options = config["cracksegformer"]
    repo = _repo("cracksegformer")
    weights = repo / "pretrained_weights" / "segformer" / "mit_b0.pth"
    if not weights.is_file():
        raise SystemExit(
            "CrackSegFormer expects an ImageNet MiT-B0 weight at "
            f"{weights}. Place the file shipped by that repository there, "
            "then rerun training."
        )
    output = runs / "cracksegformer"
    output.mkdir(parents=True, exist_ok=True)
    print(f"[cracksegformer] upstream stores checkpoints in {repo / 'logs'}")
    _run(
        [
            sys.executable, "tools/train.py",
            "--data-path", str(prepared / "cracksegformer"),
            "--num-classes", str(options["num_classes"]),
            "--phi", str(options["phi"]),
            "--pretrained", "True",
            "--lr", str(options["lr"]),
            "--batch-size", str(options["batch_size"]),
            "--epochs", str(options["epochs"]),
        ],
        cwd=repo,
    )


def train_bgcrack(config: dict, prepared: Path, runs: Path) -> None:
    options = config["bgcrack"]
    repo = _repo("bgcrack")
    source = prepared / "bgcrack" / "Dataset"
    _link_directory(repo / "Dataset", source)
    output = runs / "bgcrack"
    output.mkdir(parents=True, exist_ok=True)
    script = (repo / "train_BGCrack_2024.py").read_text(encoding="utf-8", errors="ignore")
    save_path = str(output).replace("\\", "/") + "/"
    command = [
        "--dataset", str(options["dataset"]),
        "--modelname", str(options["modelname"]),
        "--epoch", str(options["epoch"]),
        "--batchsize", str(options["batchsize"]),
        "--lr", str(options["lr"]),
        "--save_path", save_path,
    ]
    if "init_process_group" in script:
        _run(
            [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "1", "train_BGCrack_2024.py", *command],
            cwd=repo,
        )
    else:
        _run([sys.executable, "train_BGCrack_2024.py", *command], cwd=repo)


TRAINERS = {
    "deepcrack": train_deepcrack,
    "swinunet": train_swinunet,
    "cracksegformer": train_cracksegformer,
    "bgcrack": train_bgcrack,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one reference baseline.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--prepared", default="")
    parser.add_argument("--runs", default="")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args(argv)
    config = _config()
    prepared = Path(args.prepared) if args.prepared else repo_root() / config["prepared"]
    runs = Path(args.runs) if args.runs else repo_root() / config["runs"]
    if not args.skip_prepare:
        prepare(args.model, args.dataset, prepared, config)
    TRAINERS[args.model](config, prepared, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
