"""Command line entry point."""

from __future__ import annotations

import argparse

from fracturebench import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fracturebench", description="Multimodal fracture benchmark.")
    parser.add_argument("--version", action="version", version=f"fracturebench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check_parser = sub.add_parser("check", help="Verify dataset files against index.json.")
    check_parser.add_argument("--dataset", default="dataset")

    prepare_parser = sub.add_parser("prepare", help="Write baseline-specific training folders.")
    prepare_parser.add_argument("--model", default="all")
    prepare_parser.add_argument("--dataset", default="dataset")
    prepare_parser.add_argument("--out", default="")

    train_parser = sub.add_parser("train", help="Train one upstream baseline.")
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--dataset", default="dataset")
    train_parser.add_argument("--prepared", default="")
    train_parser.add_argument("--runs", default="")
    train_parser.add_argument("--skip-prepare", action="store_true")

    infer_parser = sub.add_parser("infer", help="Write full-resolution prediction masks.")
    infer_parser.add_argument("--model", required=True)
    infer_parser.add_argument("--ckpt", required=True)
    infer_parser.add_argument("--dataset", default="dataset")
    infer_parser.add_argument("--split", default="val")
    infer_parser.add_argument("--out", required=True)

    eval_parser = sub.add_parser("evaluate", help="Score prediction masks.")
    eval_parser.add_argument("--dataset", default="dataset")
    eval_parser.add_argument("--pred-dir", required=True)
    eval_parser.add_argument("--split", default="val")
    eval_parser.add_argument("--output", default="")
    eval_parser.add_argument("--tolerance-radius", type=int, default=2)
    eval_parser.add_argument("--skeleton-iterations", type=int, default=5)

    args = parser.parse_args(argv)
    if args.command == "check":
        from fracturebench.check import main as command
        return command(["--dataset", args.dataset])
    if args.command == "prepare":
        from fracturebench.prepare import main as command
        forwarded = ["--model", args.model, "--dataset", args.dataset]
        if args.out:
            forwarded.extend(["--out", args.out])
        return command(forwarded)
    if args.command == "train":
        from fracturebench.train import main as command
        forwarded = ["--model", args.model, "--dataset", args.dataset]
        if args.prepared:
            forwarded.extend(["--prepared", args.prepared])
        if args.runs:
            forwarded.extend(["--runs", args.runs])
        if args.skip_prepare:
            forwarded.append("--skip-prepare")
        return command(forwarded)
    if args.command == "infer":
        from fracturebench.infer import main as command
        return command(
            ["--model", args.model, "--ckpt", args.ckpt, "--dataset", args.dataset, "--split", args.split, "--out", args.out]
        )
    if args.command == "evaluate":
        from fracturebench.evaluate import main as command
        forwarded = ["--dataset", args.dataset, "--pred-dir", args.pred_dir, "--split", args.split,
                     "--tolerance-radius", str(args.tolerance_radius),
                     "--skeleton-iterations", str(args.skeleton_iterations)]
        if args.output:
            forwarded.extend(["--output", args.output])
        return command(forwarded)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
