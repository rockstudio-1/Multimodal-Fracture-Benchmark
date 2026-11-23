# Reference baselines

The four models are the upstream projects, checked out as git submodules under `third_party/`. This repository does not vendor their source. Training and inference import or launch that code after `scripts/prepare_data.py` has written the folder layout each project already documents.

Install the Python dependencies listed in the submodule before training it. ImageNet initialization files are not redistributed here.

| Model | Submodule | Input used here | Official training budget |
| --- | --- | --- | --- |
| DeepCrack | `third_party/DeepSegmentor` | RGB, 256 | 400 epochs at the initial learning rate, then 300 decay epochs. Batch size 1. Focal loss. |
| Swin-Unet | `third_party/Swin-Unet` | Grayscale, 224 | 150 epochs, batch size 24, base learning rate 0.05. |
| CrackSegFormer | `third_party/CrackSegFormer` | RGB, 512 | 200 epochs, batch size 16, AdamW learning rate `6e-5`, MiT-B0. |
| BGCrack | `third_party/BGCrack` | RGB plus a mask gradient, 512 | 70 epochs, batch size 9, learning rate `6e-3`. |

Defaults live in `configs/benchmark.yaml`. Upstream Swin-Unet rescales the learning rate when the batch size is a multiple of 6 and is not 24. That behavior is left unchanged.

## Weights the upstream code expects

- **Swin-Unet.** Place Swin-T at `third_party/Swin-Unet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth`. The upstream README links the checkpoint.
- **CrackSegFormer.** Place MiT-B0 at `third_party/CrackSegFormer/pretrained_weights/segformer/mit_b0.pth`. Training stops with an explicit message if the file is absent.

DeepCrack and BGCrack train from their own initializers.

## Where checkpoints are written

| Model | Path |
| --- | --- |
| DeepCrack | `runs/deepcrack/deepcrack/latest_net_G.pth` |
| Swin-Unet | `runs/swinunet/epoch_*.pth` |
| CrackSegFormer | `third_party/CrackSegFormer/logs/` (the path hardcoded by that repository) |
| BGCrack | `runs/bgcrack/steel_cracks_with_edgeBGCrack/1/` |

BGCrack's launcher uses `torch.distributed.run` when its training script calls `init_process_group`. The data directory inside that repository is a link to `work/prepared/bgcrack/Dataset`, created at launch.

## Commands

Prepare every layout:

```bash
python scripts/prepare_data.py --model all
```

Train one model:

```bash
python scripts/train.py --model deepcrack
python scripts/train.py --model swinunet
python scripts/train.py --model cracksegformer
python scripts/train.py --model bgcrack
```

Write validation masks and score them:

```bash
python scripts/infer.py --model deepcrack --ckpt runs/deepcrack/deepcrack/latest_net_G.pth --split val --out runs/deepcrack/preds
python scripts/evaluate.py --pred-dir runs/deepcrack/preds --split val --output runs/deepcrack/metrics.json
```

Prediction files are `{stem}.png`, with crack pixels at 255. They are resized back to 1024×1024 before the metrics in `docs/metrics.md` are computed. In-training numbers printed by an upstream script are not the benchmark report.
