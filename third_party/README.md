# Third-party baselines

These directories are git submodules of the upstream projects:

| Directory | Repository |
| --- | --- |
| `DeepSegmentor` | https://github.com/yhlleo/DeepSegmentor |
| `Swin-Unet` | https://github.com/HuCaoFighting/Swin-Unet |
| `CrackSegFormer` | https://github.com/Li-Hubing/CrackSegFormer |
| `BGCrack` | https://github.com/hzlbbfrog/BGCrack |

Populate them from the repository root:

```bash
git submodule update --init --recursive
```

Each project keeps the license in its own tree. Training wrappers live in `fracturebench/` and do not modify these checkouts, aside from writing Swin-Unet list files and linking the BGCrack `Dataset` directory at launch time.
