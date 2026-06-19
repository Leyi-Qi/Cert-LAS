# Cert-LAS: Toward Certified Model Ownership Verification for Text-to-Image Diffusion Models via Layer-Adaptive Smoothing

This repository contains the official implementation for "Cert-LAS: Toward Certified Model Ownership Verification for Text-to-Image Diffusion Models via Layer-Adaptive Smoothing" (ICML 2026).

## Repository Structure

```text
Cert-LAS
├── configs/                 # JSON configs for training/evaluation runs
├── scripts/                 # Reproducible shell entry points
└── src/cert_las/
    ├── train/               # Watermark training
    ├── eval/                # VSR and certified-radius evaluation
    └── utils/               # Shared diffusion, smoothing, IO, and statistics utilities
```

The current release keeps the active path intentionally narrow:

- watermark training: diffusion-classifier objective with the default sampler
- VSR evaluation: default VAE path, with DDIM available via `--sampler ddim`
- radius evaluation: closed-form Cert-LAS radius from VSR CSVs

The training CLI accepts `--sampler ddim` as the reserved hook for the future DDIM training merge. VSR evaluation supports `--sampler ddim` now.

## Installation

```bash
pip install -e .
```

The original experiments used the local `diffusers` environment from `diff_wm_certify`. If you are running on the same machine, you can also invoke the scripts with:

```bash
PYTHON=/data/home/Boheng/miniconda3/envs/diffusers/bin/python
ACCELERATE_BIN=/data/home/Boheng/miniconda3/envs/diffusers/bin/accelerate
```

## Watermark Training

Edit `configs/train_watermark.json`, then run:

```bash
bash scripts/train_watermark.sh configs/train_watermark.json
```

Useful overrides:

```bash
GPU_IDS=0 NUM_PROCESSES=1 bash scripts/train_watermark.sh \
  configs/train_watermark.json \
  --output_dir outputs/debug_watermark \
  --max_train_steps 20 \
  --save_steps 10
```

The training script saves checkpoints as:

```text
outputs/<run>/step_<n>/unet/
outputs/<run>/step_<n>/metrics.json
outputs/<run>/train_metrics.jsonl
```

## Evaluation

Run VSR and certified-radius evaluation together:

```bash
bash scripts/eval.sh configs/eval_vsr.json configs/eval_radius.json
```

DDIM evaluation uses the same script:

```bash
bash scripts/eval.sh configs/eval_vsr.json configs/eval_radius.json --sampler ddim --ddim_steps 50 \
  --output_dir outputs/watermark/step_2000-vsr0.01-ddim50
```

You can also use `configs/eval_vsr_ddim.json` as a ready-to-edit DDIM example. The script first writes VSR files:

```text
*_per_trial.csv
*_per_sample.csv
*_summary.csv
*_meta.json
```

The VSR summary includes `wr`, `rp`, `zeta`, `verification_threshold`, `ownership_verified`, and `vsr`. The threshold uses the final closed-form formula with `M=gen_trials` and `N=gen_num_images`; `rp` is estimated from the clean-side per-trial hit rates, and the finite-sample upper bound for `zeta` uses the `M` trial-level statistics by default.

Then it feeds the generated `*_per_trial.csv` into the radius evaluator and writes:

```text
certified_radius.json
certified_radius_summary.csv
```

The radius evaluator uses the mean of `clean_hit_rate` as `P_Xc` by default. `--reference_probability` is only an optional override for replaying an external baseline.

## Development Notes

The original `diff_wm_certify/` directory is not modified by this repository layout. The implementation here extracts the active training/evaluation flow into reusable modules, following the `src + configs + scripts` pattern used by DREAM and ResAlign while keeping Odysseus-style compact model/evaluation utilities.

## License

This project is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
