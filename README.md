# Deep Learning Framework for Star Cluster Properties Prediction

This repository contains a deep learning framework designed to predict properties of star clusters based on variable-size sets of their member stars. This Pytorch-Lightning-based framework implements three different architectures for set-based regression tasks, with a focus on variable-size input handling and permutation invariance.

## Requirements

- Python 3.12+
- PyTorch 2.5+
- PyTorch Lightning

## Usage

### Prepare Dataset

```sh
python ./framework/build_dataset.py
```

### Train a Model

```sh
# use `CUDA_VISIBLE_DEVICES` to select GPU(s) if needed.
CUDA_VISIBLE_DEVICES=0 python ./framework/train.py \
  # data configuration
  --data cache/9ef03baf1395 \
  --feature-keys x \
  --feature-keys y \
  --feature-keys z \
  --feature-keys vx \
  --feature-keys vy \
  --feature-keys vz \
  --target-key total_mass_within_2x_r_tidal \
  # model configuration
  --model set_transformer \
  --hparam 'hidden_dim=8' \
  --hparam 'num_heads=8' \
  --hparam 'num_sabs=4' \
  --hparam 'output_hidden_dims=(4,2)' \
  --hparam 'dropout=0.2' \
  # training configuration
  --batch-size 20480 \
  --num-workers 16 \
  -lr 1e-4 \
  -wd 3e-3 \
  --max-epochs 100 \
  --log-dir logs
```

Notes:

- Short aliases are available for convenience: `-lr` for `--learning-rate`, and `-wd` for `--weight-decay`.
- The code expects hyphen-style option names (for example `--feature-keys`, `--target-key`, `--batch-size`).
- Available models: `deep_sets`, `summary_stats`, `set_transformer`.

## Model Architecture

The framework includes three architectures: A MLP interpreting the descriptive statistics of the set, Deep Sets, as well as Set Transformer. Each architecture is designed to handle variable-size input sets and maintain permutation invariance.
