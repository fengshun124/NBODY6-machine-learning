# Machine Learning Framework for Star Cluster Properties Prediction

This repository contains a deep learning framework designed to predict properties of star clusters based on variable-size sets of their member stars. This Pytorch-Lightning-based framework implements three different architectures for set-based regression tasks, with a focus on variable-size input handling and permutation invariance.

## Requirements

Python 3.12+ with the following packages (see [`requirements.txt`](./requirements.txt) for pinned versions):

- `click`
- `joblib`
- `numpy`
- `pandas`
- `python-dotenv`
- `pytorch_lightning`
- `scikit-learn`
- `torch`
- `torchmetrics`
- `tqdm`
- `pyarrow`

Optionally, for GPU monitoring during training, install [nvitop](https://github.com/XuehaiPan/nvitop):

```bash
# using conda (conda-forge)
conda install -c conda-forge nvitop

# or using pip
pip install nvitop
```

## Usage

### Configure Environment

Create a `.env` file from the [`.env.template`](.env.template) and set the required variables (at minimum):

- `OUTPUT_BASE` (where pipeline/dataset/train outputs are written)
- `JOBLIB_ROOT` (joblib cache root used by the NBODY6 data pipeline)
- `SPLIT_MFT_JSON` (path to split manifest used by `build_dataset.py`)

```bash
# Edit .env and set the variables listed above
cp .env.template .env
```

### Prepare Dataset

Build the cached, scaled dataset from NBODY6 pipeline joblib caches. Run this from the `machine-learning` directory:

```bash
python ./src/build_dataset.py
```

Outputs produced under `OUTPUT_BASE/dataset/`:

- `raw-<split>-shard.npz` — merged per-split raw shards (train/val/test)
- `scaled-<split>-shard.npz` — scaled shards used for training and evaluation
- `feature_scaler_bundle.joblib`, `target_scaler_bundle.joblib` — fitted scalers

Notes:

- The script reads the split manifest at `SPLIT_MFT_JSON` and joblib caches under `JOBLIB_ROOT` (see NBODY6-data-pipeline for cache layout).
- If you customize feature/target keys or scaler configuration, update the call or the env vars accordingly.

### Train a model

After building the dataset, invoke the training entrypoint from `machine-learning`:

```bash
# Example: use one GPU
CUDA_VISIBLE_DEVICES=0 python ./src/train.py \
  --dataset /path/to/cached/dataset/ \
  --feature-keys x \
  --feature-keys y \
  --feature-keys z \
  --feature-keys vx \
  --feature-keys vy \
  --feature-keys vz \
  --target-key total_mass_within_2x_r_tidal \
  --model set_transformer \
  --hparam 'hidden_dim=8' \
  --hparam 'num_heads=8' \
  --hparam 'num_sabs=4' \
  --hparam 'output_hidden_dims=(4,2)' \
  --hparam 'dropout=0.2' \
  --num-workers 16 \
  -bs 20480 \
  -lr 1e-4 \
  -wd 3e-3
```

Run with `--help` to see all available options:

```bash
python ./src/train.py --help
```

- Dataset and feature/target specification:

  - `--dataset`: path to the `dataset` folder created by `build_dataset.py`.
  - `--feature-keys`: repeatable; per-element input features (e.g., `x`, `y`, `z`, `vx`, `vy`, `vz`).
  - `--target-key`: regression target to predict.
  - `--num-star-per-sample`: number of stars per input set sample.
  - `--num-sample-per-snapshot`: number of set samples drawn from each snapshot during training/validation/testing.
  - `--drop-probability`: probability of dropping stars from samples when a snapshot contains more than `--num-star-per-sample` stars (default `0.6`).
  - `--drop-ratio-range`: lower and upper bound (tuple) for the fraction of stars to drop when applying random star dropping (default `(0.1, 0.9)`).

- Model selection and hyperparameters:

  - `--model`: model architecture name (`set_transformer`, `deep_sets`, `summary_stats`).
  - `--hparam KEY=VALUE`: model hyperparameters (repeatable). Values accept Python literals (numbers, tuples, booleans).
  - `--huber-delta`: delta parameter used by the Huber loss during training (default `1.0`).

- Training configuration:

  - `--seed`: random seed for reproducibility.
  - `-lr` / `-wd` / `-bs`: shorthand for `--learning-rate`, `--weight-decay`, `--batch-size`, respectively.
  - `--num-workers`: number of DataLoader workers.
  - `--pin-memory/--no-pin-memory`: toggle `pin_memory` for the DataLoader (default `--pin-memory`).
  - `--max-epochs`: maximum number of training epochs.
  - `--patience`: early stopping patience in epochs.

See [`src/train.py`](./src/train.py) for full details on all options.

### *Quick Checklist*

- Create and edit `.env` from `.env.template`.
- Ensure `JOBLIB_ROOT` points to NBODY6 joblib caches.
- Run `python ./src/build_dataset.py` to build shards and scalers.
- Run `python ./src/train.py` with chosen `--feature-keys` and `--target-key`.
- Check logs and checkpoints under directories created in `OUTPUT_BASE`.

## Model Architecture

The framework provides three permutation-invariant architectures for variable-size input sets:

- [`SummaryStatsRegressor`](./src/model/summary_stats.py): computes descriptive statistics (e.g., mean, std, quantiles) across set features, then feeds them to an MLP for regression.

- [`DeepSetRegressor`](./src/model/deep_set.py): applies a per-element encoder, aggregates via a permutation-invariant pooling (e.g., sum/mean), and decodes with an MLP following the [Deep Sets](https://arxiv.org/abs/1703.06114) design.

- [`SetTransformerRegressor`](./src/model/set_transformer.py): uses attention-based [Set Transformer](https://arxiv.org/abs/1810.00825) blocks to model interactions among members before pooling and final MLP decoding.
