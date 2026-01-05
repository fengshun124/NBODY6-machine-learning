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
    # in case of multiple GPUs, specify which GPU to use
CUDA_VISIBLE_DEVICES=0 python ./framework/train.py \
    # path to the cached dataset, the ID is generated while constructing the dataset.
  --data 'cache/9ef03baf1395' \
    # choose one of the implemented models: 'deep_sets', 'point_net', 'set_transformer'
  --model 'set_transformer' \
    # or any feature keys in the dataset   
  --feature_keys 'x' \
  --feature_keys 'y' \
  --feature_keys 'z' \
  --feature_keys 'vx' \
  --feature_keys 'vy' \
  --feature_keys 'vz' \
    # or any target key in the dataset
  --target_key 'total_mass_within_2x_r_tidal' \
    # model hyperparameters, check the implemented models for available options
  --hparam 'hidden_dim=8' \
  --hparam 'num_heads=8' \
  --hparam 'num_sabs=4' \
  --hparam 'output_hidden_dims=(4,2)' \
  --hparam 'dropout=0.2' \
    # training hyperparameters
  --batch_size '20480' \
  --num_workers '16' \
  --learning_rate '1e-4' \
  --weight_decay '3e-3' \
  --max_epochs '100' \
  --log_dir 'logs'
```

## Model Architecture

The framework includes three architectures: A MLP interpreting the descriptive statistics of the set, Deep Sets, as well as Set Transformer. Each architecture is designed to handle variable-size input sets and maintain permutation invariance.
