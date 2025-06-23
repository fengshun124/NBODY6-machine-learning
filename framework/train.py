import os
from datetime import datetime

import click
import pytorch_lightning as pl
import torch

from model.transformer import NBodyTransformerRegressor
from module.data import NBodyDataModule
from module.sampler import NBodySnapshotNonEmptySampler

RANDOM_SEED = 42

pl.seed_everything(seed=RANDOM_SEED)
torch.set_float32_matmul_precision("medium")


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option("--num-jobs", type=click.INT, default=-1, show_default=True)
@click.option("--num-worker", type=click.INT, default=4, show_default=True)
@click.option("--val-frac", default=0.2, show_default=True)
@click.option("--batch-size", default=512, show_default=True)
@click.option("--hidden-dim", type=click.INT, default=32, show_default=True)
@click.option("--num-heads", type=click.INT, default=8, show_default=True)
@click.option("--num-inducing", type=click.INT, default=16, show_default=True)
@click.option("--num-seeds", type=click.INT, default=8, show_default=True)
@click.option("--num-sab", type=click.INT, default=2, show_default=True)
@click.option("--lr", type=click.FLOAT, default=1e-3, show_default=True)
@click.option("--max-epochs", type=click.INT, default=1000, show_default=True)
def train(
    data_path: str,
    num_jobs: int = -1,
    num_worker: int = 4,
    val_frac: float = 0.2,
    batch_size: int = 128,
    hidden_dim: int = 32,
    num_heads: int = 2,
    num_inducing: int = 16,
    num_seeds: int = 2,
    num_sab: int = 1,
    lr: float = 1e-3,
    max_epochs: int = 100,
) -> None:
    # versioning
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    model_hyper_param_str = (
        f"hd{hidden_dim}-"
        f"h{num_heads}-"
        f"ind{num_inducing}-"
        f"seed{num_seeds}-"
        f"sab{num_sab}-"
        f"bs{batch_size}-"
        f"lr{f'{lr:.0e}'.replace('-', 'n').replace('.', 'p')}"
    )
    print(f'Registering model: "{model_hyper_param_str}/{start_time_str}"')

    sampler = NBodySnapshotNonEmptySampler(nbody_simulation_root=data_path)
    sampler.collect()
    sampler.sample(
        n_star_per_sample=256,
        n_sample_per_snapshot=24,
        n_jobs=num_jobs,
    )
    print(sampler)

    data_module = NBodyDataModule(
        snapshots_pairs=sampler.sampled_snapshot_pairs,
        snapshot_scalar_dict={},
        attr_tuple_scalar_dict={
            0: "RobustScaler",
            1: "RobustScaler",
            2: "RobustScaler",
        },
        batch_size=batch_size,
        val_frac=val_frac,
        num_workers=num_worker,
        random_seed=RANDOM_SEED,
    )
    data_module.setup()

    model = NBodyTransformerRegressor(
        input_dim=data_module.feature_dim,
        output_dim=data_module.attr_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_inducing=num_inducing,
        num_seeds=num_seeds,
        num_sab=num_sab,
        lr=lr,
    )

    os.makedirs("../output/checkpoints", exist_ok=True)
    os.makedirs("../output/logs", exist_ok=True)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=pl.loggers.CSVLogger(
            save_dir="../output/logs",
            name=model_hyper_param_str,
            version=start_time_str,
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="../output/checkpoints",
                monitor="val_mse",
                mode="min",
                save_top_k=1,
                filename=(
                    f"{start_time_str}-tf_regression-{model_hyper_param_str}-"
                    "{epoch:03d}-"
                    "{val_loss:.4f}"
                ),
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_mse",
                patience=10,
                mode="min",
                verbose=True,
            ),
        ],
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
