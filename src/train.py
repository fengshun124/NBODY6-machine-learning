import ast
import logging
import os
import platform
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl
import torch
from dataset.module import NBODY6DataModule
from model import DeepSetRegressor, SetTransformerRegressor, SummaryStatsRegressor
from orchestrator import LightningRegressionOrchestrator
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from utils import OUTPUT_BASE, setup_logger

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")


# models and their default hyperparameters
MODEL_REGISTRY: dict[str, dict[str, object]] = {
    "set_transformer": {
        "class": SetTransformerRegressor,
        "defaults": {
            "hidden_dim": 6,
            "num_heads": 2,
            "num_sabs": 1,
            "output_hidden_dims": None,
            "dropout": 0.1,
            "is_apply_layer_norm": True,
        },
    },
    "deep_sets": {
        "class": DeepSetRegressor,
        "defaults": {
            "phi_hidden_dims": (8, 4),
            "rho_hidden_dims": (8, 4),
            "dropout": 0.1,
            "pooling": "mean",
        },
    },
    "summary_stats": {
        "class": SummaryStatsRegressor,
        "defaults": {
            "hidden_dims": (8, 4),
            "dropout": 0.1,
        },
    },
}


def _fmt_value(v: object) -> str:
    """Format value for version string: '.' -> 'p', '-' -> 'n'."""
    s = str(v)
    if isinstance(v, (list, tuple)):
        s = "x".join(map(str, v))
    return s.replace(".", "p").replace("-", "n")


def _parse_hparams(pairs: tuple[str, ...]) -> dict[str, object]:
    out: dict[str, object] = {}
    for p in pairs:
        if "=" not in p:
            raise click.BadParameter(f"Expected KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            val: object = v.lower() == "true"
        else:
            try:
                val = ast.literal_eval(v)
            except Exception:
                val = v
        out[k.strip()] = val
    return out


def _build_model(
    model_name: str,
    model_hparams: dict[str, object],
    input_dim: int,
    input_feature_labels: tuple[str, ...],
    output_dim: int,
    output_target_label: str,
    num_star_per_sample: int,
    num_sample_per_snapshot: int,
    drop_probability: float = 0.4,
    drop_ratio_range: tuple[float, float] = (0.1, 0.9),
    batch_size: int = 5120,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-4,
    huber_delta: float = 1.0,
    max_epochs: int = 50,
    seed: int = 42,
) -> tuple[pl.LightningModule, str]:
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise click.BadParameter(f"Unsupported model: {model_name}")

    model_info = MODEL_REGISTRY[model_name]
    defaults = model_info["defaults"]

    if unknown := set(model_hparams) - set(defaults):
        raise click.BadParameter(
            f"Unrecognized hparams for {model_name}: {unknown}. Valid: {set(defaults)}"
        )

    hparams = {**defaults, **model_hparams}
    hparams["input_dim"] = input_dim
    hparams["output_dim"] = output_dim

    match model_name:
        case "set_transformer":
            model_hparams = [
                f"hd{hparams['hidden_dim']}",
                f"nh{hparams['num_heads']}",
                f"ns{hparams['num_sabs']}",
                f"ohd{_fmt_value(hparams['output_hidden_dims'])}"
                if hparams["output_hidden_dims"] is not None
                else "no_ohd",
                f"d{_fmt_value(hparams['dropout'])}",
            ]
        case "deep_sets":
            model_hparams = [
                f"phi{_fmt_value(hparams['phi_hidden_dims'])}",
                f"rho{_fmt_value(hparams['rho_hidden_dims'])}",
                f"d{_fmt_value(hparams['dropout'])}",
                f"pool+{hparams['pooling']}",
            ]
        case "summary_stats":
            model_hparams = [
                f"h{_fmt_value(hparams['hidden_dims'])}",
                f"d{_fmt_value(hparams['dropout'])}",
            ]
        case _:
            model_hparams = []

    sampling_config = [
        f"nstar{num_star_per_sample}",
        f"nsnap{num_sample_per_snapshot}",
        f"dp{_fmt_value(drop_probability)}",
        f"dr{_fmt_value(drop_ratio_range[0])}_{_fmt_value(drop_ratio_range[1])}",
    ]
    training_config = [
        f"bs{batch_size}",
        f"lr{_fmt_value(learning_rate)}",
        f"wd{_fmt_value(weight_decay)}",
    ]
    version = "-".join(
        [f"from+{'+'.join(input_feature_labels)}", f"to+{output_target_label}"]
        + sampling_config
        + [model_name]
        + model_hparams
        + training_config
    )

    orchestrator = LightningRegressionOrchestrator(
        regressor=model_info["class"],
        hyperparameters=hparams,
        huber_delta=huber_delta,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_t_max=max_epochs,
        seed=seed,
    )

    return orchestrator, version


class TestParquetWriter(pl.Callback):
    def __init__(
        self,
        output_file: Path | str,
        merge_batch_rows: int = 65536,
    ) -> None:
        self._output_file = Path(output_file).resolve()
        self._output_file_part: Path | None = None

        self._part_writer: pq.ParquetWriter | None = None
        self._merge_batch_rows = int(merge_batch_rows)

    @staticmethod
    def _distributed_barrier(trainer: pl.Trainer) -> None:
        try:
            trainer.strategy.barrier()
            return
        except Exception:
            pass

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            local_rank = getattr(trainer, "local_rank", None)
            if torch.cuda.is_available() and local_rank is not None:
                try:
                    torch.distributed.barrier(device_ids=[local_rank])
                    return
                except TypeError:
                    pass

            torch.distributed.barrier()

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (
            torch.cuda.is_available()
            and getattr(trainer, "local_rank", None) is not None
        ):
            torch.cuda.set_device(trainer.local_rank)

        if trainer.is_global_zero:
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
            if self._output_file.is_file():
                self._output_file.unlink()

        self._distributed_barrier(trainer=trainer)

        if getattr(trainer, "world_size", 1) > 1:
            self._output_file_part = self._output_file.with_suffix(
                f".ddp-rank{trainer.global_rank}" + self._output_file.suffix
            )
        else:
            self._output_file_part = self._output_file

        if self._output_file_part.is_file():
            self._output_file_part.unlink()

        self._part_writer = None

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        predictions_scaled, targets_scaled, snapshot_ids, sample_ids = outputs

        data_module = trainer.datamodule

        table = pa.table(
            {
                "snapshot_id": (
                    snapshot_ids_np := snapshot_ids.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.int64)
                ),
                "sample_id": (
                    sample_ids_np := sample_ids.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.int64)
                ),
                "prediction_scaled": (
                    predictions_scaled_np := predictions_scaled.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                ),
                "target_scaled": (
                    targets_scaled_np := targets_scaled.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                ),
                "prediction_original": data_module.inverse_transform_targets(
                    predictions_scaled_np.reshape(-1, 1)
                ).reshape(-1),
                "target_original": data_module.inverse_transform_targets(
                    targets_scaled_np.reshape(-1, 1)
                ).reshape(-1),
            }
        )

        if self._part_writer is None:
            self._part_writer = pq.ParquetWriter(self._output_file_part, table.schema)
        self._part_writer.write_table(table)

        logger.debug(
            f"[Test batch {batch_idx}] Wrote {len(snapshot_ids_np)} snapshots totaling {len(sample_ids_np)} samples to {self._output_file_part}"
        )

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._part_writer is not None:
            self._part_writer.close()
            self._part_writer = None

        self._distributed_barrier(trainer=trainer)

        if trainer.is_global_zero and getattr(trainer, "world_size", 1) > 1:
            # merge all part files into ONE parquet (streaming)
            part_files = [
                self._output_file.with_suffix(
                    f".ddp-rank{rank}" + self._output_file.suffix
                )
                for rank in range(trainer.world_size)
            ]

            schema = pq.ParquetFile(part_files[0]).schema_arrow
            with pq.ParquetWriter(self._output_file, schema) as writer:
                for pf in part_files:
                    parquet_file = pq.ParquetFile(pf)
                    for batch in parquet_file.iter_batches(
                        batch_size=self._merge_batch_rows
                    ):
                        writer.write_table(
                            pa.Table.from_batches([batch], schema=schema)
                        )

            # cleanup
            for pf in part_files:
                try:
                    pf.unlink()
                except OSError:
                    pass

            logger.info(
                f"Concatenated test results to: {self._output_file} from {trainer.world_size} part files."
            )

        self._distributed_barrier(trainer=trainer)
        if trainer.is_global_zero:
            logger.info(f"Test results saved to: {self._output_file}")


@click.command()
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=42,
    help="Random seed.",
)
@click.option(
    "--dataset",
    "dataset_dir",
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Path to cache directory.",
)
@click.option(
    "--model",
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    required=True,
    help="Model architecture.",
)
@click.option(
    "--feature-keys",
    type=str,
    multiple=True,
    required=True,
    help="Feature keys. Example: --feature_keys x --feature_keys y",
)
@click.option(
    "--target-key",
    type=str,
    multiple=False,
    required=True,
    help="Target key. Example: --target_key time",
)
@click.option(
    "--hparam",
    "hparam_pairs",
    multiple=True,
    help="Model hparam. Example: --hparam hidden_dim=64",
)
@click.option(
    "--num-star-per-sample",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
    help="Number of stars per sample.",
)
@click.option(
    "--num-sample-per-snapshot",
    type=click.IntRange(min=1),
    default=16,
    show_default=True,
    help="Number of samples per snapshot.",
)
@click.option(
    "--drop-probability",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.6,
    show_default=True,
    help="Probability of dropping stars from samples if there are more stars than num_star_per_sample.",
)
@click.option(
    "--drop-ratio-range",
    type=click.Tuple([click.FloatRange(0.0, 1.0), click.FloatRange(0.0, 1.0)]),
    default=(0.1, 0.9),
    show_default=True,
    callback=lambda ctx, param, value: value
    if value[0] <= value[1]
    else (_ for _ in ()).throw(
        click.BadParameter(
            f"Expected lower bound <= upper bound for drop ratio range, got {value}"
        )
    ),
    help="Range of ratio of stars to drop when applying random star dropping.",
)
@click.option(
    "--num-workers",
    type=click.IntRange(min=0),
    default=16,
    show_default=True,
    help="Number of DataLoader workers.",
)
@click.option(
    "--pin-memory/--no-pin-memory",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to pin memory in DataLoader.",
)
@click.option(
    "-bs",
    "--batch-size",
    type=click.IntRange(min=1),
    default=20480,
    show_default=True,
    help="Batch size.",
)
@click.option(
    "-lr",
    "--learning-rate",
    "learning_rate",
    type=float,
    default=3e-3,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "-wd",
    "--weight-decay",
    "weight_decay",
    type=float,
    default=1e-4,
    show_default=True,
    help="Weight decay.",
)
@click.option(
    "--huber-delta",
    "huber_delta",
    type=float,
    default=1.0,
    show_default=True,
    help="Huber loss delta parameter.",
)
@click.option(
    "--max-epochs",
    "max_epochs",
    type=int,
    default=50,
    show_default=True,
    help="Maximum number of training epochs.",
)
@click.option(
    "--patience",
    type=int,
    default=10,
    show_default=True,
    help="Early stopping patience when no improvement in validation loss.",
)
def train(
    seed: int,
    dataset_dir: Path,
    model: str,
    feature_keys: tuple[str, ...],
    target_key: str,
    hparam_pairs: tuple[str, ...],
    num_star_per_sample: int,
    num_sample_per_snapshot: int,
    drop_probability: float,
    drop_ratio_range: tuple[float, float],
    num_workers: int,
    pin_memory: bool,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    huber_delta: float,
    max_epochs: int,
    patience: int,
) -> None:
    setup_logger(OUTPUT_BASE / "log" / "train.log")

    pl.seed_everything(seed, workers=True)
    parsed_hparams = _parse_hparams(hparam_pairs)

    logger.info("Training Configuration:")
    logger.info(f" - Feature Keys: {feature_keys}")
    logger.info(f" - Target Key: {target_key}")

    logger.info(f"Loading data from: {dataset_dir}")
    data_module = NBODY6DataModule(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        num_star_per_sample=num_star_per_sample,
        num_sample_per_snapshot=num_sample_per_snapshot,
        drop_probability=drop_probability,
        drop_ratio_range=drop_ratio_range,
        feature_keys=feature_keys,
        target_key=target_key,
    )
    data_module.setup()

    orchestrator, version_str = _build_model(
        model_name=model,
        input_dim=data_module.feature_dim,
        input_feature_labels=data_module.feature_keys,
        output_dim=data_module.target_dim,
        output_target_label=data_module.target_key,
        num_star_per_sample=num_star_per_sample,
        num_sample_per_snapshot=num_sample_per_snapshot,
        drop_probability=drop_probability,
        drop_ratio_range=drop_ratio_range,
        model_hparams=parsed_hparams,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        huber_delta=huber_delta,
        max_epochs=max_epochs,
        seed=seed,
    )
    logger.info(f"Model Version: {version_str}")

    output_dir = OUTPUT_BASE / "experiments" / version_str
    output_dir.mkdir(parents=True, exist_ok=True)

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(num_devices)]
        logger.info(f"Using GPU(s): count={num_devices}, names={names}")
    else:
        cpu_count = os.cpu_count()
        cpu_spec = platform.processor() or platform.machine() or "Unknown"
        logger.info(f"Using CPU: count={cpu_count}, spec={cpu_spec}")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_devices,
        strategy="ddp" if num_devices > 1 else "auto",
        max_epochs=max_epochs,
        gradient_clip_val=0.5,
        log_every_n_steps=100,
        deterministic=True,
        logger=CSVLogger(
            name="",
            version="",
            save_dir=output_dir,
        ),
        callbacks=[
            ModelCheckpoint(
                monitor="val_huber_loss",
                mode="min",
                save_top_k=1,
                dirpath=str(output_dir),
                filename=f"checkpoint-{version_str}-{{epoch:03d}}-{{val_huber_loss:.4e}}",
            ),
            EarlyStopping(
                monitor="val_huber_loss",
                patience=patience,
                mode="min",
                strict=True,
            ),
            LearningRateMonitor(
                logging_interval="epoch",
                log_momentum=True,
                log_weight_decay=True,
            ),
            TestParquetWriter(
                output_file=(
                    test_parquet_file := (
                        output_dir / f"{version_str}-test.parquet"
                    ).resolve()
                ),
            ),
        ],
    )

    if trainer.is_global_zero:
        logger.info("Model Architecture:\n%s", orchestrator.model)

    # resume from checkpoint if available
    if ckpt_files := sorted(
        output_dir.glob(f"checkpoint-{version_str}-*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    ):
        resume_ckpt = str(ckpt_files[-1])
        logger.info(f"Found checkpoint, resuming from: {resume_ckpt}")
        trainer.fit(orchestrator, datamodule=data_module, ckpt_path=resume_ckpt)
    else:
        trainer.fit(orchestrator, datamodule=data_module)

    trainer.test(orchestrator, datamodule=data_module, ckpt_path="best")

    # load the test results (Parquet) and print samples
    if trainer.is_global_zero:
        try:
            df = pd.read_parquet(test_parquet_file)
            logger.info("Test Results Samples:")
            logger.info("\n%s", df.sample(10).to_string(index=False))
        except Exception as e:
            logger.error(f"Failed to load test results from {test_parquet_file}: {e}")


if __name__ == "__main__":
    train()
