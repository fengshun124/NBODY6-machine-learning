import ast
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dataset.module import NBODY6DataModule
from model import DeepSetRegressor, SetTransformerRegressor, SummaryStatsRegressor
from orchestrator import LightningRegressionOrchestrator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils import OUTPUT_BASE, setup_logger

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")


# model and their default hyperparameters
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
    batch_size: int = 5120,
    learning_rate: float = 1e-4,
    weight_decay: float = 3e-3,
    huber_delta: float = 1.0,
    seed: int | None = None,
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

    train_param = [
        f"bs{batch_size}",
        f"lr{_fmt_value(learning_rate)}",
        f"wd{_fmt_value(weight_decay)}",
    ]
    version = "-".join(
        [f"from+{'+'.join(input_feature_labels)}", f"to+{output_target_label}"]
        + [model_name]
        + model_hparams
        + train_param
    )

    orchestrator = LightningRegressionOrchestrator(
        regressor=model_info["class"],
        hyperparameters=hparams,
        huber_delta=huber_delta,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )

    return orchestrator, version


class TestNPZWriter(pl.Callback):
    def __init__(
        self,
        output_file: Path | str,
    ) -> None:
        self._output_file = Path(output_file).resolve()

        self._sample_id_buffer = []
        self._prediction_original_buffer = []
        self._prediction_scaled_buffer = []
        self._target_original_buffer = []
        self._target_scaled_buffer = []

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._output_file.parent.mkdir(parents=True, exist_ok=True)

        self._sample_id_buffer.clear()
        self._prediction_original_buffer.clear()
        self._prediction_scaled_buffer.clear()
        self._target_original_buffer.clear()
        self._target_scaled_buffer.clear()

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        predictions_scaled, targets_scaled, sample_ids = outputs

        predictions_scaled_np = predictions_scaled.detach().cpu().numpy()
        targets_scaled_np = targets_scaled.detach().cpu().numpy()
        sample_ids_np = sample_ids.detach().cpu().numpy()

        data_module = trainer.datamodule
        predictions_original_np = data_module.inverse_transform_targets(
            predictions_scaled_np
        )
        targets_original_np = data_module.inverse_transform_targets(targets_scaled_np)

        self._sample_id_buffer.append(sample_ids_np)
        self._prediction_scaled_buffer.append(predictions_scaled_np)
        self._target_scaled_buffer.append(targets_scaled_np)
        self._prediction_original_buffer.append(predictions_original_np)
        self._target_original_buffer.append(targets_original_np)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        sample_ids = np.concatenate(self._sample_id_buffer, axis=0)
        predictions_scaled = np.concatenate(self._prediction_scaled_buffer, axis=0)
        targets_scaled = np.concatenate(self._target_scaled_buffer, axis=0)
        predictions_original = np.concatenate(self._prediction_original_buffer, axis=0)
        targets_original = np.concatenate(self._target_original_buffer, axis=0)

        np.savez_compressed(
            self._output_file,
            sample_id=sample_ids,
            prediction_scaled=predictions_scaled,
            target_scaled=targets_scaled,
            prediction_original=predictions_original,
            target_original=targets_original,
        )

        logger.info(f"Test results saved to: {self._output_file}")

        self._sample_id_buffer.clear()
        self._prediction_original_buffer.clear()
        self._prediction_scaled_buffer.clear()
        self._target_original_buffer.clear()
        self._target_scaled_buffer.clear()


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
    default=1e-4,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "-wd",
    "--weight-decay",
    "weight_decay",
    type=float,
    default=3e-3,
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
    default=5,
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
    num_workers: int,
    pin_memory: bool,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    huber_delta: float,
    max_epochs: int,
    patience: int,
) -> None:
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
        num_sample_per_snapshot=12,
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
        model_hparams=parsed_hparams,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        huber_delta=huber_delta,
        seed=seed,
    )
    logger.info(f"Model Version: {version_str}")

    output_dir = OUTPUT_BASE / "experiments" / version_str
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model Architecture:\n%s", orchestrator.model)

    trainer = pl.Trainer(
        accelerator="auto",
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
            TestNPZWriter(
                output_file=(
                    test_npz_file := (output_dir / f"{version_str}-test.npz").resolve()
                ),
            ),
        ],
    )

    trainer.fit(orchestrator, datamodule=data_module)
    trainer.test(orchestrator, datamodule=data_module, ckpt_path="best")

    # load the test results and print samples
    try:
        with np.load(test_npz_file) as npz:
            df = pd.DataFrame(
                {
                    "sample_id": npz["sample_id"],
                    "prediction_original": npz["prediction_original"].flatten(),
                    "target_original": npz["target_original"].flatten(),
                    "prediction_scaled": npz["prediction_scaled"].flatten(),
                    "target_scaled": npz["target_scaled"].flatten(),
                }
            )
        logger.info("Test Results Samples:")
        logger.info("\n%s", df.head(10).to_string(index=False))
    except Exception as e:
        logger.error(f"Failed to load test results from {test_npz_file}: {e}")


if __name__ == "__main__":
    train()
