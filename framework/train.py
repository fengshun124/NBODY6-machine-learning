import ast
import logging
from pathlib import Path

import click
import pytorch_lightning as pl
import torch
from dataset.module import NBody6DataModule
from model.deep_set import DeepSetRegressor
from model.set_transformer import SetTransformerRegressor
from model.summary_stats import SummaryStatsRegressor
from orchestrator import LightningRegressionOrchestrator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

torch.set_float32_matmul_precision("medium")

logger = logging.getLogger(__name__)


def setup_logger(log_file: str) -> None:
    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(processName)s][%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
            handlers=[logging.StreamHandler()],
            force=True,
        )
        return

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(processName)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
        force=True,
    )


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
    output_dim: int,
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
            ]
        case "deep_sets":
            model_hparams = [
                f"phi{_fmt_value(hparams['phi_hidden_dims'])}",
                f"rho{_fmt_value(hparams['rho_hidden_dims'])}",
            ]
        case "summary_stats":
            model_hparams = [f"h{_fmt_value(hparams['hidden_dims'])}"]
        case _:
            model_hparams = []

    train_param = [
        f"bs{batch_size}",
        f"lr{_fmt_value(learning_rate)}",
        f"wd{_fmt_value(weight_decay)}",
    ]
    version = "-".join([model_name] + model_hparams + train_param)

    orchestrator = LightningRegressionOrchestrator(
        regressor=model_info["class"],
        hyperparameters=hparams,
        huber_delta=huber_delta,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )

    return orchestrator, version


@click.command()
@click.option("--seed", default=42, help="Random seed.")
@click.option(
    "--data",
    type=click.Path(exists=True),
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
@click.option("--target-key", type=str, required=True, help="Target key.")
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
)
@click.option("--pin-memory/--no-pin-memory", default=True)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=20480,
    show_default=True,
)
@click.option(
    "-lr",
    "--learning-rate",
    "learning_rate",
    type=float,
    default=1e-4,
    show_default=True,
)
@click.option(
    "-wd", "--weight-decay", "weight_decay", type=float, default=3e-3, show_default=True
)
@click.option(
    "--huber-delta", "huber_delta", type=float, default=1.0, show_default=True
)
@click.option("--max-epochs", "max_epochs", type=int, default=50, show_default=True)
@click.option("--patience", type=int, default=5, show_default=True)
@click.option("--log-dir", type=click.Path(), default=None)
def train(
    seed: int,
    data: str | Path,
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
    log_dir: str | None,
) -> None:
    base_log_dir = Path(log_dir) if log_dir else Path(__file__).parent / "logs"
    base_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(base_log_dir / "train.log")
    setup_logger(log_file)

    pl.seed_everything(seed, workers=True)
    parsed_hparams = _parse_hparams(hparam_pairs)

    logger.info("Training Configuration:")
    logger.info(f" - Feature Keys: {feature_keys}")
    logger.info(f" - Target Key: {target_key}")

    cache_dir = Path(data)
    logger.info(f"Loading data from: {cache_dir}")
    data_module = NBody6DataModule(
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        feature_keys=feature_keys,
        target_key=target_key,
    )
    data_module.setup()
    # print data info

    orchestrator, version = _build_model(
        model_name=model,
        input_dim=data_module.input_dim,
        output_dim=data_module.output_dim,
        model_hparams=parsed_hparams,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        huber_delta=huber_delta,
        seed=seed,
    )

    logger.info("Model architecture:\n%s", orchestrator.model)

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=max_epochs,
        gradient_clip_val=0.5,
        log_every_n_steps=100,
        deterministic=True,
        logger=CSVLogger(name="training", version=version, save_dir=base_log_dir),
        callbacks=[
            ModelCheckpoint(
                monitor="val_huber_loss",
                mode="min",
                save_top_k=1,
                dirpath=str(base_log_dir / "checkpoints"),
                filename=f"{version}-{{epoch:03d}}-{{val_huber_loss:.4e}}",
            ),
            EarlyStopping(
                monitor="val_huber_loss",
                patience=patience,
                mode="min",
                strict=True,
            ),
        ],
    )

    trainer.fit(orchestrator, datamodule=data_module)
    trainer.test(orchestrator, datamodule=data_module)

    # evaluation
    try:
        orchestrator.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for feats, tgt_norm, masks in data_module.test_dataloader():
                feats = feats.to(orchestrator.device)
                tgt_norm = tgt_norm.to(orchestrator.device)
                masks = masks.to(orchestrator.device)

                pred_norm = orchestrator(feats, masks)
                preds_all.append(data_module.denormalize_targets(pred_norm).cpu())
                targets_all.append(data_module.denormalize_targets(tgt_norm).cpu())

        predictions = torch.cat(preds_all, dim=0)
        targets = torch.cat(targets_all, dim=0)

        mae = (predictions - targets).abs().mean().item()
        rmse = ((predictions - targets) ** 2).mean().sqrt().item()
        rel = (
            (predictions - targets).abs() / (targets.abs() + 1e-8)
        ).mean().item() * 100

        logger.info("=" * 60)
        logger.info("FULL TEST SET EVALUATION (ORIGINAL SCALE)")
        logger.info("=" * 60)
        logger.info(f"Total test samples: {len(predictions)}")

        for i in range(predictions.shape[1]):
            mae_i = (predictions[:, i] - targets[:, i]).abs().mean().item()
            rmse_i = ((predictions[:, i] - targets[:, i]) ** 2).mean().sqrt().item()
            rel_i = (
                (predictions[:, i] - targets[:, i]).abs() / (targets[:, i].abs() + 1e-8)
            ).mean().item() * 100
            logger.info(
                f" Output dim {i}: MAE={mae_i:.4f} RMSE={rmse_i:.4f} RelErr={rel_i:.2f}%"
            )

        logger.info("-" * 60)
        logger.info(f"Overall: MAE={mae:.4f} RMSE={rmse:.4f} RelErr={rel:.2f}%")
        logger.info("=" * 60)

        # report some of the samples for manual inspection
        n_samples = min(20, len(predictions))
        if n_samples > 0:
            indices = torch.randperm(len(predictions))[:n_samples].sort()[0]
            print("\n" + f" RANDOM TEST SAMPLES (n={n_samples}) ".center(60, "="))
            print(f"{'Index':>8} {'Prediction':>15} {'Target':>15} {'Error':>15}")
            print("-" * 60)

            for idx in indices:
                p = (
                    predictions[idx, 0].item()
                    if predictions.shape[1] > 0
                    else predictions[idx].item()
                )
                t = (
                    targets[idx, 0].item()
                    if targets.shape[1] > 0
                    else targets[idx].item()
                )
                print(f"{idx.item():>8} {p:>15.4f} {t:>15.4f} {p - t:>15.4f}")

            sample_mae = (
                torch.abs(predictions[indices] - targets[indices]).mean().item()
            )
            print("-" * 60)
            print(f"Sample Mean Absolute Error: {sample_mae:.4f}")
            print(" END OF RANDOM SAMPLES ".center(60, "="))

    except Exception as e:
        logger.error(f"Error during test evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    train()
