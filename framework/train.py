from pathlib import Path

import pytorch_lightning as pl
import torch
from dataset.module import NBody6DataModule
from model.set_transformer import SetTransformerRegressor
from orchestrator import LightningRegressionOrchestrator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


def main() -> None:
    SEED = 42
    pl.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("medium")

    # data loading configuration
    CACHE_DIR = Path(__file__).parent.parent / "cache/9ef03baf1395"
    BATCH_SIZE = 20480
    NUM_WORKERS = 16

    # features and target keys
    TARGET_NAME = "total_mass_within_2x_r_tidal"
    FEATURE_NAMES = [
        # "lon_deg",
        # "lat_deg",
        # "pm_lon_coslat_mas_yr",
        # "pm_lat_mas_yr",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
    ]

    # model configuration
    HIDDEN_DIM = 8
    NUM_SABS = 4
    NUM_HEADS = 8
    OUTPUT_HIDDEN_DIMS = (4, 2)
    DROPOUT = 0.2

    # Training configuration
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 3e-3
    HUBER_DELTA = 1.0
    MAX_EPOCHS = 100

    # logging configuration
    LOG_DIR = Path(__file__).parent.parent / "logs"
    VERSION_STR = "nbody6-set_transformer-mass"

    # initialize data module
    print(f"Loading data from: {CACHE_DIR}")
    data_module = NBody6DataModule(
        cache_dir=CACHE_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        feature_keys=FEATURE_NAMES,
        target_key=TARGET_NAME,
    )

    # Setup to get dimensions
    data_module.setup()

    print(f"Feature keys: {data_module.feature_keys}")
    print(f"Target keys: {data_module.target_keys}")
    print(f"Input dim: {data_module.input_dim}")
    print(f"Output dim: {data_module.output_dim}")

    # ========================
    # Initialize Model
    # ========================
    model_hparams = {
        "input_dim": data_module.input_dim,
        "output_dim": data_module.output_dim,
        "hidden_dim": HIDDEN_DIM,
        "num_heads": NUM_HEADS,
        "num_sabs": NUM_SABS,
        "output_hidden_dims": OUTPUT_HIDDEN_DIMS,
        "dropout": DROPOUT,
    }

    orchestrator = LightningRegressionOrchestrator(
        regressor=SetTransformerRegressor,
        hyperparameters=model_hparams,
        huber_delta=HUBER_DELTA,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
    )

    print(f"\nModel architecture:\n{orchestrator.model}")

    # ========================
    # Setup Trainer
    # ========================
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=0.5,
        log_every_n_steps=100,
        deterministic=True,
        logger=CSVLogger(
            save_dir=str(LOG_DIR),
            name="training",
            version=VERSION_STR,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=str(LOG_DIR / "checkpoints"),
                filename=f"{VERSION_STR}-{{epoch:03d}}-{{val_huber_loss:.4e}}",
                monitor="val_huber_loss",
                mode="min",
                save_top_k=1,
            ),
            EarlyStopping(
                monitor="val_huber_loss",
                patience=5,
                mode="min",
                strict=True,
            ),
        ],
    )

    trainer.fit(orchestrator, datamodule=data_module)

    print("\n" + " RUNNING TEST ".center(60, "="))
    trainer.test(orchestrator, datamodule=data_module)

    print("\n" + " PREDICTION VS GROUND TRUTH (Original Scale) ".center(60, "="))

    # Get a batch from test set
    batch = next(iter(data_module.test_dataloader()))
    features, targets_norm, masks = [x.to(orchestrator.device) for x in batch]

    # Predict and denormalize
    orchestrator.eval()
    preds_all, t_all = [], []

    with torch.no_grad():
        for features, targets_norm, masks in data_module.test_dataloader():
            features = features.to(orchestrator.device)
            targets_norm = targets_norm.to(orchestrator.device)
            masks = masks.to(orchestrator.device)

            preds_norm = orchestrator(features, masks)

            preds_all.append(data_module.denormalize_targets(preds_norm).cpu())
            t_all.append(data_module.denormalize_targets(targets_norm).cpu())

    predictions_orig = torch.cat(preds_all, dim=0)
    targets_orig = torch.cat(t_all, dim=0)

    mae = (predictions_orig - targets_orig).abs().mean().item()
    rmse = ((predictions_orig - targets_orig) ** 2).mean().sqrt().item()
    rel = (
        (predictions_orig - targets_orig).abs() / (targets_orig.abs() + 1e-8)
    ).mean().item() * 100

    print("\nEVALUATION ON TEST SET (ORIGINAL SCALE):")
    # iterate over output dimensions
    for i in range(predictions_orig.shape[1]):
        mae_i = (predictions_orig[:, i] - targets_orig[:, i]).abs().mean().item()
        rmse_i = (
            ((predictions_orig[:, i] - targets_orig[:, i]) ** 2).mean().sqrt().item()
        )
        rel_i = (
            (predictions_orig[:, i] - targets_orig[:, i]).abs()
            / (targets_orig[:, i].abs() + 1e-8)
        ).mean().item() * 100

        print(f" Output dim {i}: MAE={mae_i:.4f} RMSE={rmse_i:.4f} RelErr={rel_i:.2f}%")

    print(f"TEST MAE={mae:.4f} RMSE={rmse:.4f} RelErr={rel:.2f}%")


if __name__ == "__main__":
    main()
