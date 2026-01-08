from typing import Any, Type

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


class LightningRegressionOrchestrator(pl.LightningModule):
    def __init__(
        self,
        regressor: nn.Module | Type[nn.Module],
        hyperparameters: dict[str, Any],
        huber_delta: float = 1.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "model_hparams": hyperparameters,
                "seed": seed,
            }
        )

        if isinstance(regressor, nn.Module):
            self.model = regressor
        else:
            self.model = regressor(**hyperparameters)

        self.loss_fn = nn.HuberLoss(delta=huber_delta)

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.train_mse = torchmetrics.MeanSquaredError()

        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_mse = torchmetrics.MeanSquaredError()

        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()

    def forward(
        self, X: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.model(X, mask)

    def _evaluation(
        self, batch, stage: str
    ) -> torch.Tensor:
        inputs, targets, mask = batch
        mask = mask.bool()

        preds = self(inputs, mask).view(-1)
        targets = targets.view(-1).to(preds)

        loss = self.loss_fn(preds, targets)

        self.log(
            f"{stage}_huber_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        for metric_key in ["mae", "mse"]:
            metric = getattr(self, f"{stage}_{metric_key}")
            metric(preds, targets)
            self.log(
                f"{stage}_{metric_key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._evaluation(batch=batch, stage="train")

    def validation_step(self, batch, batch_idx) -> None:
        self._evaluation(batch=batch, stage="val")

    def test_step(self, batch, batch_idx) -> None:
        self._evaluation(batch=batch, stage="test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_huber_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# FOR TESTING PURPOSES ONLY!!!
class ToySetDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        max_set_size: int = 60,
        input_dim: int = 6,
        y_mean: float | None = None,
        y_std: float | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.max_set_size = max_set_size
        self.input_dim = input_dim

        # (samples, set_size, input_dim)
        self.data = torch.randn(num_samples, max_set_size, input_dim)

        self.set_sizes = torch.randint(1, max_set_size + 1, (num_samples,))
        self.masks = torch.arange(max_set_size).expand(
            num_samples, max_set_size
        ) < self.set_sizes.unsqueeze(1)

        # mock targets: mean of valid elements in the first dimension
        self.targets = torch.zeros(num_samples, 1)
        for i in range(num_samples):
            # calculate target using ONLY valid elements
            valid_elements = self.data[i, : self.set_sizes[i], 0]
            self.targets[i] = valid_elements.mean() * 10.0 + 5.0

        # store normalization parameters or compute from this dataset
        if y_mean is not None and y_std is not None:
            self.y_mean = y_mean
            self.y_std = y_std
        else:
            self.y_mean = self.targets.mean()
            self.y_std = self.targets.std()

        # normalize targets
        self.norm_targets = (self.targets - self.y_mean) / (self.y_std + 1e-6)

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.y_std + self.y_mean

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx], self.norm_targets[idx], self.masks[idx]


class ToySetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_samples: int = 1000,
        max_set_size: int = 96,
        input_dim: int = 6,
        batch_size: int = 32,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_set_size = max_set_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.split = train_val_test_split
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        # calculate split sizes
        train_len = int(self.num_samples * self.split[0])
        val_len = int(self.num_samples * self.split[1])
        test_len = self.num_samples - train_len - val_len

        # create training dataset first to compute normalization stats
        train_dataset = ToySetDataset(
            num_samples=train_len,
            max_set_size=self.max_set_size,
            input_dim=self.input_dim,
        )

        # initialize normalization parameters from training set only
        y_mean = train_dataset.y_mean
        y_std = train_dataset.y_std

        # create val and test datasets with training set's normalization
        val_dataset = ToySetDataset(
            num_samples=val_len,
            max_set_size=self.max_set_size,
            input_dim=self.input_dim,
            y_mean=y_mean,
            y_std=y_std,
        )

        test_dataset = ToySetDataset(
            num_samples=test_len,
            max_set_size=self.max_set_size,
            input_dim=self.input_dim,
            y_mean=y_mean,
            y_std=y_std,
        )

        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = test_dataset

        # store the full dataset reference for denormalization
        self.y_mean = y_mean
        self.y_std = y_std

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.y_std + self.y_mean


if __name__ == "__main__":
    from pytorch_lightning.loggers import CSVLogger

    try:
        from framework.model.set_transformer import SetTransformerRegressor
    except ImportError:
        from model.set_transformer import SetTransformerRegressor

    # set seed for reproducibility
    SEED = 42
    pl.seed_everything(SEED, workers=True)

    toy_data_module = ToySetDataModule(
        num_samples=2000,
        max_set_size=96,
        input_dim=4,
        num_workers=4,
        batch_size=100,
        seed=SEED,
    )

    orchestrator = LightningRegressionOrchestrator(
        regressor=SetTransformerRegressor,
        hyperparameters={
            "input_dim": 4,
            "output_dim": 1,
            "hidden_dim": 16,
            "num_heads": 4,
            "num_sabs": 2,
            "output_hidden_dims": (8,),
        },
        learning_rate=1e-3,
        seed=SEED,
    )

    # train and test model
    version_str = "toy-set_transformer"
    trainer = pl.Trainer(
        accelerator="auto",
        log_every_n_steps=10,
        max_epochs=300,
        deterministic=True,
        logger=CSVLogger(
            save_dir="/tmp",
            name="logs",
            version=version_str,
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="/tmp/checkpoints",
                filename=f"{version_str}-{{epoch:03d}}-{{val_huber_loss:.4e}}",
                monitor="val_huber_loss",
                mode="min",
                save_top_k=1,
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_huber_loss",
                patience=5,
                mode="min",
                strict=True,
            ),
        ],
    )
    trainer.fit(orchestrator, datamodule=toy_data_module)
    trainer.test(orchestrator, datamodule=toy_data_module)

    # map back a batch of normalized targets to original scale
    toy_data_module.setup()
    test_loader = toy_data_module.test_dataloader()
    dummy_batch = next(iter(test_loader))
    x, y_norm, mask = dummy_batch

    orchestrator.eval()
    with torch.no_grad():
        y_hat_norm = orchestrator(x, mask)

    y_hat_real = toy_data_module.denormalize(y_hat_norm)
    y_real = toy_data_module.denormalize(y_norm)

    print("\n" + f" TEST RESULT (Batch size={len(y_hat_real)}) ".center(60, "="))
    print(f"{'Index':>8} {'Prediction':>15} {'Target':>15} {'Error':>15}")
    print("-" * 60)
    for i in range(len(y_hat_real)):
        pred = y_hat_real[i].item()
        target = y_real[i].item()
        error = pred - target
        print(f"{i:>8} {pred:>15.4f} {target:>15.4f} {error:>15.4f}")

    mae = torch.abs(y_hat_real - y_real).mean().item()
    print("-" * 60)
    print(f"Mean Absolute Error: {mae:.4f}")
    print(" END OF TEST ".center(60, "="))
