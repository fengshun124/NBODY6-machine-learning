from pathlib import Path
from typing import Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from dataset.scaler import ArrayScalerBundle
from dataset.shard import Shard
from torch.utils.data import DataLoader, Dataset


class NBODY6SnapshotDataset(Dataset):
    def __init__(
        self,
        shard: Shard | Path | str,
        feature_keys: Sequence[str] | str,
        target_key: str,
        num_star_per_sample: int = 96,
        num_sample_per_snapshot: int = 10,
        seed: int = 42,
        # data augmentation for snapshots containing sufficient stars
        drop_probability: float = 0.4,
        drop_ratio_range: tuple[float, float] = (0.1, 0.9),
    ) -> None:
        super().__init__()
        # validate and store shard
        self._shard = (
            shard if isinstance(shard, Shard) else Shard.from_npz(Path(shard).resolve())
        )

        # validate and cache feature column indices
        self._feature_keys = (
            [feature_keys] if isinstance(feature_keys, str) else list(feature_keys)
        )
        if not all(key in self._shard.feature_keys for key in self._feature_keys):
            raise ValueError(
                f"Expected feature_keys to be a subset of {self._shard.feature_keys}, got {feature_keys}."
            )
        self._feature_col_indices = [
            self._shard.feature_keys.index(k) for k in self._feature_keys
        ]
        # validate and cache target column index
        self._target_key = target_key
        if self._target_key not in self._shard.target_keys:
            raise ValueError(
                f"Expected target_key to be one of {self._shard.target_keys}, got {target_key}."
            )
        self._target_keys = [self._target_key]
        self._target_col_idx = [self._shard.target_keys.index(self._target_key)]

        # validate num_star_per_sample and num_sample_per_snapshot
        if num_star_per_sample <= 0 or not isinstance(num_star_per_sample, int):
            raise ValueError(
                f"Expected positive 'int' for 'num_star_per_sample', got {num_star_per_sample}."
            )
        self._num_star_per_sample = num_star_per_sample
        if num_sample_per_snapshot <= 0 or not isinstance(num_sample_per_snapshot, int):
            raise ValueError(
                f"Expected positive 'int' for 'num_sample_per_snapshot', got {num_sample_per_snapshot}."
            )
        self._num_sample_per_snapshot = num_sample_per_snapshot

        self._seed = seed

        # validate data augmentation parameters
        if not (0.0 <= drop_probability <= 1.0):
            raise ValueError(
                f"Expected 'drop_probability' to be in [0.0, 1.0], got {drop_probability}."
            )
        self._drop_probability = drop_probability
        if not (0.0 <= drop_ratio_range[0] < drop_ratio_range[1] <= 1.0):
            raise ValueError(
                f"Expected 'drop_ratio_range' to be in [0.0, 1.0] with lower < upper, got {drop_ratio_range}."
            )
        self._drop_ratio_range = drop_ratio_range

        # initialize random number generator
        self._seed_seq = np.random.SeedSequence(
            [seed, self._num_star_per_sample, self._num_sample_per_snapshot]
        )

    def __repr__(self):
        return (
            f"{__class__.__name__}(shard={self._shard}, "
            f"feature_keys={self._feature_keys}, "
            f"target_keys={self._target_keys}, "
            f"sample_config={self.sample_params}, "
            f"augment_config={self.augment_params}"
            ")"
        )

    def __len__(self) -> int:
        return self.n_snapshots * self._num_sample_per_snapshot

    def __getitem__(self, index):
        sample_idx = int(index % self._num_sample_per_snapshot)
        snapshot_idx = int(index // self._num_sample_per_snapshot)

        # deterministic RNG per sample
        rng = np.random.default_rng(
            np.random.SeedSequence(
                entropy=self._seed_seq.entropy,
                spawn_key=(
                    *self._seed_seq.spawn_key,
                    snapshot_idx,
                    sample_idx,
                ),
            )
        )

        # extract and filter data using cached indices
        full_features, full_targets = self._shard[snapshot_idx]
        features = full_features[:, self._feature_col_indices]
        targets = np.asarray([full_targets[i] for i in self._target_col_idx])

        n_stars = features.shape[0]

        # prepare output buffers (padded positions explicitly remain zero)
        sampled_features = np.zeros(
            (self._num_star_per_sample, self.feature_dim), dtype=features.dtype
        )
        valid_mask = np.zeros((self._num_star_per_sample,), dtype=bool)

        # shrink pool via optional augmentation
        if n_stars >= self._num_star_per_sample:
            pool = np.arange(n_stars)
            if rng.random() < self._drop_probability:
                drop_ratio = rng.uniform(*self._drop_ratio_range)
                keep_count = max(1, int((1.0 - drop_ratio) * n_stars))
                pool = rng.choice(pool, size=keep_count, replace=False)
            # sample with replacement only if needed to reach required size
            if len(pool) >= self._num_star_per_sample:
                chosen = rng.choice(
                    pool,
                    size=self._num_star_per_sample,
                    replace=False,
                )
            else:
                extra = rng.choice(
                    pool,
                    size=self._num_star_per_sample - len(pool),
                    replace=True,
                )
                chosen = np.concatenate([pool, extra])

            sampled_features[:] = features[chosen]
            valid_mask[:] = True
        # insufficient stars: take all, remaining positions stay zero-padded
        else:
            sampled_features[:n_stars] = features
            valid_mask[:n_stars] = True

        return sampled_features, targets, valid_mask

    @property
    def shard(self) -> Shard:
        return self._shard

    @property
    def n_snapshots(self) -> int:
        return len(self._shard)

    @property
    def feature_keys(self) -> list[str]:
        return self._feature_keys

    @property
    def feature_dim(self) -> int:
        return len(self._feature_keys)

    @property
    def target_keys(self) -> list[str]:
        return self._target_keys

    @property
    def target_dim(self) -> int:
        return len(self._target_keys)

    @property
    def target_key(self) -> str:
        return self._target_key

    @property
    def sample_params(self) -> dict:
        return {
            "num_star_per_sample": self._num_star_per_sample,
            "num_sample_per_snapshot": self._num_sample_per_snapshot,
            "seed": self._seed,
        }

    @property
    def augment_params(self) -> dict:
        return {
            "drop_probability": self._drop_probability,
            "drop_ratio_range": self._drop_ratio_range,
            "seed": self._seed,
        }


class NBODY6DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: Path | str,
        feature_keys: Sequence[str] | str,
        target_key: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        num_star_per_sample: int = 96,
        num_sample_per_snapshot: int = 10,
        drop_probability: float = 0.4,
        drop_ratio_range: tuple[float, float] = (0.1, 0.9),
        shuffle_train: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._dataset_dir = Path(dataset_dir)
        self._feature_keys = (
            tuple(feature_keys)
            if not isinstance(feature_keys, str)
            else (feature_keys,)
        )

        self._target_key = target_key

        self.loader_params = {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory),
        }
        self.dataset_params = {
            "feature_keys": self._feature_keys,
            "target_key": self._target_key,
            "num_star_per_sample": int(num_star_per_sample),
            "num_sample_per_snapshot": int(num_sample_per_snapshot),
            "drop_probability": float(drop_probability),
            "drop_ratio_range": drop_ratio_range,
            "seed": int(seed),
        }
        self.shuffle_train = bool(shuffle_train)

        self.train_dataset: NBODY6SnapshotDataset | None = None
        self.val_dataset: NBODY6SnapshotDataset | None = None
        self.test_dataset: NBODY6SnapshotDataset | None = None

        self._feature_scaler_bundle: ArrayScalerBundle | None = None
        self._target_scaler_bundle: ArrayScalerBundle | None = None

    def __repr__(self):
        return super().__repr__()

    @property
    def feature_keys(self) -> tuple[str, ...]:
        return self._feature_keys

    @property
    def feature_dim(self) -> int:
        return len(self._feature_keys)

    @property
    def target_key(self) -> str:
        return self._target_key

    @property
    def target_dim(self) -> int:
        return 1

    @property
    def feature_scaler_bundle(self) -> ArrayScalerBundle | None:
        return self._feature_scaler_bundle

    @property
    def target_scaler_bundle(self) -> ArrayScalerBundle | None:
        return self._target_scaler_bundle

    def setup(self, stage: str | None = None) -> None:
        # load scalers
        self._feature_scaler_bundle = ArrayScalerBundle.from_joblib(
            self._dataset_dir / "feature_scaler_bundle.joblib"
        )
        self._target_scaler_bundle = ArrayScalerBundle.from_joblib(
            self._dataset_dir / "target_scaler_bundle.joblib"
        )

        # initialize datasets
        if stage in (None, "fit", "validate"):
            if self.train_dataset is None:
                self.train_dataset = NBODY6SnapshotDataset(
                    shard=self._dataset_dir / "scaled-train-shard.npz",
                    **self.dataset_params,
                )
            if self.val_dataset is None:
                self.val_dataset = NBODY6SnapshotDataset(
                    shard=self._dataset_dir / "scaled-val-shard.npz",
                    **self.dataset_params,
                )

        if stage in (None, "test", "predict") and self.test_dataset is None:
            self.test_dataset = NBODY6SnapshotDataset(
                shard=self._dataset_dir / "scaled-test-shard.npz",
                **self.dataset_params,
            )

    @staticmethod
    def _collate_fn(
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, targets, masks = zip(*batch)
        return (
            torch.from_numpy(np.stack(features)).float(),
            torch.from_numpy(np.stack(targets)).float(),
            torch.from_numpy(np.stack(masks)).bool(),
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            shuffle=self.shuffle_train,
            collate_fn=self._collate_fn,
            **self.loader_params,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("validate")
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            collate_fn=self._collate_fn,
            **self.loader_params,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            collate_fn=self._collate_fn,
            **self.loader_params,
        )

    def inverse_transform_features(
        self,
        features: torch.Tensor | np.ndarray,
    ) -> np.ndarray:
        if not self._feature_scaler_bundle:
            raise RuntimeError("Scalers were not loaded. Call `setup()` first.")

        arr = (
            features.detach().cpu().numpy()
            if isinstance(features, torch.Tensor)
            else np.asarray(features)
        )

        b, s, f = arr.shape
        flat = self._feature_scaler_bundle.inverse_transform(
            data=arr.reshape(-1, f),
            keys=self._feature_keys,
        )
        return flat.reshape(b, s, f)

    def inverse_transform_targets(
        self,
        targets: torch.Tensor | np.ndarray,
    ) -> np.ndarray:
        if not self._target_scaler_bundle:
            raise RuntimeError("Scalers were not loaded. Call `setup()` first.")

        arr = (
            targets.detach().cpu().numpy()
            if isinstance(targets, torch.Tensor)
            else np.asarray(targets)
        )
        return self._target_scaler_bundle.inverse_transform(
            data=arr,
            keys=(self._target_key,),
        )
