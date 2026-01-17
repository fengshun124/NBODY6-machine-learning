import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from dataset.scaler import ArrayScalerBundle
from dataset.shard import Shard
from torch.utils.data import DataLoader, Dataset


def _worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    dataset = worker_info.dataset
    base_seed = getattr(dataset, "_base_seed", 0)
    seed = base_seed + worker_id

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


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

        # load shard
        self._shard = (
            shard if isinstance(shard, Shard) else Shard.from_npz(Path(shard).resolve())
        )

        # validate and cache feature keys (store as tuple for consistency)
        self._feature_keys = (
            (feature_keys,) if isinstance(feature_keys, str) else tuple(feature_keys)
        )
        if not all(key in self._shard.feature_keys for key in self._feature_keys):
            raise ValueError(
                f"Expected feature_keys to be a subset of {self._shard.feature_keys}, got {feature_keys}."
            )
        self._feature_col_indices = [
            self._shard.feature_keys.index(k) for k in self._feature_keys
        ]

        # validate and cache target key
        self._target_key = target_key
        if target_key not in self._shard.target_keys:
            raise ValueError(
                f"Expected target_key to be one of {self._shard.target_keys}, got {target_key}."
            )
        self._target_col_idx = self._shard.target_keys.index(target_key)

        # Validate sampling parameters
        if num_star_per_sample <= 0 or not isinstance(num_star_per_sample, int):
            raise ValueError(
                f"Expected positive integer for 'num_star_per_sample', got {num_star_per_sample}."
            )
        self._num_star_per_sample = num_star_per_sample

        if num_sample_per_snapshot <= 0 or not isinstance(num_sample_per_snapshot, int):
            raise ValueError(
                f"Expected positive integer for 'num_sample_per_snapshot', got {num_sample_per_snapshot}."
            )
        self._num_sample_per_snapshot = num_sample_per_snapshot

        # validate augmentation parameters
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

        # store seed for determinism
        self._seed = seed
        self._base_seed = seed
        self._seed_seq = np.random.SeedSequence(seed)

    def __len__(self) -> int:
        return self.n_snapshots * self._num_sample_per_snapshot

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        sample_idx = int(index % self._num_sample_per_snapshot)
        snapshot_idx = int(index // self._num_sample_per_snapshot)

        # create deterministic RNG for this specific sample
        rng = np.random.default_rng(
            np.random.SeedSequence(
                entropy=self._seed_seq.entropy,
                spawn_key=(
                    self._seed_seq.spawn_key[0] if self._seed_seq.spawn_key else 0,
                    snapshot_idx,
                    sample_idx,
                ),
            )
        )

        # get data from shard
        full_features, full_targets = self._shard[snapshot_idx]
        features = full_features[:, self._feature_col_indices]
        target = full_targets[self._target_col_idx]

        n_stars = features.shape[0]

        # initialize output buffers
        sampled_features = np.zeros(
            (self._num_star_per_sample, self.feature_dim), dtype=np.float32
        )
        valid_mask = np.zeros(self._num_star_per_sample, dtype=bool)

        if n_stars >= self._num_star_per_sample:
            # enough stars: sample without replacement
            chosen = rng.choice(
                n_stars,
                size=self._num_star_per_sample,
                replace=False,
            )
            sampled_features[:] = features[chosen]
            valid_mask[:] = True

            # apply optional dropout augmentation on sampled data
            if rng.random() < self._drop_probability:
                drop_ratio = rng.uniform(*self._drop_ratio_range)
                drop_count = int(drop_ratio * self._num_star_per_sample)
                drop_count = max(0, min(drop_count, self._num_star_per_sample - 1))

                if drop_count > 0:
                    drop_indices = rng.choice(
                        self._num_star_per_sample,
                        size=drop_count,
                        replace=False,
                    )
                    sampled_features[drop_indices] = 0
                    valid_mask[drop_indices] = False
        else:
            # insufficient stars: use all available + padding
            sampled_features[:n_stars] = features
            valid_mask[:n_stars] = True

        return (
            sampled_features,
            np.array([target], dtype=np.float32),
            valid_mask,
            index,
        )

    @property
    def shard(self) -> Shard:
        return self._shard

    @property
    def n_snapshots(self) -> int:
        return len(self._shard)

    @property
    def feature_keys(self) -> tuple[str, ...]:
        return self._feature_keys

    @property
    def feature_dim(self) -> int:
        return len(self._feature_keys)

    @property
    def target_key(self) -> str:
        return self._target_key


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
        self._seed = seed

        # DataLoader configuration
        self.loader_params = {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory),
        }

        # worker settings for multi-worker loading
        if self.loader_params["num_workers"] > 0:
            self.loader_params.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                    "worker_init_fn": _worker_init_fn,
                }
            )

        # Dataset configuration
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

        # lazy-loaded datasets
        self.train_dataset: NBODY6SnapshotDataset | None = None
        self.val_dataset: NBODY6SnapshotDataset | None = None
        self.test_dataset: NBODY6SnapshotDataset | None = None

        # scalers
        self._feature_scaler_bundle: ArrayScalerBundle | None = None
        self._target_scaler_bundle: ArrayScalerBundle | None = None

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

        # initialize train/val datasets
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

        # initialize test dataset
        if stage in (None, "test", "predict"):
            if self.test_dataset is None:
                self.test_dataset = NBODY6SnapshotDataset(
                    shard=self._dataset_dir / "scaled-test-shard.npz",
                    **self.dataset_params,
                )

    @staticmethod
    def _collate_fn(
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, targets, masks, sample_ids = zip(*batch)
        return (
            torch.from_numpy(np.stack(features)).float(),
            torch.from_numpy(np.stack(targets)).float().squeeze(-1),
            torch.from_numpy(np.stack(masks)).bool(),
            torch.as_tensor(sample_ids).long(),
        )

    def _make_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=shuffle,
                seed=self._seed,
            )

        dl_shuffle = False if sampler is not None else shuffle

        gen = torch.Generator()
        gen.manual_seed(self._seed)

        return DataLoader(
            dataset,
            sampler=sampler,
            shuffle=dl_shuffle,
            collate_fn=self._collate_fn,
            generator=gen,
            **self.loader_params,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return self._make_dataloader(self.train_dataset, shuffle=self.shuffle_train)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("validate")
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return self._make_dataloader(self.test_dataset, shuffle=False)

    def inverse_transform_features(
        self,
        features: torch.Tensor | np.ndarray,
    ) -> np.ndarray:
        if not self._feature_scaler_bundle:
            raise RuntimeError("Scalers not loaded. Call setup() first.")

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
            raise RuntimeError("Scalers not loaded. Call setup() first.")

        arr = (
            targets.detach().cpu().numpy()
            if isinstance(targets, torch.Tensor)
            else np.asarray(targets)
        )
        return self._target_scaler_bundle.inverse_transform(
            data=arr.reshape(-1, 1),
            keys=(self._target_key,),
        ).squeeze(-1)
