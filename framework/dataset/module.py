from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pytorch_lightning as pl
import torch
from dataset.normalizer import Normalizer
from torch.utils.data import DataLoader, Dataset


class NBody6Dataset(Dataset):
    def __init__(
        self,
        features: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        masks: Union[np.ndarray, torch.Tensor],
        feature_indices: Optional[List[int]] = None,
        target_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        # convert to tensors if needed
        self._features = (
            torch.from_numpy(features).float()
            if isinstance(features, np.ndarray)
            else features.float()
        )
        self._targets = (
            torch.from_numpy(targets).float()
            if isinstance(targets, np.ndarray)
            else targets.float()
        )
        self._masks = (
            torch.from_numpy(masks).bool()
            if isinstance(masks, np.ndarray)
            else masks.bool()
        )

        self._feature_indices = feature_indices
        if feature_indices is not None:
            self._features = self._features[:, :, feature_indices]

        self._target_idx = target_idx
        self._validate(target_idx=target_idx)

    def _validate(self, target_idx: Optional[int] = None) -> None:
        # validate shapes
        assert self._features.dim() == 3, (
            f"Expected 3D features, got {self._features.dim()}D"
        )
        assert self._targets.dim() == 2, (
            f"Expected 2D targets, got {self._targets.dim()}D"
        )
        assert self._masks.dim() == 2, f"Expected 2D masks, got {self._masks.dim()}D"

        n_samples = len(self._features)
        assert len(self._targets) == n_samples, (
            f"Mismatched samples: features={n_samples}, targets={len(self._targets)}"
        )
        assert len(self._masks) == n_samples, (
            f"Mismatched samples: features={n_samples}, masks={len(self._masks)}"
        )
        assert self._masks.shape[1] == self._features.shape[1], (
            f"Mismatched set size: features={self._features.shape[1]}, masks={self._masks.shape[1]}"
        )

        if target_idx is not None:
            assert 0 <= target_idx < self._targets.shape[1], (
                f"target_idx {target_idx} out of range for {self._targets.shape[1]} targets"
            )

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features[idx]
        masks = self._masks[idx]

        if self._target_idx is not None:
            targets = self._targets[idx, self._target_idx : self._target_idx + 1]
        else:
            targets = self._targets[idx]

        return features, targets, masks

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @property
    def targets(self) -> torch.Tensor:
        return self._targets

    @property
    def masks(self) -> torch.Tensor:
        return self._masks

    @property
    def n_samples(self) -> int:
        return len(self._features)

    @property
    def n_stars(self) -> int:
        return self._features.shape[1]

    @property
    def n_features(self) -> int:
        return self._features.shape[2]

    @property
    def n_targets(self) -> int:
        if self._target_idx is not None:
            return 1
        return self._targets.shape[1]


class NBody6DataModule(pl.LightningDataModule):
    def __init__(
        self,
        cache_dir: Union[str, Path],
        feature_keys: Sequence[str],
        target_key: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._cache_dir = Path(cache_dir)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._target_idx = target_idx
        self._feature_keys = feature_keys
        self._target_key = target_key

        # load normalizers and metadata
        self._load_normalizers()

        # convert feature names to indices if provided
        self._feature_indices: Optional[List[int]] = None
        if feature_keys is not None:
            self._feature_indices = self._get_feature_indices(feature_keys)

        # convert target name to index if provided
        if target_key is not None:
            self._target_idx = self._get_target_index(target_key)

        # will be populated in setup()
        self._train_ds: Optional[NBody6Dataset] = None
        self._val_ds: Optional[NBody6Dataset] = None
        self._test_ds: Optional[NBody6Dataset] = None

    def _load_normalizers(self) -> None:
        normalizers_path = self._cache_dir / "normalizers.joblib"
        if not normalizers_path.exists():
            raise FileNotFoundError(f"Normalizers not found: {normalizers_path}")

        normalizer_data = joblib.load(normalizers_path)

        # reconstruct Normalizer objects from dicts
        self._feature_normalizers = [
            Normalizer.from_dict(d) for d in normalizer_data["feature_normalizers"]
        ]
        self._target_normalizers = [
            Normalizer.from_dict(d) for d in normalizer_data["target_normalizers"]
        ]
        self._feature_keys = list(normalizer_data["feature_keys"])
        self._target_keys = list(normalizer_data["target_keys"])

    def _get_feature_indices(self, feature_names: List[str]) -> List[int]:
        indices = []
        for name in feature_names:
            if name not in self._feature_keys:
                raise ValueError(
                    f"Feature '{name}' not found. Available features: {self._feature_keys}"
                )
            indices.append(self._feature_keys.index(name))
        return indices

    def _get_target_index(self, target_name: str) -> int:
        if target_name not in self._target_keys:
            raise ValueError(
                f"Target '{target_name}' not found. Available targets: {self._target_keys}"
            )
        return self._target_keys.index(target_name)

    def _load_split(self, split_name: str) -> Optional[Dict[str, np.ndarray]]:
        split_path = self._cache_dir / f"{split_name}.npz"
        if not split_path.exists():
            return None

        with np.load(split_path, allow_pickle=True) as data:
            return {
                "features": data["features"],
                "targets": data["targets"],
                "masks": data["masks"],
            }

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            train_data = self._load_split("train")
            if train_data is not None:
                self._train_ds = NBody6Dataset(
                    features=train_data["features"],
                    targets=train_data["targets"],
                    masks=train_data["masks"],
                    target_idx=self._target_idx,
                    feature_indices=self._feature_indices,
                )

            val_data = self._load_split("val")
            if val_data is not None:
                self._val_ds = NBody6Dataset(
                    features=val_data["features"],
                    targets=val_data["targets"],
                    masks=val_data["masks"],
                    target_idx=self._target_idx,
                    feature_indices=self._feature_indices,
                )

        if stage in (None, "test"):
            test_data = self._load_split("test")
            if test_data is not None:
                self._test_ds = NBody6Dataset(
                    features=test_data["features"],
                    targets=test_data["targets"],
                    masks=test_data["masks"],
                    target_idx=self._target_idx,
                    feature_indices=self._feature_indices,
                )

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("Train dataset not available. Call setup('fit') first.")
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError(
                "Validation dataset not available. Call setup('fit') first."
            )
        return DataLoader(
            self._val_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError("Test dataset not available. Call setup('test') first.")
        return DataLoader(
            self._test_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def denormalize_targets(self, y_norm: torch.Tensor) -> torch.Tensor:
        device = y_norm.device
        y_np = y_norm.detach().cpu().numpy()

        original_shape = y_np.shape

        # determine which target keys to use based on target_idx
        if self._target_idx is not None:
            # only denormalize the selected target
            target_keys_to_use = [self._target_keys[self._target_idx]]
            y_2d = y_np.reshape(-1, 1)
        else:
            target_keys_to_use = self._target_keys
            y_2d = y_np.reshape(-1, len(self._target_keys))

        for normalizer in self._target_normalizers:
            # find which columns this normalizer applies to
            col_indices = []
            for col in normalizer.columns:
                if col in target_keys_to_use:
                    col_indices.append(target_keys_to_use.index(col))

            if col_indices:
                y_2d[:, col_indices] = normalizer.inverse_transform(
                    y_2d[:, col_indices]
                )

        return torch.from_numpy(y_2d.reshape(original_shape)).to(device)

    def denormalize_features(self, x_norm: torch.Tensor) -> torch.Tensor:
        device = x_norm.device
        x_np = x_norm.detach().cpu().numpy()

        original_shape = x_np.shape
        # Get the feature keys that are actually being used
        feature_keys_used = (
            [self._feature_keys[i] for i in self._feature_indices]
            if self._feature_indices is not None
            else self._feature_keys
        )
        # reshape to 2D: (n_samples * n_stars, n_features)
        x_2d = x_np.reshape(-1, len(feature_keys_used))

        for normalizer in self._feature_normalizers:
            col_indices = []
            for col in normalizer.columns:
                if col in feature_keys_used:
                    col_indices.append(feature_keys_used.index(col))
            if col_indices:
                x_2d[:, col_indices] = normalizer.inverse_transform(
                    x_2d[:, col_indices]
                )

        return torch.from_numpy(x_2d.reshape(original_shape)).to(device)

    @property
    def feature_keys(self) -> List[str]:
        if self._feature_indices is not None:
            return [self._feature_keys[i] for i in self._feature_indices]
        return self._feature_keys

    @property
    def all_feature_keys(self) -> List[str]:
        """Return all available feature keys before filtering."""
        return self._feature_keys

    @property
    def target_keys(self) -> List[str]:
        if self._target_idx is not None:
            return [self._target_keys[self._target_idx]]
        return self._target_keys

    @property
    def all_target_keys(self) -> List[str]:
        return self._target_keys

    @property
    def input_dim(self) -> int:
        if self._feature_indices is not None:
            return len(self._feature_indices)
        return len(self._feature_keys)

    @property
    def output_dim(self) -> int:
        if self._target_idx is not None:
            return 1
        return len(self._target_keys)

    @property
    def feature_normalizers(self) -> List[Normalizer]:
        return self._feature_normalizers

    @property
    def target_normalizers(self) -> List[Normalizer]:
        return self._target_normalizers
