import warnings
from pathlib import Path
from typing import Iterator

import numpy as np


class Shard:
    def __init__(
        self,
        feature: np.ndarray,
        feature_keys: tuple[str, ...] | str,
        target: np.ndarray,
        target_keys: tuple[str, ...] | str,
        ptr: np.ndarray,
    ) -> None:
        # feature in 2D array (n_sample x feature_dim)
        self._feature = np.asarray(feature, dtype=np.float32, order="C")
        self._feature_keys = (
            tuple([feature_keys])
            if isinstance(feature_keys, str)
            else tuple(feature_keys)
        )

        # target in 2D array (n_sample x target_dim)
        self._target = (lambda t: t if t.ndim != 1 else t.reshape(-1, 1))(
            np.asarray(target, dtype=np.float32, order="C")
        )
        self._target_keys = (
            tuple([target_keys]) if isinstance(target_keys, str) else tuple(target_keys)
        )

        # pointer array (n_sample + 1,)
        self._ptr = np.asarray(ptr, dtype=np.int64)

        self._validate()

    def _validate(self) -> None:
        # check array dimensions
        if self._feature.ndim != 2:
            raise ValueError(
                f"Expected feature to be 2D, got shape={self._feature.shape}"
            )
        if self._target.ndim != 2:
            raise ValueError(
                f"Expected target to be 2D, got shape={self._target.shape}"
            )
        if self._ptr.ndim != 1 or self._ptr.size < 2:
            raise ValueError("ptr must be 1D with length >= 2")

        # check key / value consistency
        if (feature_dim := self._feature.shape[1]) != len(self._feature_keys):
            raise ValueError(
                f"Expected feature dimension {len(self._feature_keys)}, got {feature_dim}."
            )
        if (target_dim := self._target.shape[1]) != len(self._target_keys):
            raise ValueError(
                f"Expected target dimension {len(self._target_keys)}, got {target_dim}."
            )

        # check ptr consistency
        if self._ptr[0] != 0:
            raise ValueError(
                f"Expected pointer array to start with 0, got {self._ptr[0]}"
            )
        if np.any(self._ptr[1:] < self._ptr[:-1]):
            raise ValueError("Pointer array must be non-decreasing.")
        if self._ptr[-1] != self._feature.shape[0]:
            raise ValueError(
                f"Expected pointer array to end with {self._feature.shape[0]}, "
                f"got {self._ptr[-1]}"
            )
        if (n_sample := self._target.shape[0]) != (self._ptr.shape[0] - 1):
            raise ValueError(f"Expected samples {len(self)} sample(s), got {n_sample}.")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"feature_keys={self._feature_keys}, "
            f"target_keys={self._target_keys}, "
            f"n_samples={len(self)}, "
            f"n_total_star={self._feature.shape[0]}"
            ")"
        )

    def __len__(self) -> int:
        return int(self._ptr.shape[0] - 1)

    @property
    def feature(self) -> np.ndarray:
        return self._feature.copy()

    @property
    def feature_keys(self) -> tuple[str, ...]:
        return self._feature_keys

    @property
    def feature_dim(self) -> int:
        return self._feature.shape[1]

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @property
    def target_keys(self) -> tuple[str, ...]:
        return self._target_keys

    @property
    def target_dim(self) -> int:
        return self._target.shape[1]

    @property
    def pointer(self) -> np.ndarray:
        return self._ptr.copy()

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._feature[self._ptr[idx] : self._ptr[idx + 1]].copy(),
            self._target[idx].copy(),
        )

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for i in range(len(self)):
            yield self[i]

    @property
    def snapshots(self) -> list[tuple[np.ndarray, np.ndarray]]:
        warnings.warn(
            "Materializing all snapshots may consume large memory; prefer iteration.",
            UserWarning,
        )
        return [self[i] for i in range(len(self))]

    def _is_compatibility(self, other: "Shard") -> bool:
        if not isinstance(other, Shard):
            return False
        if self._feature_keys != other._feature_keys:
            return False
        if self._target_keys != other._target_keys:
            return False
        return True

    # merge two compatible SplitShard instances
    def _merged_arrays(
        self, other: "Shard"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._is_compatibility(other):
            raise ValueError("Incompatible SplitShard instances.")

        offset = int(self._ptr[-1])
        new_feature = np.concatenate([self._feature, other._feature], axis=0)
        new_target = np.concatenate([self._target, other._target], axis=0)
        new_ptr = np.concatenate([self._ptr, other._ptr[1:] + offset], axis=0).astype(
            np.int64, copy=False
        )

        return new_feature, new_target, new_ptr

    # A.extend(B)
    def extend(self, other: "Shard") -> None:
        new_feature, new_target, new_ptr = self._merged_arrays(other)
        self._feature = new_feature
        self._target = new_target
        self._ptr = new_ptr

    # A += B
    def __iadd__(self, other: "Shard") -> "Shard":
        self.extend(other)
        return self

    # C = A + B
    def __add__(self, other: "Shard") -> "Shard":
        new_feature, new_target, new_ptr = self._merged_arrays(other)
        return Shard(
            feature=new_feature,
            feature_keys=self._feature_keys,
            target=new_target,
            target_keys=self._target_keys,
            ptr=new_ptr,
        )

    # I/O operations
    def to_npz(self, filepath: Path | str) -> None:
        filepath = Path(filepath).resolve()
        # NumPy would silently append .npz if not present,
        # thus the temporary file are named as 'xx.tmp.npz' to avoid unintended append.
        tmp_filepath = filepath.with_suffix(".tmp" + filepath.suffix)

        try:
            np.savez_compressed(
                tmp_filepath,
                feature=self._feature,
                feature_keys=np.array(self._feature_keys, dtype="U"),
                target=self._target,
                target_keys=np.array(self._target_keys, dtype="U"),
                ptr=self._ptr,
            )
            tmp_filepath.replace(filepath)
        finally:
            tmp_filepath.unlink(missing_ok=True)

    @classmethod
    def from_npz(cls, filepath: Path | str) -> "Shard":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        npz = np.load(filepath, allow_pickle=False)
        return cls(
            feature=npz["feature"],
            feature_keys=tuple(npz["feature_keys"].tolist()),
            target=npz["target"],
            target_keys=tuple(npz["target_keys"].tolist()),
            ptr=npz["ptr"],
        )

    def to_dict(self) -> dict:
        return {
            "feature": self._feature,
            "feature_keys": self._feature_keys,
            "target": self._target,
            "target_keys": self._target_keys,
            "ptr": self._ptr,
        }
