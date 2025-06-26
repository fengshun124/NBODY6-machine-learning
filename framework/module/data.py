from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from torch.utils.data import DataLoader, Dataset


class NBodyDataset(Dataset):
    def __init__(
        self,
        snapshot_pairs: List[Tuple[Tuple[Union[int, float, str], ...], pd.DataFrame]],
    ):
        self._snapshots, self._attr_tuples = zip(*snapshot_pairs)

    def __len__(self):
        return len(self._snapshots)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(
                self._attr_tuples[idx].values.astype(np.float32), dtype=torch.float32
            ),
            torch.tensor(
                np.array(self._snapshots[idx], dtype=np.float32), dtype=torch.float32
            ),
        )


class NBodyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        snapshots_pairs: List[
            Tuple[Tuple[Union[int, float, str], ...], List[pd.DataFrame]]
        ],
        snapshot_scalar_dict: Dict[
            Tuple[int, ...], Union[None, StandardScaler, MinMaxScaler, RobustScaler]
        ],
        attr_tuple_scalar_dict: Dict[
            int, Union[None, StandardScaler, MinMaxScaler, RobustScaler]
        ],
        batch_size: int = 32,
        val_frac: float = 0.2,
        num_workers: int = 4,
        random_seed: int = 42,
    ):
        super().__init__()
        self._raw_snapshot_pairs = [
            (attr_tuple, dataframe)
            for attr_tuple, dataframes in snapshots_pairs
            for dataframe in dataframes
        ]
        self._batch_size = batch_size
        self._val_frac = val_frac
        self._num_workers = num_workers
        self._random_seed = random_seed
        self._snapshot_scalar_dict = snapshot_scalar_dict
        self._attr_tuple_scalar_dict = attr_tuple_scalar_dict
        self.train_dataset: Optional[NBodyDataset] = None
        self.val_dataset: Optional[NBodyDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        np.random.seed(self._random_seed)
        torch.manual_seed(self._random_seed)
        
        label_encoders = {}
        if self._raw_snapshot_pairs:
            num_attributes = len(self._raw_snapshot_pairs[0])
            for attr_idx in range(num_attributes):
                attribute_values = [
                    item[attr_idx] for item, _ in self._raw_snapshot_pairs
                ]
                if any(isinstance(x, str) for x in attribute_values):
                    label_encoder = LabelEncoder()
                    label_encoder.fit(
                        [val for val in attribute_values if isinstance(val, str)]
                    )
                    label_encoders[attr_idx] = label_encoder

            updated_pairs = []
            for attr_tuple, snapshot_df in self._raw_snapshot_pairs:
                attr_list = list(attr_tuple)
                for attr_idx, label_encoder in label_encoders.items():
                    if isinstance(attr_list[attr_idx], str):
                        attr_list[attr_idx] = label_encoder.transform(
                            [attr_list[attr_idx]]
                        )[0]
                updated_pairs.append((tuple(attr_list), snapshot_df))
            self._raw_snapshot_pairs = updated_pairs

        train_indices, val_indices = np.split(
            np.random.permutation(len(self._raw_snapshot_pairs)),
            [int(len(self._raw_snapshot_pairs) * (1 - self._val_frac))],
        )
        train_pairs = [self._raw_snapshot_pairs[i] for i in train_indices]
        val_pairs = [self._raw_snapshot_pairs[i] for i in val_indices]

        for feature_indices, scalar in self._snapshot_scalar_dict.items():
            if scalar is None:
                continue
            scalar.fit(
                np.concatenate(
                    [
                        snapshot_df.iloc[:, list(feature_indices)].values.flatten()
                        for _, snapshot_df in train_pairs
                    ]
                ).reshape(-1, 1)
            )
            for idx, (attr_tuple, snapshot_df) in enumerate(train_pairs):
                df_copy = snapshot_df.copy()
                for col_idx in feature_indices:
                    df_copy.iloc[:, col_idx] = scalar.transform(
                        snapshot_df.iloc[:, col_idx].values.reshape(-1, 1)
                    ).flatten()
                train_pairs[idx] = (attr_tuple, df_copy)

            for idx, (attr_tuple, snapshot_df) in enumerate(val_pairs):
                df_copy = snapshot_df.copy()
                for col_idx in feature_indices:
                    df_copy.iloc[:, col_idx] = scalar.transform(
                        snapshot_df.iloc[:, col_idx].values.reshape(-1, 1)
                    ).flatten()
                val_pairs[idx] = (attr_tuple, df_copy)

        for attr_idx, scalar in self._attr_tuple_scalar_dict.items():
            if scalar is None:
                continue
            scalar.fit(
                np.array([item[attr_idx] for item, _ in train_pairs]).reshape(-1, 1)
            )
            for idx, (attr_tuple, snapshot_df) in enumerate(train_pairs):
                scaled_val = scalar.transform(
                    np.array(attr_tuple[attr_idx]).reshape(-1, 1)
                )[0][0]
                new_tuple = (
                    attr_tuple[:attr_idx] + (scaled_val,) + attr_tuple[attr_idx + 1 :]
                )
                train_pairs[idx] = (new_tuple, snapshot_df)

            for idx, (attr_tuple, snapshot_df) in enumerate(val_pairs):
                scaled_val = scalar.transform(
                    np.array(attr_tuple[attr_idx]).reshape(-1, 1)
                )[0][0]
                new_tuple = (
                    attr_tuple[:attr_idx] + (scaled_val,) + attr_tuple[attr_idx + 1 :]
                )
                val_pairs[idx] = (new_tuple, snapshot_df)

        self.train_dataset = NBodyDataset(train_pairs)
        self.val_dataset = NBodyDataset(val_pairs)

    @property
    def feature_dim(self) -> Optional[int]:
        if self.train_dataset and len(self.train_dataset) > 0:
            return self.train_dataset[0][0].shape[1]
        return None

    @property
    def attr_dim(self) -> Optional[int]:
        if self.train_dataset and len(self.train_dataset) > 0:
            return len(self.train_dataset[0][1])
        return None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            persistent_workers=self._num_workers > 1,
            pin_memory=self._num_workers > 1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            persistent_workers=self._num_workers > 1,
            pin_memory=self._num_workers > 1,
        )
