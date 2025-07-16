import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from module.nbody6.base.base import NBody6OutputFile
from module.nbody6.base.fort82 import NBody6Fort82
from module.nbody6.base.fort83 import NBody6Fort83
from module.nbody6.base.out9 import NBody6OUT9
from module.nbody6.base.out34 import NBody6OUT34


class NBody6OutputLoader:
    def __init__(self, root: Union[str, Path]) -> None:
        self._root = Path(root).resolve()

        self._file_dict: Dict[str, NBody6OutputFile] = {}
        self._init_file_dict()

        self._snapshot_dict: Optional[
            Dict[float, Dict[str, Union[Dict[str, any], pd.DataFrame]]]
        ] = None

        self._timestamps: Optional[List[float]] = None

    def _init_file_dict(self) -> None:
        self._file_dict = {
            "OUT34": NBody6OUT34(self._root / "OUT34"),
            "OUT9": NBody6OUT9(self._root / "OUT9"),
            "fort.82": NBody6Fort82(self._root / "fort.82"),
            "fort.83": NBody6Fort83(self._root / "fort.83"),
        }

    @property
    def file_dict(self) -> Dict[str, NBody6OutputFile]:
        if not self._file_dict:
            warnings.warn("File dictionary is empty. Call load() first.")
        return self._file_dict

    def load(self, is_strict: bool = True) -> Dict[str, NBody6OutputFile]:
        # check if files are already loaded. If so, reset them.
        if any(nbody6_file.data_dict for nbody6_file in self._file_dict.values()):
            warnings.warn("Files are already loaded. Reloading them.")
            self._init_file_dict()

        if not is_strict:
            warnings.warn(
                "Loading files in non-strict mode. `NaN` values may presented in the data."
            )

        for filename, nbody6_file in self._file_dict.items():
            nbody6_file.load(is_strict=is_strict)

        # check if all files share the same timestamps with tolerance of 1e-2
        timestamp_dict = {
            name: nbody6_file.timestamps
            for name, nbody6_file in self._file_dict.items()
        }

        # check if all files have the same number of timestamps
        timestamp_len_dict = {name: len(ts) for name, ts in timestamp_dict.items()}
        if len(set(timestamp_len_dict.values())) > 1:
            raise ValueError(
                f"Timestamp lengths mismatch: {timestamp_len_dict}. "
                "All files must have the same number of timestamps."
            )

        for i, timestamps in enumerate(zip(*timestamp_dict.values())):
            if max(timestamps) - min(timestamps) > 2e-2:
                mismatched_timestamps = {
                    name: ts[i] for name, ts in timestamp_dict.items()
                }
                raise ValueError(
                    f"Timestamp mismatch at index {i}: {mismatched_timestamps}"
                )

        # set standard timestamps to OUT34
        self._timestamps = list(timestamp_dict["OUT34"])

        return self._file_dict

    def merge(self) -> Dict[float, Dict[str, Union[Dict[str, any], pd.DataFrame]]]:
        for timestamp in self._timestamps:
            pass

    # def _merge(self, timestamp: float):
    #     # extract snapshots with header and data from corresponding timestamp
    #     out34_snapshot = self._file_dict["OUT34"][timestamp]
    #     out9_snapshot = self._file_dict["OUT9"][timestamp]
    #     fort82_snapshot = self._file_dict["fort.82"][timestamp]
    #     fort83_snapshot = self._file_dict["fort.83"][timestamp]
