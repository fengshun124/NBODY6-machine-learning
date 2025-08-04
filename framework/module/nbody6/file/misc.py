import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from module.nbody6.file.base import NBody6OutputFile


class NBody6DensityCenterFile(NBody6OutputFile):
    def __init__(self, filepath: Union[str, Path]) -> None:
        super().__init__(
            filepath,
            {
                "row_schema": {
                    "time": (0, float),
                    "r_tidal": (1, float),
                    "density_center_x": (2, float),
                    "density_center_y": (3, float),
                    "density_center_z": (4, float),
                    # "n_stars": (5, int),
                    # "total_mass": (6, float),
                }
            },
        )
        self._data: Optional[Dict[float, Dict[str, float]]] = None

    def load(self, is_strict: bool = True):
        if self._data:
            warnings.warn(
                f"Reloading {self._filepath}, all previous data will be discarded.",
                UserWarning,
            )
        self._data = {}

        with open(self._filepath, "r") as file:
            lines = file.readlines()

        for ln_num, line in enumerate(lines):
            if line.startswith("#"):
                continue
            row_dict = self._map_tokens(
                ln_data=(ln_num + 1, line.split()),
                schema=self._config["row_schema"],
                allow_missing=not is_strict,
            )

            self._data[row_dict["time"]] = {
                "header": row_dict,
                "data": pd.DataFrame(row_dict, index=[0]),
                "_header_line": ln_num,
            }

            # validate if negative tidal radius presents
            if row_dict["r_tidal"] < 0:
                warnings.warn(
                    f"{self._filename} - [LINE {ln_num + 1}] "
                    f"Negative tidal radius '{row_dict['r_tidal']}' found. "
                )

        self._data = dict(sorted(self._data.items()))

        return self._data
