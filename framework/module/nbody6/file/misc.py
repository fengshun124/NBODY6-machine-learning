import warnings
from pathlib import Path
from typing import Union

import pandas as pd
from module.nbody6.file.base import NBody6FileParserBase, NBody6FileSnapshot


class NBody6DensityCenterFile(NBody6FileParserBase):
    def __init__(self, filepath: Union[str, Path]) -> None:
        super().__init__(
            filepath,
            {
                "header_prefix": "#",
                "header_length": 0,
                "row_schema": {
                    "time": (0, float),
                    "r_tidal": (1, float),
                    "density_center_x": (2, float),
                    "density_center_y": (3, float),
                    "density_center_z": (4, float),
                    # "n_stars": (5, int),
                    # "total_mass": (6, float),
                },
                "header_schema": {},
            },
        )

    def _parse_file(self):
        with self._filepath.open("r", encoding="utf-8") as f:
            lines = [
                (idx, line.strip())
                for idx, line in enumerate(f, start=1)
                if line.strip() and not line.strip().startswith("#")
            ]

        for ln_num, txt in lines:
            tokens = txt.split()
            yield ((str(ln_num), []), [(ln_num, tokens)])

    def load(self, is_strict: bool = True):
        if self._data:
            warnings.warn(
                f"Reloading {self._filepath}, all previous data will be discarded.",
                UserWarning,
            )
        self._data = {}

        for header_data, row_data in self._parse_file():
            header_ln_ref = header_data[0]
            try:
                if not row_data:
                    continue

                row_ln_num, row_tokens = row_data[0]
                try:
                    row_dict = self._map_tokens(
                        ln_data=(row_ln_num, row_tokens),
                        schema=self._config["row_schema"],
                        allow_missing=not is_strict,
                    )
                except Exception as e:
                    raise ValueError(f"LINE {row_ln_num}: {e}") from e

                header_dict = row_dict
                data_df = pd.DataFrame.from_records([row_dict])
                timestamp = round(header_dict["time"], 2)

                # validate negative tidal radius
                if header_dict["r_tidal"] < 0:
                    warnings.warn(
                        f"{self._filename} - [LINE {header_ln_ref}] "
                        f"Negative tidal radius '{header_dict['r_tidal']}' found.",
                        UserWarning,
                    )

                if timestamp in self._data:
                    warnings.warn(
                        f"{self._filename} - "
                        f"[LINE {header_ln_ref}] timestamp {timestamp} duplicated with "
                        f"[LINE {self._data[timestamp].header_ln_ref}]. "
                        "Keeping the last occurrence.",
                        UserWarning,
                    )
                self._data[timestamp] = NBody6FileSnapshot(
                    header=header_dict, data=data_df, header_ln_ref=header_ln_ref
                )

            except Exception as e:
                error_str = str(e)
                if "LINE" in error_str:
                    line_info, message = error_str.split(":", 1)
                    context = f"[{self._filename} - {line_info.strip()}]"
                else:
                    context = f"[{self._filename} - LINE {header_ln_ref}]"
                    message = error_str

                raise ValueError(f"{context} {message.strip()}") from None

        if self._data:
            sorted_timestamps = sorted(self._data.keys())
            self._data = {ts: self._data[ts] for ts in sorted_timestamps}
