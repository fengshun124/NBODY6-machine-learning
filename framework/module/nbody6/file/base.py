import warnings
from abc import ABC
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd


class NBody6OutputFile(ABC):
    def __init__(self, filepath: Union[str, Path], load_config: Dict[str, Any]) -> None:
        self._filepath = Path(filepath)
        if not self._filepath.is_file():
            raise FileNotFoundError(f"File not found: {self._filepath}")

        self._filename = self._filepath.name
        self._config = load_config
        self._data: Optional[
            Dict[float, Dict[str, Union[Dict[str, Any], pd.DataFrame]]]
        ] = None

    def __repr__(self):
        return f"{__class__.__name__}({self._filepath})"

    def __len__(self):
        if self._data is not None:
            return len(self._data)
        return -1

    def __getitem__(
        self, timestamp: float
    ) -> Dict[str, Union[Dict[str, Any], pd.DataFrame]]:
        if self._data is None:
            raise ValueError("Data not loaded. Please call load() first.")
        if timestamp not in self._data:
            raise KeyError(f"Timestamp {timestamp} not found in data.")

        data_dict = self._data[timestamp].copy()
        data_dict.pop("_header_line", None)
        return data_dict

    @property
    def timestamps(self) -> List[float]:
        if self._data is None:
            warnings.warn("Data not loaded. Please call load() first.", UserWarning)
            return []
        return list(self._data.keys())

    def update_timestamp(self, old_timestamp: float, new_timestamp: float) -> None:
        if self._data is None:
            raise ValueError("Data not loaded. Please call load() first.")
        if old_timestamp not in self._data:
            raise KeyError(f"Timestamp {old_timestamp} not found in data.")
        if new_timestamp in self._data:
            raise KeyError(f"Timestamp {new_timestamp} already exists in data.")

        self._data[new_timestamp] = self._data.pop(old_timestamp)

    @property
    def data_dict(
        self,
    ) -> Optional[Dict[float, Dict[str, Union[Dict[str, Any], pd.DataFrame]]]]:
        if self._data is None:
            warnings.warn("Data not loaded. Please call load() first.", UserWarning)
        return {
            timestamp: {k: v for k, v in data.items() if not k.startswith("_")}
            for timestamp, data in self._data.items()
        }

    @property
    def data_list(
        self,
    ) -> Optional[List[Tuple[float, Dict[str, Union[Dict[str, Any], pd.DataFrame]]]]]:
        return list(self.data_dict.items()) if self.data_dict else None

    def load(self, is_strict: bool = True):
        if self._data:
            warnings.warn(
                f"Reloading {self._filepath}, all previous data will be discarded.",
                UserWarning,
            )
        self._data = {}

        for header_data, row_data in self._parse_file():
            try:
                header_dict = self._map_tokens(
                    ln_data=header_data,
                    schema=self._config["header_schema"],
                    allow_missing=not is_strict,
                )
                row_dicts = [
                    self._map_tokens(
                        ln_data=row,
                        schema=self._config["row_schema"],
                        allow_missing=not is_strict,
                    )
                    for row in row_data
                ]

                sort_key = "name" if "name" in self._config["row_schema"] else "name1"
                data_df = (
                    pd.DataFrame.from_records(row_dicts)
                    .sort_values(by=sort_key)
                    .reset_index(drop=True)
                )

                timestamp = round(header_dict["time"], 2)

                # warning if timestamp already exists
                if timestamp in self._data:
                    warnings.warn(
                        f"{self._filename} - "
                        f"[LINE {header_data[0]}] timestamp {timestamp} duplicated with "
                        f"[LINE {self._data[timestamp]['_header_line']}]. "
                        "Keeping the last occurrence.",
                        UserWarning,
                    )

                self._data[timestamp] = {
                    "header": header_dict,
                    "data": data_df,
                    "_header_line": header_data[0],
                }

            except Exception as e:
                raise ValueError(
                    f"[{self._filename} - LINE {header_data[0]}] {e}"
                ) from e

        # sort the data by timestamp
        self._data = dict(sorted(self._data.items()))

        return self._data

    def _parse_file(self):
        header_marker = self._config["header_prefix"]
        footer_marker = self._config.get("footer_prefix", None)

        with self._filepath.open("r", encoding="utf-8") as f:
            groups = groupby(
                (
                    (idx + 1, line.strip())
                    for idx, line in enumerate(f)
                    if line.strip()
                    and not (footer_marker and line.strip().startswith(footer_marker))
                ),
                lambda line_data: line_data[1].startswith(header_marker),
            )

            for is_header, group_iter in groups:
                # skip non-header lines
                if not is_header:
                    continue

                # collect header data first before processing
                group = list(group_iter)
                if not group:
                    continue

                # process header lines
                for ln_num, header_line in group:
                    if not header_line:
                        warnings.warn(
                            f"[{self._filename} - LINE {ln_num}] "
                            f"Empty header line found in {self._filepath}, "
                            "possibly malformed file, skipping.",
                        )
                        continue

                # collect header tokens for mapping
                start_ln, stop_ln = group[0][0], group[-1][0]
                header_tokens = [
                    token
                    for _, line in group
                    for token in line.split(header_marker, 1)[1].split()
                ]

                try:
                    next_is_header, next_group_iter = next(groups)
                    row_data = [] if next_is_header else list(next_group_iter)
                except StopIteration:
                    row_data = []

                yield (
                    (
                        f"{start_ln}-{stop_ln}"
                        if start_ln != stop_ln
                        else str(start_ln),
                        header_tokens,
                    ),
                    [(ln_num, line.split()) for ln_num, line in row_data],
                )

    def _map_tokens(
        self,
        ln_data: Tuple[Union[str, int], list[str]],
        schema: Dict[str, Tuple[Union[int, List[int]], Callable]],
        allow_missing: bool = False,
    ) -> Dict[str, Any]:
        ln_num, tokens = ln_data

        mapped_data = {}
        for key, (idx, converter) in schema.items():
            try:
                if isinstance(idx, int):
                    value = tokens[idx]
                else:
                    value = [tokens[i] for i in idx]
                mapped_data[key] = converter(value)

                # special treatment for `name` keys (`name` / `name1` & `name2`)
                if key.startswith("name") and mapped_data[key] <= 0:
                    warnings.warn(
                        f"[{self._filename} - LINE {ln_num}] "
                        f"Invalid value for '{key}': {mapped_data[key]}. "
                    )

            except IndexError:
                if allow_missing:
                    mapped_data[key] = None
                else:
                    raise ValueError(f"Missing required token '{key}' at index {idx}")
            except Exception as e:
                raise ValueError(e)

        if not mapped_data:
            raise ValueError(f"No valid tokens found in {tokens}")

        return mapped_data
