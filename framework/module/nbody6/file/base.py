import warnings
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd


@dataclass
class NBody6FileSnapshot:
    header: Dict[str, Any]
    data: pd.DataFrame
    header_ln_ref: str

    def __repr__(self):
        return (
            f"{__class__.__name__}"
            f"(header={self.header}, data_shape={self.data.shape}, header_ln_ref='{self.header_ln_ref}')"
        )


class NBody6FileParserBase(ABC):
    def __init__(self, filepath: Union[str, Path], load_config: Dict[str, Any]) -> None:
        self._filepath = Path(filepath).resolve()
        if not self._filepath.is_file():
            raise FileNotFoundError(f"File not found: {self._filepath}")

        self._filename = str(self._filepath)
        self._config = load_config
        self._data: Optional[Dict[float, NBody6FileSnapshot]] = None

    def __repr__(self):
        return f"{__class__.__name__}({self._filepath})"

    def __len__(self):
        if self._data is not None:
            return len(self._data)
        return -1

    def __getitem__(self, timestamp: float) -> NBody6FileSnapshot:
        if self._data is None:
            raise ValueError("Data not loaded. Please call load() first.")
        if timestamp not in self._data:
            raise KeyError(f"Timestamp {timestamp} not found in data.")

        return self._data[timestamp]

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
    def data_dict(self) -> Optional[Dict[float, NBody6FileSnapshot]]:
        if self._data is None:
            warnings.warn("Data not loaded. Please call load() first.", UserWarning)
        return self._data

    @property
    def data_list(self) -> Optional[List[Tuple[float, NBody6FileSnapshot]]]:
        return list(self.data_dict.items()) if self.data_dict else None

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
                header_dict = self._map_tokens(
                    ln_data=header_data,
                    schema=self._config["header_schema"],
                    allow_missing=not is_strict,
                )

                row_dicts = []
                for row_ln_num, row_tokens in row_data:
                    try:
                        row_dicts.append(
                            self._map_tokens(
                                ln_data=(row_ln_num, row_tokens),
                                schema=self._config["row_schema"],
                                allow_missing=not is_strict,
                            )
                        )
                    except Exception as e:
                        raise ValueError(f"LINE {row_ln_num}: {e}") from e

                if row_dicts:
                    sort_key = (
                        "name" if "name" in self._config["row_schema"] else "name1"
                    )
                    data_df = (
                        pd.DataFrame.from_records(row_dicts)
                        .sort_values(by=sort_key)
                        .reset_index(drop=True)
                    )
                else:
                    data_df = pd.DataFrame(columns=self._config["row_schema"].keys())

                timestamp = round(header_dict["time"], 2)

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

        return self._data

    def _parse_file(self):
        header_prefix = self._config["header_prefix"]
        header_length = self._config.get("header_length", None)
        footer_prefix = self._config.get("footer_prefix", None)

        if header_length is None:
            warnings.warn(
                f'[{self._filename}] "header_length" not specified. '
                f"All consecutive lines starting with '{header_prefix}'"
                "will be treated as a single header block.",
                UserWarning,
            )

        with self._filepath.open("r", encoding="utf-8") as f:
            lines = [
                (idx, txt)
                for idx, line in enumerate(f, start=1)
                if (txt := line.strip())
                and not (footer_prefix and txt.startswith(footer_prefix))
            ]

        ln_idx = 0
        while ln_idx < len(lines):
            if not lines[ln_idx][1].startswith(header_prefix):
                ln_idx += 1
                continue

            header_block = []
            start_idx = ln_idx
            if header_length:
                if ln_idx + header_length > len(lines):
                    break

                header_block = lines[ln_idx : ln_idx + header_length]
                for ln_num, txt in header_block:
                    if not txt.startswith(header_prefix):
                        raise ValueError(
                            f"Expected {header_length} consecutive header lines, "
                            f'but line {ln_num} does not start with "{header_prefix}".'
                        )
                ln_idx += header_length
            else:
                while ln_idx < len(lines) and lines[ln_idx][1].startswith(
                    header_prefix
                ):
                    ln_idx += 1
                header_block = lines[start_idx:ln_idx]

            if not header_block:
                continue

            start_ln, stop_ln = header_block[0][0], header_block[-1][0]
            ln_ref = f"{start_ln}-{stop_ln}" if start_ln != stop_ln else str(start_ln)

            header_tokens = [
                token
                for _, txt in header_block
                for token in txt.lstrip(header_prefix).split()
            ]

            rows = []
            while ln_idx < len(lines) and not lines[ln_idx][1].startswith(
                header_prefix
            ):
                row_ln_num, row_txt = lines[ln_idx]
                rows.append((row_ln_num, row_txt.split()))
                ln_idx += 1

            yield ((ln_ref, header_tokens), rows)

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
                raw = tokens[idx] if isinstance(idx, int) else [tokens[i] for i in idx]
                mapped_data[key] = converter(raw)

                # special handling for 'name' keys
                if (
                    key.startswith("name")
                    and isinstance(mapped_data[key], int)
                    and mapped_data[key] <= 0
                ):
                    warnings.warn(
                        f"[{self._filename} - LINE {ln_num}] "
                        f"Possibly multiple system '{key}': {mapped_data[key]}."
                    )
            except IndexError:
                if not allow_missing:
                    raise ValueError(f"Missing value for '{key}'") from None
                mapped_data[key] = None
            except Exception as e:
                raise ValueError(f"Cannot process value for '{key}': {e}") from e

        if not mapped_data:
            raise ValueError("No data mapped from tokens")
        return mapped_data
