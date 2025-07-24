import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import astropy.units as u
import joblib
import numpy as np
import pandas as pd
from astropy.constants import L_sun, R_sun, sigma_sb

from module.nbody6.plot.animate import animate_nbody6_snapshots
from module.nbody6.file.base import NBody6OutputFile
from module.nbody6.file.fort82 import NBody6Fort82
from module.nbody6.file.fort83 import NBody6Fort83
from module.nbody6.file.out9 import NBody6OUT9
from module.nbody6.file.out34 import NBody6OUT34


class NBody6OutputLoader:
    def __init__(self, root: Union[str, Path]) -> None:
        self._root = Path(root).resolve()

        self._file_dict: Dict[str, NBody6OutputFile] = {}
        self._init_file_dict()
        self._is_file_dict_loaded = False

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

    def __repr__(self):
        return f"NBody6OutputLoader(root={self._root})"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def file_dict(self) -> Dict[str, NBody6OutputFile]:
        if not self._file_dict:
            warnings.warn("File dictionary is empty. Call load() first.")
        return self._file_dict

    def load(self, is_strict: bool = True) -> Dict[str, NBody6OutputFile]:
        # check if files are already loaded. If so, reset them.
        if self._is_file_dict_loaded:
            warnings.warn(
                "Files are already loaded. Existing data will be cleared before reloading."
            )
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
        ref_timestamps = list(timestamp_dict["OUT34"])
        for name, nbody6_file in self._file_dict.items():
            if name == "OUT34":
                continue
            for ts, ref_ts in zip(nbody6_file.timestamps, ref_timestamps):
                if ts != ref_ts:
                    nbody6_file.update_timestamp(ts, ref_ts)

        self._timestamps = ref_timestamps
        self._is_file_dict_loaded = True
        return self._file_dict

    @property
    def snapshot_dict(
        self,
    ) -> Optional[Dict[float, Dict[str, Union[Dict[str, any], pd.DataFrame]]]]:
        if self._snapshot_dict is None:
            warnings.warn("Snapshot dictionary is not initialized. Call merge() first.")
            return None
        if not self._is_file_dict_loaded:
            warnings.warn("File dictionary is not loaded. Call load() first.")
        return self._snapshot_dict

    @property
    def snapshot_list(
        self,
    ) -> Optional[List[Tuple[float, Dict[str, Union[Dict[str, any], pd.DataFrame]]]]]:
        return list(self.snapshot_dict.items()) if self.snapshot_dict else None

    def merge(self) -> Dict[float, Dict[str, Union[Dict[str, any], pd.DataFrame]]]:
        self._snapshot_dict = {}
        for timestamp in self._timestamps:
            self._snapshot_dict[timestamp] = self._merge(timestamp)

        return self._snapshot_dict

    def _merge(self, timestamp: float):
        # extract snapshots with header and data from corresponding timestamp
        out34_snapshot = self._file_dict["OUT34"][timestamp]
        out9_snapshot = self._file_dict["OUT9"][timestamp]
        fort82_snapshot = self._file_dict["fort.82"][timestamp]
        fort83_snapshot = self._file_dict["fort.83"][timestamp]

        xyz_conversion = out34_snapshot["header"]["rbar"]
        uvw_conversion = out34_snapshot["header"]["vstar"]

        # convert N.B unit to physical units
        merged_snapshot = out34_snapshot["data"].copy()
        # append suffix `_NB` to mark N.B units
        merged_snapshot.rename(
            columns={
                "x": "x_NB",
                "y": "y_NB",
                "z": "z_NB",
                "vx": "vx_NB",
                "vy": "vy_NB",
                "vz": "vz_NB",
            },
            inplace=True,
        )
        for col in ["x", "y", "z"]:
            merged_snapshot[col] = merged_snapshot[f"{col}_NB"] * xyz_conversion
        for col in ["vx", "vy", "vz"]:
            merged_snapshot[col] = merged_snapshot[f"{col}_NB"] * uvw_conversion

        reg_bin_df = pd.merge(
            out9_snapshot["data"],
            fort82_snapshot["data"],
            on=["name1", "name2"],
            how="outer",
            suffixes=("_out9", "_fort82"),
        )

        # convert from logarithmic to linear scale
        # L/L_sun = 10^(`log10(L/L_sun)[zlum]`)
        reg_bin_df["_l1"] = np.power(10, reg_bin_df["zlum1"])
        reg_bin_df["_l2"] = np.power(10, reg_bin_df["zlum2"])
        # R/R_sun = 10^(`log10(R/R_sun)[rad]`)
        reg_bin_df["_r1"] = np.power(10, reg_bin_df["rad1"])
        reg_bin_df["_r2"] = np.power(10, reg_bin_df["rad2"])

        # effective luminosity
        # L_eff / L_sun = (L1 + L2) / L_sun = L1/L_sun + L2/L_sun
        reg_bin_df["L_sol"] = reg_bin_df["_l1"] + reg_bin_df["_l2"]
        # effective radius
        # R_eff / R_sun = sqrt(R1^2 + R2^2) / R_sun = sqrt((R1/R_sun)^2 + (R2/R_sun)^2)
        reg_bin_df["R_sol"] = np.sqrt(
            np.power(reg_bin_df["_r1"], 2) + np.power(reg_bin_df["_r2"], 2)
        )
        # effective temperature
        # T_eff = (L_eff / (4 * pi * R_eff^2 * sigma_sb))^(1/4)
        reg_bin_df["T_eff"] = [
            (((L_sol * L_sun) / (4 * np.pi * (R_sol * R_sun) ** 2 * sigma_sb)) ** 0.25)
            .to(u.K)
            .value
            for L_sol, R_sol in zip(
                reg_bin_df["L_sol"],
                reg_bin_df["R_sol"],
            )
        ]
        reg_bin_df.rename(
            columns={
                "L_sol": "L_sol_reg_bin",
                "R_sol": "R_sol_reg_bin",
                "T_eff": "T_eff_reg_bin",
            },
            inplace=True,
        )

        single_df = fort83_snapshot["data"].copy()
        # T_eff = 10^(log10(T_eff))
        single_df["T_eff_single"] = np.power(10, single_df["tempe"])
        # L/L_sun = 10^(log10(L/L_sun))
        single_df["L_sol_single"] = np.power(10, single_df["zlum"])
        # R/R_sun = 10^(log10(R/R_sun))
        single_df["R_sol_single"] = np.power(10, single_df["rad"])

        # merge binary into main snapshot
        merged_snapshot = pd.merge(
            merged_snapshot,
            reg_bin_df[["cmName", "T_eff_reg_bin", "L_sol_reg_bin", "R_sol_reg_bin"]],
            left_on="name",
            right_on="cmName",
            how="left",
        )
        # merge single stars into main snapshot
        merged_snapshot = pd.merge(
            merged_snapshot,
            single_df[["name", "T_eff_single", "L_sol_single", "R_sol_single"]],
            on="name",
            how="left",
        )

        # flag binary systems
        merged_snapshot["is_binary"] = merged_snapshot["T_eff_reg_bin"].notna()

        # merge columns
        merged_snapshot["T_eff"] = merged_snapshot["T_eff_single"].combine_first(
            merged_snapshot["T_eff_reg_bin"]
        )
        merged_snapshot["L_sol"] = merged_snapshot["L_sol_single"].combine_first(
            merged_snapshot["L_sol_reg_bin"]
        )
        merged_snapshot["R_sol"] = merged_snapshot["R_sol_single"].combine_first(
            merged_snapshot["R_sol_reg_bin"]
        )
        # surface flux with respect to solar units
        # F/F_sun = (L/L_sun) / (R/R_sun)^2
        merged_snapshot["F_sol"] = (
            merged_snapshot["L_sol"] / merged_snapshot["R_sol"] ** 2
        )
        merged_snapshot["log_T_eff"] = np.log10(merged_snapshot["T_eff"])
        merged_snapshot["log_L_sol"] = np.log10(merged_snapshot["L_sol"])
        merged_snapshot["log_R_sol"] = np.log10(merged_snapshot["R_sol"])
        merged_snapshot["log_F_sol"] = np.log10(merged_snapshot["F_sol"])

        return {
            "header": out34_snapshot["header"],
            "data": merged_snapshot[
                out34_snapshot["data"].columns.tolist()
                + [
                    "T_eff",
                    "L_sol",
                    "R_sol",
                    "F_sol",
                    "log_T_eff",
                    "log_L_sol",
                    "log_R_sol",
                    "log_F_sol",
                    "is_binary",
                ]
            ],
            "raw": {
                "OUT34": out34_snapshot["data"],
                "OUT9": out9_snapshot["data"],
                "fort.82": fort82_snapshot["data"],
                "fort.83": fort83_snapshot["data"],
            },
        }

    def export_snapshot_dict(
        self, output_path: Union[str, Path], overwrite: bool = False
    ) -> None:
        if self._snapshot_dict is None:
            raise ValueError(
                "Snapshot dictionary is not initialized. Call merge() first."
            )

        output_path = Path(output_path).resolve()
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File `{output_path}` already exists. Use overwrite=True to force overwrite."
            )

        joblib.dump(self._snapshot_dict, output_path)
        print(f"Snapshot dictionary exported to `{output_path}`.")

    def animate_snapshots(
        self,
        output_path: Optional[Union[str, Path]] = None,
        fig_title_text: Optional[str] = None,
        animation_fps: int = 10,
        animation_dpi: int = 300,
    ):
        snapshot_list = self.snapshot_list
        if not snapshot_list:
            raise ValueError("No snapshots available to animate.")

        animation = animate_nbody6_snapshots(
            snapshot_list=snapshot_list,
            output_path=output_path,
            fig_title_text=fig_title_text,
            animation_fps=animation_fps,
            animation_dpi=animation_dpi,
        )
        return animation
