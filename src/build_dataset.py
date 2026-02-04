import gc
import json
import logging
import os
import re
from functools import reduce
from pathlib import Path
from typing import Literal, Sequence

import click
import joblib
import numpy as np
import pandas as pd
from dataset.scaler import ArrayScalerBundle, NormMethod
from dataset.shard import Shard
from dotenv import load_dotenv
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from utils import OUTPUT_BASE, setup_logger

load_dotenv()

JOBLIB_ROOT_ENV = os.getenv("JOBLIB_ROOT")
if not JOBLIB_ROOT_ENV:
    raise RuntimeError("Environment variable 'JOBLIB_ROOT' is not set")
JOBLIB_ROOT = Path(JOBLIB_ROOT_ENV).resolve(strict=True)

# Pattern: Rad04-zmet0002-M4-0003
SIM_ATTR_PATTERN = re.compile(r"Rad(\d{2})-zmet(\d{4})-M(\d)-(\d{4})")

logger = logging.getLogger(__name__)

SplitType = Literal["train", "val", "test"]


def _cache_per_run_shard(
    run_id: str,
    split: SplitType,
    dataset_dir: Path,
    feature_keys: Sequence[str],
    target_keys: Sequence[str],
    min_stars: int = 10,
    log_file: Path | str | None = None,
) -> None:
    # setup logger
    setup_logger(
        (
            Path(log_file).resolve()
            if log_file is not None
            else (OUTPUT_BASE / "log" / "build_dataset.log").resolve()
        )
    )

    shard_file = (dataset_dir / split / f"{run_id}-shard.npz").resolve()
    if shard_file.is_file():
        logger.debug(f"[{split}][{run_id}] shard file exists, skip.")
        return
    # ensure parent dir exists
    shard_file.parent.mkdir(parents=True, exist_ok=True)

    shard = None
    try:
        match = SIM_ATTR_PATTERN.match(run_id)
        if not match:
            raise ValueError(f"run_id '{run_id}' does not match expected pattern")

        sim_attrs = {
            "init_gc_radius": int(match.group(1)),
            "init_metallicity": int(match.group(2)),
            "init_mass_lv": int(match.group(3)),
            "init_pos": int(match.group(4)),
        }

        # construct shard file
        joblib_path = JOBLIB_ROOT / f"{run_id}-obs.joblib"
        if not joblib_path.exists():
            raise FileNotFoundError(f"Joblib file not found: {joblib_path}")
        raw_joblib_file = joblib.load(joblib_path)
        snapshot_arr = [
            {
                "feature": pd.DataFrame(snapshot["stars"])
                .loc[lambda df: df["is_within_2x_r_tidal"]]
                .reset_index(drop=True)[list(feature_keys)]
                .to_numpy(),
                "target": np.asarray([snapshot["header"][key] for key in target_keys]),
                "meta": np.asarray(
                    [
                        snapshot["header"]["time"],
                        snapshot["header"]["total_mass_within_2x_r_tidal"],
                        sim_attrs["init_gc_radius"],
                        sim_attrs["init_metallicity"],
                        sim_attrs["init_mass_lv"],
                        sim_attrs["init_pos"],
                    ]
                ),
            }
            for coord, series in raw_joblib_file.items()
            for timestamp, snapshot in series.items()
            if (len(pd.DataFrame(snapshot["stars"])) >= min_stars)
            and ("is_within_2x_r_tidal" in pd.DataFrame(snapshot["stars"]).columns)
        ]

        if not snapshot_arr:
            logger.warning(
                f"[{split}][{run_id}] No valid snapshots found (min_stars={min_stars})"
            )
            return

        shard = Shard(
            feature=np.concatenate([s["feature"] for s in snapshot_arr], axis=0),
            feature_keys=feature_keys,
            target=np.stack([s["target"] for s in snapshot_arr], axis=0),
            target_keys=target_keys,
            meta=np.stack([s["meta"] for s in snapshot_arr], axis=0),
            meta_keys=(
                "time",
                "total_mass_within_2x_r_tidal",
                "init_gc_radius",
                "init_metallicity",
                "init_mass_lv",
                "init_pos",
            ),
            ptr=np.cumsum(
                [0] + [s["feature"].shape[0] for s in snapshot_arr], dtype=np.int64
            ),
        )
        logger.debug(f"[{split}][{run_id}] writing shard file ...")

        shard.to_npz(shard_file)
        logger.debug(f"[{split}][{run_id}] shard file created: {shard_file}")
    except Exception as e:
        logger.exception(f"[{split}][{run_id}] Failed: {e!r}")
    finally:
        del shard
        gc.collect()


def _merge_per_run_shards(
    split: SplitType,
    dataset_dir: Path,
    run_ids: Sequence[str],
    log_file: Path | str | None = None,
) -> None:
    # setup logger
    setup_logger(
        (
            Path(log_file).resolve()
            if log_file is not None
            else (OUTPUT_BASE / "log" / "build_dataset.log").resolve()
        )
    )

    merged_shard_file = (dataset_dir / f"raw-{split}-shard.npz").resolve()
    if merged_shard_file.is_file():
        logger.info(f"[{split}] merged shard file exists, skip.")
        return

    try:
        cached_shard_files = sorted(
            (dataset_dir / split / f"{run_id}-shard.npz" for run_id in run_ids),
            key=lambda p: p.name,
        )
        if missing := [p for p in cached_shard_files if not p.is_file()]:
            raise FileNotFoundError(f"Missing shard files: {missing}")
    except Exception as e:
        raise RuntimeError(f"[{split}] Failed to fetch cached shard files: {e!r}")

    merged_shard = None
    try:
        with tqdm(
            total=len(cached_shard_files),
            desc=f"[{split}] Merging shards",
            unit="shard",
            dynamic_ncols=True,
            leave=False,
            position=1,
        ) as pbar:

            def _add(a, b):
                pbar.update(1)
                return a + b

            merged_shard = reduce(
                _add,
                (Shard.from_npz(shard_file) for shard_file in cached_shard_files),
            )
        logger.debug(f"[{split}] writing merged shard file ...")
        merged_shard.to_npz(merged_shard_file)
        logger.info(f"[{split}] merged shard file created: {merged_shard_file}")
    except Exception as e:
        logger.exception(f"[{split}] Failed: {e!r}")
    finally:
        per_run_dir = Path(dataset_dir) / split
        if per_run_dir.exists() and per_run_dir.is_dir():
            removed_count = 0
            for child in per_run_dir.iterdir():
                try:
                    if child.is_file() and (
                        child.name.endswith("-shard.npz")
                        or child.name.endswith(".tmp.npz")
                    ):
                        child.unlink()
                        removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {child}: {e}")
            logger.debug(f"[{split}] Cleaned up {removed_count} temporary shard files")
            try:
                per_run_dir.rmdir()
                logger.debug(f"[{split}] Removed temporary directory: {per_run_dir}")
            except OSError as e:
                logger.debug(f"Directory not empty or in use, left {per_run_dir}: {e}")
        del merged_shard
        gc.collect()


def _scale_raw_shard(
    feature_scaler_bundle: ArrayScalerBundle,
    target_scaler_bundle: ArrayScalerBundle,
    split: SplitType,
    dataset_dir: Path,
    log_file: Path | str | None = None,
) -> None:
    # setup logger
    setup_logger(
        (
            Path(log_file).resolve()
            if log_file is not None
            else (OUTPUT_BASE / "log" / "build_dataset.log").resolve()
        )
    )

    raw_shard_file = (Path(dataset_dir) / f"raw-{split}-shard.npz").resolve()
    if not raw_shard_file.exists():
        logger.error(f"[{split}] raw shard file not found: {raw_shard_file}")
        return

    scaled_shard_file = (Path(dataset_dir) / f"scaled-{split}-shard.npz").resolve()
    if scaled_shard_file.is_file():
        logger.info(f"[{split}] scaled shard exists, skip.")
        return

    shard = None
    try:
        shard = Shard.from_npz(raw_shard_file)

        scaled_feature = feature_scaler_bundle.transform(
            shard.feature, shard.feature_keys
        )
        scaled_target = target_scaler_bundle.transform(shard.target, shard.target_keys)

        scaled_shard = Shard(
            feature=scaled_feature,
            feature_keys=shard.feature_keys,
            target=scaled_target,
            target_keys=shard.target_keys,
            meta=shard.meta,
            meta_keys=shard.meta_keys,
            ptr=shard.pointer,
        )

        logger.debug(f"[{split}] writing scaled shard file ...")
        scaled_shard.to_npz(scaled_shard_file)
        logger.info(f"[{split}] scaled shard created: {scaled_shard_file}")
    except Exception as e:
        logger.exception(f"[{split}] Failed to scale shard: {e!r}")
    finally:
        del shard
        gc.collect()


def build_dataset(
    dataset_exp_path: Path | str,
    split_mft_json: Path | str,
    feature_keys: Sequence[str],
    target_keys: Sequence[str],
    feature_scaler_config: dict[tuple[str, ...], NormMethod],
    target_scaler_config: dict[tuple[str, ...], NormMethod],
    log_file: Path | str | None = None,
) -> None:
    # setup logger
    setup_logger(
        (
            Path(log_file).resolve()
            if log_file is not None
            else (OUTPUT_BASE / "log" / "build_dataset.log").resolve()
        )
    )

    dataset_exp_path = (
        (OUTPUT_BASE / dataset_exp_path)
        if dataset_exp_path is not None
        else (OUTPUT_BASE / "dataset")
    ).resolve()
    dataset_exp_path.mkdir(parents=True, exist_ok=True)

    with open(split_mft_json, "r") as f:
        manifest = {
            k: v for k, v in json.load(f).items() if k in ["train", "val", "test"]
        }

    # validate manifest completeness
    required_splits = {"train", "val", "test"}
    missing_splits = required_splits - set(manifest.keys())
    if missing_splits:
        raise ValueError(f"Manifest missing required splits: {missing_splits}")

    # validate that each split has data
    if empty_splits := [k for k, v in manifest.items() if not v]:
        raise ValueError(f"Manifest has empty splits: {empty_splits}")

    logger.info(
        f"Loaded split manifest from {split_mft_json}: { {k: len(v) for k, v in manifest.items()} }"
    )

    if splits := [
        split
        for split in ["train", "val", "test"]
        if not (dataset_exp_path / f"raw-{split}-shard.npz").is_file()
    ]:
        logger.info(f"Starting to collect shards for splits: {splits}")
        Parallel(n_jobs=30)(
            delayed(_cache_per_run_shard)(
                run_id=run_id,
                split=split,
                dataset_dir=dataset_exp_path,
                feature_keys=feature_keys,
                target_keys=target_keys,
                min_stars=10,
                log_file=log_file,
            )
            for split in tqdm(
                splits,
                unit="split",
                dynamic_ncols=True,
                leave=False,
                position=0,
            )
            for run_id in tqdm(
                manifest[split],
                unit="run",
                dynamic_ncols=True,
                leave=False,
                position=1,
            )
        )
        logger.info("All shards collected, continue to merging...")
    else:
        logger.info("All shards already collected, skip to merging...")

    Parallel(n_jobs=3)(
        delayed(_merge_per_run_shards)(
            split=split,
            dataset_dir=dataset_exp_path,
            run_ids=manifest[split],
            log_file=log_file,
        )
        for split in ["train", "val", "test"]
    )
    logger.info("Split shards merged, continue to scaling...")

    # validate that all merged shards exist
    for split in ["train", "val", "test"]:
        merged_path = dataset_exp_path / f"raw-{split}-shard.npz"
        if not merged_path.exists():
            raise FileNotFoundError(f"Expected merged shard not found: {merged_path}")

    logger.info("All merged shards validated.")

    # initialize scaler bundles from config
    feature_scaler_bundle = ArrayScalerBundle(feature_scaler_config)
    target_scaler_bundle = ArrayScalerBundle(target_scaler_config)

    # fit scalers using the TRAIN shard
    train_shard_path = dataset_exp_path / "raw-train-shard.npz"
    if not train_shard_path.exists():
        raise FileNotFoundError(f"Train shard not found: {train_shard_path}")

    train_shard = Shard.from_npz(train_shard_path)
    if len(train_shard) == 0:
        raise ValueError("Train shard is empty, cannot fit scalers")

    feature_scaler_bundle.fit(train_shard.feature, train_shard.feature_keys)
    target_scaler_bundle.fit(train_shard.target, train_shard.target_keys)
    del train_shard
    gc.collect()
    feature_scaler_bundle.to_joblib(dataset_exp_path / "feature_scaler_bundle.joblib")
    target_scaler_bundle.to_joblib(dataset_exp_path / "target_scaler_bundle.joblib")
    logger.info("Scalers fitted and saved.")

    try:
        dataset_config = {
            "manifest": manifest,
            "feature_keys": list(feature_keys),
            "target_keys": list(target_keys),
            "feature_scaler_config": feature_scaler_bundle.to_dict(),
            "target_scaler_config": target_scaler_bundle.to_dict(),
            "feature_scaler_joblib": "feature_scaler_bundle.joblib",
            "target_scaler_joblib": "target_scaler_bundle.joblib",
        }
        cfg_exp_path = dataset_exp_path / "dataset_config.json"
        with open(cfg_exp_path, "w") as ef:
            json.dump(dataset_config, ef, indent=2)
        logger.info(
            f"Exported combined split manifest and scaler config to {cfg_exp_path}"
        )
    except Exception as e:
        logger.exception(f"Failed to export combined JSON: {e!r}")

    for split in tqdm(
        ["train", "val", "test"],
        unit="split",
        dynamic_ncols=True,
        leave=False,
        position=0,
    ):
        _scale_raw_shard(
            feature_scaler_bundle=feature_scaler_bundle,
            target_scaler_bundle=target_scaler_bundle,
            split=split,
            dataset_dir=dataset_exp_path,
            log_file=log_file,
        )

    logger.info("Dataset building completed.")


@click.command()
@click.option(
    "--dataset-export-path",
    "dataset_export_path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Path to save the dataset. If omitted uses OUTPUT_BASE/dataset.",
)
@click.option(
    "--split-mft-json",
    "split_mft_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to JSON file containing train/val/test split metadata (run IDs).",
)
def main(dataset_export_path: Path, split_mft_json: Path) -> None:
    build_dataset(
        dataset_exp_path=dataset_export_path,
        split_mft_json=split_mft_json,
        feature_keys=(
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "lon_deg",
            "lat_deg",
            "pm_lon_coslat_mas_yr",
            "pm_lat_mas_yr",
            "log_L_L_sol",
        ),
        target_keys=(
            "time",
            "total_mass_within_2x_r_tidal",
        ),
        feature_scaler_config={
            ("x", "y", "z"): "robust",
            ("vx", "vy", "vz"): "robust",
            ("lon_deg", "lat_deg"): "robust",
            ("pm_lon_coslat_mas_yr", "pm_lat_mas_yr"): "robust",
            ("log_L_L_sol",): "robust",
        },
        target_scaler_config={
            ("time",): "robust",
            ("total_mass_within_2x_r_tidal",): "log10_standard",
        },
    )


if __name__ == "__main__":
    main()
