import gc
import json
import logging
import os
from functools import reduce
from pathlib import Path
from typing import Literal, Sequence

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
JOBLIB_ROOT = Path(os.getenv("JOBLIB_ROOT")).resolve()

logger = logging.getLogger(__name__)

SplitType = Literal["train", "val", "test"]


def _load_split_manifest(manifest_path: Path | str) -> dict:
    # load split manifest
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = {
            k: v for k, v in json.load(f).items() if k in ["train", "val", "test"]
        }
    return manifest


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
    tmp_shard_file = shard_file.with_suffix(".tmp.npz")
    if tmp_shard_file.exists():
        logger.debug(f"[{split}][{run_id}] removing temporary shard file ...")
        tmp_shard_file.unlink()
    # ensure parent dir exists
    shard_file.parent.mkdir(parents=True, exist_ok=True)

    shard = None
    try:
        # construct shard file
        raw_joblib_file = joblib.load(JOBLIB_ROOT / f"{run_id}-obs.joblib")
        snapshots = [
            {
                "feature": pd.DataFrame(snapshot["stars"])
                .loc[lambda df: df["is_within_2x_r_tidal"]]
                .reset_index(drop=True)[list(feature_keys)]
                .to_numpy(),
                "target": np.asarray([snapshot["header"][key] for key in target_keys]),
            }
            for coord, series in raw_joblib_file.items()
            for timestamp, snapshot in series.items()
            if (len(pd.DataFrame(snapshot["stars"])) >= min_stars)
            and ("is_within_2x_r_tidal" in pd.DataFrame(snapshot["stars"]).columns)
        ]
        shard = Shard(
            feature=np.concatenate([s["feature"] for s in snapshots], axis=0),
            feature_keys=feature_keys,
            target=np.stack([s["target"] for s in snapshots], axis=0),
            target_keys=target_keys,
            ptr=np.cumsum(
                [0] + [s["feature"].shape[0] for s in snapshots], dtype=np.int64
            ),
        )

        # atomic write
        logger.debug(f"[{split}][{run_id}] writing shard file ...")
        shard.to_npz(tmp_shard_file)
        tmp_shard_file.replace(shard_file)
        logger.debug(f"[{split}][{run_id}] shard file created: {shard_file}")
    except Exception as e:
        logger.exception(f"[{split}][{run_id}] Failed: {e!r}")
    finally:
        if tmp_shard_file.exists():
            tmp_shard_file.unlink()
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
    tmp_merged_shard_file = merged_shard_file.with_suffix(".tmp.npz")
    if tmp_merged_shard_file.exists():
        logger.debug(f"[{split}] removing temporary merged shard file ...")
        tmp_merged_shard_file.unlink()

    try:
        cached_shard_files = sorted(
            (dataset_dir / split / f"{run_id}-shard.npz" for run_id in run_ids),
            key=lambda p: p.name,
        )
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
        # atomic write
        logger.debug(f"[{split}] writing merged shard file ...")
        merged_shard.to_npz(tmp_merged_shard_file)
        tmp_merged_shard_file.replace(merged_shard_file)
        logger.info(f"[{split}] merged shard file created: {merged_shard_file}")
    except Exception as e:
        logger.exception(f"[{split}] Failed: {e!r}")
    finally:
        if tmp_merged_shard_file.exists():
            tmp_merged_shard_file.unlink()
        # remove per-run shard files to save space (safe, no rmtree)
        per_run_dir = Path(dataset_dir) / split
        if per_run_dir.exists() and per_run_dir.is_dir():
            for child in per_run_dir.iterdir():
                try:
                    if child.is_file() and (
                        child.name.endswith("-shard.npz")
                        or child.name.endswith(".tmp.npz")
                    ):
                        child.unlink()
                except Exception:
                    logger.debug(f"Failed to remove {child}", exc_info=True)
            try:
                per_run_dir.rmdir()
            except OSError:
                logger.debug(f"Directory not empty, left {per_run_dir}")
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
    tmp_scaled_shard_file = scaled_shard_file.with_suffix(".tmp.npz")
    if tmp_scaled_shard_file.exists():
        logger.debug(f"[{split}] removing temporary scaled shard file ...")
        tmp_scaled_shard_file.unlink()

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
            ptr=shard.pointer,
        )

        logger.debug(f"[{split}] writing scaled shard file ...")
        scaled_shard.to_npz(tmp_scaled_shard_file)
        tmp_scaled_shard_file.replace(scaled_shard_file)
        logger.info(f"[{split}] scaled shard created: {scaled_shard_file}")
    except Exception as e:
        logger.exception(f"[{split}] Failed to scale shard: {e!r}")
    finally:
        if tmp_scaled_shard_file.exists():
            tmp_scaled_shard_file.unlink()
        del shard
        gc.collect()


def build_dataset(
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

    dataset_dir = OUTPUT_BASE / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_split_manifest(manifest_path := Path(os.getenv("SPLIT_MFT_JSON")))
    logger.info(
        f"Loaded split manifest from {manifest_path}: { {k: len(v) for k, v in manifest.items()} }"
    )

    if splits := [
        split
        for split in ["train", "val", "test"]
        if not (dataset_dir / f"raw-{split}-shard.npz").is_file()
    ]:
        logger.info(f"Starting to collect shards for splits: {splits}")
        Parallel(n_jobs=30)(
            delayed(_cache_per_run_shard)(
                run_id=run_id,
                split=split,
                dataset_dir=dataset_dir,
                feature_keys=feature_keys,
                target_keys=target_keys,
                min_stars=10,
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
            dataset_dir=dataset_dir,
            run_ids=manifest[split],
        )
        for split in ["train", "val", "test"]
    )
    logger.info("Split shards merged, continue to scaling...")

    # initialize scaler bundles from config
    feature_scaler_bundle = ArrayScalerBundle(feature_scaler_config)
    target_scaler_bundle = ArrayScalerBundle(target_scaler_config)

    # fit scalers using the TRAIN shard
    train_shard = Shard.from_npz(dataset_dir / "raw-train-shard.npz")
    feature_scaler_bundle.fit(train_shard.feature, train_shard.feature_keys)
    target_scaler_bundle.fit(train_shard.target, train_shard.target_keys)
    del train_shard
    gc.collect()
    feature_scaler_bundle.to_joblib(dataset_dir / "feature_scaler_bundle.joblib")
    target_scaler_bundle.to_joblib(dataset_dir / "target_scaler_bundle.joblib")
    logger.info("Scalers fitted and saved.")

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
            dataset_dir=dataset_dir,
        )

    logger.info("Dataset building completed.")


if __name__ == "__main__":
    build_dataset(
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
