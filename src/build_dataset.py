import gc
import hashlib
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


def _calc_file_md5(file_path: Path, chunk_size: int = 8192) -> str:
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def _fetch_file_md5(
    file_path: Path,
    checksums: dict[str, str],
) -> str | None:
    if not file_path.is_file():
        return None

    file_name = file_path.name
    expected_md5 = checksums.get(file_name)
    if not expected_md5:
        # no stored checksum - cannot verify integrity, force rebuild
        logger.warning(f"Checksum missing for {file_name}; rebuild.")
        return None

    actual_md5 = _calc_file_md5(file_path)
    if actual_md5 != expected_md5:
        logger.warning(
            f"Checksum mismatch for {file_name}: expected {expected_md5}, got {actual_md5}"
        )
        return None
    return actual_md5


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
        logger.debug(f"[{split}][{run_id}] shard exists; skip.")
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
        feature_keys_list = list(feature_keys)
        target_keys_list = list(target_keys)
        snapshot_arr = []
        for coord, series in raw_joblib_file.items():
            for _timestamp, snapshot in series.items():
                df = pd.DataFrame(snapshot["stars"])
                if len(df) < min_stars:
                    continue
                if "is_within_2x_r_tidal" not in df.columns:
                    continue

                filtered = df.loc[df["is_within_2x_r_tidal"]].reset_index(drop=True)
                if filtered.empty:
                    continue

                snapshot_arr.append(
                    {
                        "feature": filtered[feature_keys_list].to_numpy(),
                        "target": np.asarray(
                            [snapshot["header"][key] for key in target_keys_list]
                        ),
                        "meta": np.asarray(
                            [
                                snapshot["header"]["time"],
                                snapshot["header"]["total_mass_within_2x_r_tidal"],
                                sim_attrs["init_gc_radius"],
                                sim_attrs["init_metallicity"],
                                sim_attrs["init_mass_lv"],
                                sim_attrs["init_pos"],
                                # (x, 0, 0) in pc
                                int(coord[0]),
                            ]
                        ),
                    }
                )

        if not snapshot_arr:
            logger.warning(
                f"[{split}][{run_id}] no valid snapshots (min_stars={min_stars})"
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
                "pseudo_obs_distance_pc",
            ),
            ptr=np.cumsum(
                [0] + [s["feature"].shape[0] for s in snapshot_arr], dtype=np.int64
            ),
        )
        logger.debug(f"[{split}][{run_id}] writing shard ...")

        shard.to_npz(shard_file)
        logger.debug(f"[{split}][{run_id}] shard saved: {shard_file}")
    except Exception as e:
        logger.exception(f"[{split}][{run_id}] Failed: {e!r}")
    finally:
        del shard
        gc.collect()


def _merge_per_run_shards(
    split: SplitType,
    dataset_dir: Path,
    run_ids: Sequence[str],
    checksums: dict[str, str] | None = None,
    log_file: Path | str | None = None,
) -> str | None:
    # setup logger
    setup_logger(
        (
            Path(log_file).resolve()
            if log_file is not None
            else (OUTPUT_BASE / "log" / "build_dataset.log").resolve()
        )
    )

    merged_shard_file = (dataset_dir / f"raw-{split}-shard.npz").resolve()
    if md5_hash := _fetch_file_md5(merged_shard_file, checksums or {}):
        logger.info(f"[{split}] merged shard valid; skip.")
        return md5_hash

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
    md5_hash = None
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
        logger.debug(f"[{split}] writing merged shard ...")
        merged_shard.to_npz(merged_shard_file)
        md5_hash = _calc_file_md5(merged_shard_file)
        logger.info(
            f"[{split}] merged shard saved: {merged_shard_file} (md5={md5_hash})"
        )
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
            logger.debug(f"[{split}] cleaned {removed_count} temp shards")
            try:
                per_run_dir.rmdir()
                logger.debug(f"[{split}] removed temp dir: {per_run_dir}")
            except OSError as e:
                logger.debug(f"Directory not empty or in use, left {per_run_dir}: {e}")
        del merged_shard
        gc.collect()
    return md5_hash


def _scale_raw_shard(
    feature_scaler_bundle: ArrayScalerBundle,
    target_scaler_bundle: ArrayScalerBundle,
    split: SplitType,
    dataset_dir: Path,
    checksums: dict[str, str] | None = None,
    log_file: Path | str | None = None,
) -> str | None:
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
    if md5_hash := _fetch_file_md5(scaled_shard_file, checksums or {}):
        logger.info(f"[{split}] scaled shard valid; skip.")
        return md5_hash

    shard = None
    md5_hash = None
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

        logger.debug(f"[{split}] writing scaled shard ...")
        scaled_shard.to_npz(scaled_shard_file)
        md5_hash = _calc_file_md5(scaled_shard_file)
        logger.info(
            f"[{split}] scaled shard saved: {scaled_shard_file} (md5={md5_hash})"
        )
    except Exception as e:
        logger.exception(f"[{split}] Failed to scale shard: {e!r}")
    finally:
        del shard
        gc.collect()
    return md5_hash


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
        f"Loaded split manifest {split_mft_json}: { {k: len(v) for k, v in manifest.items()} }"
    )

    # initialize scaler bundles from config
    feature_scaler_bundle = ArrayScalerBundle(feature_scaler_config)
    target_scaler_bundle = ArrayScalerBundle(target_scaler_config)

    # export combined manifest and scaler config
    cfg_exp_path = dataset_exp_path / "dataset_config.json"
    existing_checksums: dict[str, str] = {}
    try:
        dataset_config = {
            "manifest": manifest,
            "feature_keys": list(feature_keys),
            "target_keys": list(target_keys),
            "feature_scaler_config": feature_scaler_bundle.to_dict(),
            "target_scaler_config": target_scaler_bundle.to_dict(),
        }

        # if exists, check if consistent (excluding checksums which are computed later)
        if cfg_exp_path.is_file():
            with open(cfg_exp_path, "r") as ef:
                existing_config = json.load(ef)

            existing_checksums = existing_config.get("checksums", {})
            if diff_keys := [
                k
                for k in dataset_config.keys()
                if existing_config.get(k) != dataset_config[k]
            ]:
                raise ValueError(
                    f"Existing dataset_config.json differs from the new one on keys: {diff_keys}"
                )
            else:
                logger.info(f"Dataset config matches: {cfg_exp_path}")
        else:
            with open(cfg_exp_path, "w") as ef:
                json.dump(dataset_config, ef, indent=2)
            logger.info(f"Exported dataset config: {cfg_exp_path}")
    except Exception as e:
        logger.exception(f"Failed to export combined JSON: {e!r}")

    # collect per-run shards for splits that not yet have valid merged raw shards
    shard_collections = [
        split
        for split in ["train", "val", "test"]
        if not _fetch_file_md5(
            dataset_exp_path / f"raw-{split}-shard.npz", existing_checksums
        )
    ]

    if shard_collections:
        # determine missing per-run shards
        if missing_tasks := [
            (split, run_id)
            for split in shard_collections
            for run_id in manifest[split]
            if not (dataset_exp_path / split / f"{run_id}-shard.npz").is_file()
        ]:
            logger.info(
                f"Collecting {len(missing_tasks)} per-run shards for splits: {shard_collections}"
            )
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
                for split, run_id in tqdm(
                    missing_tasks,
                    unit="shard",
                    dynamic_ncols=True,
                    leave=False,
                )
            )
            logger.info("Per-run shards ready; merging.")
        else:
            logger.info("Per-run shards ready; merging.")
    else:
        logger.info("Merged raw shards valid; skip per-run collection.")

    # merge per-run shards into raw-{split}-shard.npz
    raw_shard_md5s = {}
    for split in ["train", "val", "test"]:
        md5_hash = _merge_per_run_shards(
            split=split,
            dataset_dir=dataset_exp_path,
            run_ids=manifest[split],
            checksums=existing_checksums,
            log_file=log_file,
        )
        if md5_hash:
            raw_shard_md5s[f"raw-{split}-shard.npz"] = md5_hash

    # validate that all merged shards exist
    for split in ["train", "val", "test"]:
        merged_path = dataset_exp_path / f"raw-{split}-shard.npz"
        if not merged_path.exists():
            raise FileNotFoundError(f"Expected merged shard not found: {merged_path}")

    logger.info("Merged raw shards ready; scaling.")

    # fit scalers if not already fitted
    feature_scaler_path = dataset_exp_path / "feature_scaler_bundle.joblib"
    target_scaler_path = dataset_exp_path / "target_scaler_bundle.joblib"

    scaler_md5s = {}
    if _fetch_file_md5(feature_scaler_path, existing_checksums) and _fetch_file_md5(
        target_scaler_path, existing_checksums
    ):
        logger.info("Scalers valid; loading.")
        feature_scaler_bundle = ArrayScalerBundle.from_joblib(feature_scaler_path)
        target_scaler_bundle = ArrayScalerBundle.from_joblib(target_scaler_path)
        # keep existing checksums
        scaler_md5s["feature_scaler_bundle.joblib"] = existing_checksums.get(
            "feature_scaler_bundle.joblib"
        )
        scaler_md5s["target_scaler_bundle.joblib"] = existing_checksums.get(
            "target_scaler_bundle.joblib"
        )
    else:
        logger.info("Fitting scalers on train shard.")
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

        feature_scaler_bundle.to_joblib(feature_scaler_path)
        target_scaler_bundle.to_joblib(target_scaler_path)
        scaler_md5s["feature_scaler_bundle.joblib"] = _calc_file_md5(
            feature_scaler_path
        )
        scaler_md5s["target_scaler_bundle.joblib"] = _calc_file_md5(target_scaler_path)
        logger.info(
            "Scalers saved (feature md5=%s, target md5=%s)",
            scaler_md5s["feature_scaler_bundle.joblib"],
            scaler_md5s["target_scaler_bundle.joblib"],
        )

    # scale raw shards if not already scaled
    scaled_shard_md5s = {}
    for split in ["train", "val", "test"]:
        md5_hash = _scale_raw_shard(
            feature_scaler_bundle=feature_scaler_bundle,
            target_scaler_bundle=target_scaler_bundle,
            split=split,
            dataset_dir=dataset_exp_path,
            checksums=existing_checksums,
            log_file=log_file,
        )
        if md5_hash:
            scaled_shard_md5s[f"scaled-{split}-shard.npz"] = md5_hash

    # update dataset_config.json with checksums
    try:
        cfg_exp_path = dataset_exp_path / "dataset_config.json"
        if cfg_exp_path.is_file():
            with open(cfg_exp_path, "r") as f:
                dataset_config = json.load(f)
        else:
            dataset_config = {}

        # Merge checksums (keep existing if not recomputed)
        existing_checksums = dataset_config.get("checksums", {})
        all_checksums = {
            **existing_checksums,
            **raw_shard_md5s,
            **scaled_shard_md5s,
            **scaler_md5s,
        }
        dataset_config["checksums"] = all_checksums

        with open(cfg_exp_path, "w") as f:
            json.dump(dataset_config, f, indent=2)
        logger.info(f"Updated checksums: {cfg_exp_path}")
    except Exception as e:
        logger.exception(f"Failed to update checksums in config: {e!r}")

    logger.info("Dataset build completed.")


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
