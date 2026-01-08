import hashlib
import json
import logging
import os
import shutil
from logging.handlers import RotatingFileHandler
from pathlib import Path

import dotenv
import joblib
import numpy as np
from dataset.collector import SnapshotCollector
from dataset.normalizer import Normalizer
from joblib import Parallel, delayed
from tqdm.auto import tqdm

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_gb(x: float) -> str:
    return f"{x:,.3f}"


def setup_logger(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(processName)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3),
        ],
        force=True,
    )


def _is_valid_npz(path: Path) -> bool:
    try:
        with np.load(path) as data:
            _ = list(data.keys())
        return True
    except Exception:
        return False


def dump_run_shard(
    raw_file: Path,
    output_path: Path,
    feature_keys: list[str],
    target_keys: list[str],
    n_sample_per_snapshot: int,
    n_star_per_sample: int,
    drop_probability: float,
    drop_ratio_range: tuple[float, float],
    seed: int,
) -> str | None:
    run_id = raw_file.stem.replace("-obs", "")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        collector = SnapshotCollector.from_joblib(raw_file)
        n_total_samples = len(collector) * n_sample_per_snapshot

        logger.debug(
            f"[dump] Processing run {run_id}: will produce {_fmt_int(n_total_samples)} samples -> {output_path.name}"
        )

        features = np.empty(
            (n_total_samples, n_star_per_sample, len(feature_keys)), dtype=np.float32
        )
        masks = np.empty((n_total_samples, n_star_per_sample), dtype=bool)
        targets = np.empty((n_total_samples, len(target_keys)), dtype=np.float32)

        for idx, sample in enumerate(
            collector.sample_iter(
                feature_keys=feature_keys,
                target_keys=target_keys,
                n_sample_per_snapshot=n_sample_per_snapshot,
                n_star_per_sample=n_star_per_sample,
                drop_probability=drop_probability,
                drop_ratio_range=drop_ratio_range,
                seed=(seed ^ (hash(run_id) & 0xFFFFFFFF)) & 0xFFFFFFFF,
            )
        ):
            features[idx] = sample["features"]
            masks[idx] = sample["valid_mask"]
            targets[idx] = sample["targets"]

        # atomic write
        temp_path = output_path.parent / f".tmp_{output_path.name}"
        np.savez_compressed(temp_path, features=features, masks=masks, targets=targets)
        temp_path.rename(output_path)

        logger.debug(f"[dump] Finished run {run_id}: wrote {output_path.name}")
        return run_id

    except Exception as e:
        logger.error(f"Failed to process {raw_file.name}: {e}")
        # cleanup potential partial files
        temp_path = output_path.parent / f".tmp_{output_path.name}"
        if temp_path.exists():
            temp_path.unlink()
        if output_path.exists():
            output_path.unlink()
        return None


def merge_split_shards(
    split_dir: Path,
    output_path: Path,
    feature_keys: list[str],
    target_keys: list[str],
    n_star_per_sample: int,
) -> bool:
    shard_files = sorted(split_dir.glob("*.npz"))

    logger.info(
        f"[merge] {split_dir.name}: found {_fmt_int(len(shard_files))} shard(s)"
    )

    if not shard_files:
        logger.warning(f"No shards found in {split_dir}, skipping merge.")
        return False

    # validate and filter out corrupted shards
    valid_shards = []
    for shard_file in shard_files:
        if _is_valid_npz(shard_file):
            valid_shards.append(shard_file)
        else:
            logger.warning(f"[merge] Removing corrupted shard: {shard_file.name}")
            shard_file.unlink()

    if not valid_shards:
        logger.warning(f"No valid shards in {split_dir}, skipping merge.")
        return False

    shard_files = valid_shards

    # collect metadata and validate consistency
    run_ids = []
    sample_counts = []

    for shard_file in shard_files:
        run_id = shard_file.stem
        run_ids.append(run_id)
        with np.load(shard_file) as data:
            n_stars = data["features"].shape[1]
            if n_stars != n_star_per_sample:
                raise ValueError(
                    f"Inconsistent n_stars in {shard_file.name}: expected {n_star_per_sample}, got {n_stars}"
                )
            n_samples = data["features"].shape[0]
            sample_counts.append(n_samples)
            logger.debug(
                f"[merge] {shard_file.name}: {_fmt_int(n_samples)} samples, n_stars={n_stars}"
            )

    total_samples = sum(sample_counts)
    n_features = len(feature_keys)
    n_targets = len(target_keys)

    logger.info(
        f"[merge] merging {_fmt_int(len(shard_files))} shards -> {output_path.name} ({_fmt_int(total_samples)} samples)"
    )

    bytes_features = (
        total_samples * n_star_per_sample * n_features * np.dtype(np.float32).itemsize
    )
    bytes_masks = total_samples * n_star_per_sample * np.dtype(bool).itemsize
    bytes_targets = total_samples * n_targets * np.dtype(np.float32).itemsize
    bytes_idx = total_samples * np.dtype(np.int32).itemsize * 2
    est_bytes = bytes_features + bytes_masks + bytes_targets + bytes_idx
    est_gb = est_bytes / (1024**3)
    logger.info(f"[merge] prealloc ~ {_fmt_gb(est_gb)} GB")

    all_features = np.empty(
        (total_samples, n_star_per_sample, n_features), dtype=np.float32
    )
    all_masks = np.empty((total_samples, n_star_per_sample), dtype=bool)
    all_targets = np.empty((total_samples, n_targets), dtype=np.float32)
    sample_to_run_idx = np.empty(total_samples, dtype=np.int32)
    sample_to_local_idx = np.empty(total_samples, dtype=np.int32)

    # load and concatenate shards
    offset = 0
    for run_idx, shard_file in enumerate(
        tqdm(
            shard_files,
            desc="Merging",
            unit="shard",
            dynamic_ncols=True,
        )
    ):
        logger.debug(
            f"[merge] {run_idx + 1}/{len(shard_files)} {shard_file.name} (offset {_fmt_int(offset)})"
        )
        with np.load(shard_file) as data:
            n_samples = data["features"].shape[0]
            end = offset + n_samples

            all_features[offset:end] = data["features"]
            all_masks[offset:end] = data["masks"]
            all_targets[offset:end] = data["targets"]

            sample_to_run_idx[offset:end] = run_idx
            sample_to_local_idx[offset:end] = np.arange(n_samples, dtype=np.int32)

            offset = end

    logger.info(f"[merge] writing {output_path.name}")

    # Atomic write for merged file
    temp_path = output_path.parent / f".tmp_{output_path.name}"
    np.savez_compressed(
        temp_path,
        features=all_features,
        masks=all_masks,
        targets=all_targets,
        run_ids=np.array(run_ids, dtype=object),
        sample_to_run_idx=sample_to_run_idx,
        sample_to_local_idx=sample_to_local_idx,
        feature_keys=np.array(feature_keys),
        target_keys=np.array(target_keys),
    )
    temp_path.rename(output_path)

    logger.info(
        f"[merge] merged -> {output_path.name} ({_fmt_int(total_samples)} samples)"
    )
    return True


def normalize(
    cache_root: Path,
    splits: List[str],
    feature_keys: List[str],
    target_keys: List[str],
    feature_normalizers: List[Normalizer],
    target_normalizers: List[Normalizer],
) -> None:
    normalizers_path = cache_root / "normalizers.joblib"

    # validate existing normalized files
    all_normalized_valid = True
    for split in splits:
        normalized_path = cache_root / f"{split}.npz"
        if normalized_path.exists():
            if not _is_valid_npz(normalized_path):
                logger.warning(
                    f"[normalize] Corrupted normalized file detected, removing: {normalized_path.name}"
                )
                normalized_path.unlink()
                all_normalized_valid = False
        else:
            all_normalized_valid = False

    # check if normalizers as well as all normalized files exist and valid
    if all_normalized_valid and normalizers_path.exists():
        logger.info("[normalize] Normalization phase already completed, skipping.")
        return

    feature_key_to_idx = {k: i for i, k in enumerate(feature_keys)}
    target_key_to_idx = {k: i for i, k in enumerate(target_keys)}

    # load training raw data
    train_raw_path = cache_root / "train-raw.npz"
    if not train_raw_path.exists():
        raise FileNotFoundError(
            f"Training raw file not found: {train_raw_path}. "
            "Ensure data collection and merging phases completed successfully."
        )

    if not _is_valid_npz(train_raw_path):
        raise ValueError(
            f"Training raw file is corrupted: {train_raw_path}. "
            "Delete it and re-run to regenerate."
        )

    logger.info(f"[normalize] Loading training data from {train_raw_path.name}")
    with np.load(train_raw_path, allow_pickle=True) as data:
        train_features = data["features"]  # (n_samples, n_stars, n_features)
        train_targets = data["targets"]  # (n_samples, n_targets)

    logger.info(
        f"[normalize] Training data: features ({_fmt_int(train_features.shape[0])}, {train_features.shape[1]}, {train_features.shape[2]}), targets ({_fmt_int(train_targets.shape[0])}, {train_targets.shape[1]})"
    )

    # fit feature normalizers
    logger.info(
        f"[normalize] Fitting {len(feature_normalizers)} feature normalizer(s) on training data"
    )
    for norm in feature_normalizers:
        col_indices = [feature_key_to_idx[c] for c in norm.columns]
        # Extract columns: shape (n_samples, n_stars, len(columns))
        subset = train_features[:, :, col_indices]
        norm.fit(subset)
        logger.info(f"[normalize] - Fitted: {norm}")

    # fit target normalizers
    logger.info(
        f"[normalize] Fitting {len(target_normalizers)} target normalizer(s) on training data"
    )
    for norm in target_normalizers:
        col_indices = [target_key_to_idx[c] for c in norm.columns]
        # extract columns: shape (n_samples, len(columns))
        subset = train_targets[:, col_indices]
        norm.fit(subset)
        logger.info(f"[normalize] - Fitted: {norm}")

    del train_features, train_targets

    # save normalizer
    logger.info(f"[normalize] Saving normalizers to {normalizers_path.name}")
    normalizer_data = {
        "feature_normalizers": [n.to_dict() for n in feature_normalizers],
        "target_normalizers": [n.to_dict() for n in target_normalizers],
        "feature_keys": feature_keys,
        "target_keys": target_keys,
    }
    joblib.dump(normalizer_data, normalizers_path)

    # apply normalization to all splits
    for split in splits:
        raw_path = cache_root / f"{split}-raw.npz"
        normalized_path = cache_root / f"{split}.npz"

        if normalized_path.exists() and _is_valid_npz(normalized_path):
            logger.info(
                f"[normalize] {split}: normalized file already exists, skipping."
            )
            continue

        if not raw_path.exists():
            logger.warning(f"[normalize] {split}: raw file not found, skipping.")
            continue

        if not _is_valid_npz(raw_path):
            logger.warning(f"[normalize] {split}: raw file is corrupted, skipping.")
            continue

        logger.info(f"[normalize] {split}: loading raw data")
        with np.load(raw_path, allow_pickle=True) as data:
            features = data["features"].copy()  # (n_samples, n_stars, n_features)
            masks = data["masks"].copy()  # (n_samples, n_stars)
            targets = data["targets"].copy()  # (n_samples, n_targets)
            run_ids = data["run_ids"].copy()
            sample_to_run_idx = data["sample_to_run_idx"].copy()
            sample_to_local_idx = data["sample_to_local_idx"].copy()
            loaded_feature_keys = data["feature_keys"].copy()
            loaded_target_keys = data["target_keys"].copy()

        logger.info(
            f"[normalize] {split}: applying normalization to {_fmt_int(features.shape[0])} samples"
        )

        # apply feature normalizers
        for norm in feature_normalizers:
            col_indices = [feature_key_to_idx[c] for c in norm.columns]
            # Extract columns: shape (n_samples, n_stars, len(columns))
            subset = features[:, :, col_indices]
            features[:, :, col_indices] = norm.transform(subset)

        # apply target normalizers
        for norm in target_normalizers:
            col_indices = [target_key_to_idx[c] for c in norm.columns]
            # extract columns: shape (n_samples, len(columns))
            subset = targets[:, col_indices]
            targets[:, col_indices] = norm.transform(subset)

        logger.info(f"[normalize] {split}: writing {normalized_path.name}")

        # atomic write
        temp_path = normalized_path.parent / f".tmp_{normalized_path.name}"
        np.savez_compressed(
            temp_path,
            features=features,
            masks=masks,
            targets=targets,
            run_ids=run_ids,
            sample_to_run_idx=sample_to_run_idx,
            sample_to_local_idx=sample_to_local_idx,
            feature_keys=loaded_feature_keys,
            target_keys=loaded_target_keys,
        )
        temp_path.rename(normalized_path)

        logger.info(f"[normalize] {split}: done")

    logger.info("[normalize] Normalization phase complete.")


def build_dataset(
    manifests: Dict[str, List[str]],
    n_sample_per_snapshot: int,
    n_star_per_sample: int,
    feature_keys: List[str],
    target_keys: List[str],
    feature_normalizers: List[Normalizer],
    target_normalizers: List[Normalizer],
    drop_probability: float,
    drop_ratio_range: Tuple[float, float],
    seed: int,
    export_path: Path,
) -> None:
    export_path.mkdir(parents=True, exist_ok=True)
    setup_logger((export_path / "build_dataset.log"))

    cfg_dict = {
        "feature_keys": feature_keys,
        "target_keys": target_keys,
        "feature_normalizers": [
            {"columns": sorted(norm.columns), "method": norm.method}
            for norm in feature_normalizers
        ],
        "target_normalizers": [
            {"columns": sorted(norm.columns), "method": norm.method}
            for norm in target_normalizers
        ],
        "n_sample_per_snapshot": n_sample_per_snapshot,
        "n_star_per_sample": n_star_per_sample,
        "drop_probability": drop_probability,
        "drop_ratio_range": drop_ratio_range,
        "seed": seed,
        "manifest_summary": {
            split: hashlib.sha256(",".join(sorted(run_ids)).encode()).hexdigest()[:12]
            for split, run_ids in manifests.items()
        },
    }
    cfg_hash = hashlib.sha256(
        json.dumps(cfg_dict, sort_keys=True).encode()
    ).hexdigest()[:12]

    logger.info(f"Dataset build configuration: {json.dumps(cfg_dict, indent=2)}")
    logger.info(f"Dataset configuration hash: {cfg_hash}")

    cache_root = (export_path / f"{cfg_hash}").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    # save both configuration and full manifest
    meta_dict = ({**cfg_dict, "manifests": manifests},)
    with (cache_root / "meta.json").open("w") as f:
        json.dump(meta_dict, f, indent=2)
    logger.info(f"Saved dataset configuration to {cache_root / 'meta.json'}")

    splits = list(manifests.keys())

    # process each split: generate shards -> merge -> cleanup
    for split in splits:
        split_dir = cache_root / split
        merged_raw_file = cache_root / f"{split}-raw.npz"

        # check and validate existing merged file
        if merged_raw_file.exists():
            if _is_valid_npz(merged_raw_file):
                logger.info(
                    f"[{split}] Raw file already exists, proceeding to next split."
                )
                continue
            else:
                logger.warning(f"[{split}] Corrupted raw file detected, removing.")
                merged_raw_file.unlink()
                # fall through to regenerate

        split_dir.mkdir(parents=True, exist_ok=True)

        run_ids = manifests[split]

        # cleanup any corrupted shards first
        for shard_file in list(split_dir.glob("*.npz")):
            if not _is_valid_npz(shard_file):
                logger.warning(f"[{split}] Removing corrupted shard: {shard_file.name}")
                shard_file.unlink()

        if (obs_dir := os.getenv("OBS_JOBLIB_DIR")) is None:
            raise ValueError("Environment variable 'OBS_JOBLIB_DIR' is not set")

        pending = [
            (raw_file, split_dir / f"{run_id}.npz")
            for run_id in run_ids
            if (raw_file := Path(obs_dir) / f"{run_id}-obs.joblib").exists()
            and not (split_dir / f"{run_id}.npz").exists()
        ]

        n_existing = sum(1 for _ in split_dir.glob("*.npz"))

        if not pending:
            if n_existing > 0:
                logger.info(
                    f"[{split}] All {_fmt_int(n_existing)} shards exist, proceeding to merge."
                )
        else:
            logger.info(
                f"[{split}] Processing {_fmt_int(len(pending))} shards ({_fmt_int(n_existing)} already done)..."
            )

            Parallel(n_jobs=-1, backend="multiprocessing")(
                tqdm(
                    [
                        delayed(dump_run_shard)(
                            raw_file=raw_file,
                            output_path=output_path,
                            feature_keys=feature_keys,
                            target_keys=target_keys,
                            n_sample_per_snapshot=n_sample_per_snapshot,
                            n_star_per_sample=n_star_per_sample,
                            drop_probability=drop_probability,
                            drop_ratio_range=drop_ratio_range,
                            seed=seed,
                        )
                        for raw_file, output_path in pending
                    ],
                    desc=f"Shards [{split}]",
                    unit="file",
                    dynamic_ncols=True,
                )
            )

            n_completed = sum(1 for _ in split_dir.glob("*.npz"))
            logger.info(
                f"[{split}] Data collection phase complete. {_fmt_int(n_completed)} shards processed."
            )

        # merge shards into raw file
        if not split_dir.exists() or not any(split_dir.glob("*.npz")):
            logger.warning(f"[{split}] No shards found, skipping merge.")
            continue

        success = merge_split_shards(
            split_dir=split_dir,
            output_path=merged_raw_file,
            feature_keys=feature_keys,
            target_keys=target_keys,
            n_star_per_sample=n_star_per_sample,
        )

        # cleanup individual shards after successful merge
        if success:
            if split_dir.exists() and split_dir.is_dir():
                shutil.rmtree(split_dir)
                logger.info(f"[{split}] Removed shard directory: {split_dir}")
        else:
            logger.warning(f"[{split}] Merge failed, shard directory not removed.")

    # fit and apply normalizers
    normalize(
        cache_root=cache_root,
        splits=splits,
        feature_keys=feature_keys,
        target_keys=target_keys,
        feature_normalizers=feature_normalizers,
        target_normalizers=target_normalizers,
    )


if __name__ == "__main__":
    FEATURE_KEYS = [
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
    ]
    TARGET_KEYS = ["time", "total_mass_within_2x_r_tidal"]

    if (mft_json_path := os.getenv("MFT_JSON")) is None:
        raise ValueError("Environment variable 'MFT_JSON' is not set")

    with Path(mft_json_path).resolve().open("r") as f:
        manifest_dict = json.load(f)

    build_dataset(
        manifests=manifest_dict,
        n_sample_per_snapshot=6,
        n_star_per_sample=48,
        feature_keys=FEATURE_KEYS,
        target_keys=TARGET_KEYS,
        feature_normalizers=[
            Normalizer(columns=["x", "y", "z"], method="z-score"),
            Normalizer(columns=["vx", "vy", "vz"], method="z-score"),
            Normalizer(columns=["lon_deg", "lat_deg"], method="z-score"),
            Normalizer(
                columns=["pm_lon_coslat_mas_yr", "pm_lat_mas_yr"], method="z-score"
            ),
            Normalizer(columns=["log_L_L_sol"], method="z-score"),
        ],
        target_normalizers=[
            Normalizer(columns=["time"], method="z-score"),
            Normalizer(columns=["total_mass_within_2x_r_tidal"], method="log"),
        ],
        drop_probability=0.2,
        drop_ratio_range=(0.15, 0.25),
        seed=42,
        export_path=Path("../cache"),
    )
