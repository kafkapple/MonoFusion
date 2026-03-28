"""
Validate MonoFusion preprocessing outputs for M5t2 PoC.

Checks file counts, shapes, dtypes, and value ranges for each
preprocessing stage: dust3r, raw_moge_depth, aligned_moge_depth, dinov2_features.

Usage:
    python validate_preprocessing.py \
        --data_root /node_data/joon/data/monofusion/m5t2_poc
"""
import argparse
import sys
from pathlib import Path

import numpy as np

CAMERAS = [f"m5t2_undist_cam{i:02d}" for i in range(4)]
MOGE_CAMERAS = [f"m5t2_cam{i:02d}" for i in range(4)]
EXPECTED_FRAMES = 60


def check_files(directory: Path, ext: str, expected: int, label: str) -> list[str]:
    """Check file count and return list of issues."""
    issues = []
    if not directory.exists():
        issues.append(f"CRITICAL: {label} — directory missing: {directory}")
        return issues

    # Only real files, not symlinks
    files = sorted(f for f in directory.glob(f"*{ext}") if f.is_file() and not f.is_symlink())
    if len(files) != expected:
        severity = "CRITICAL" if len(files) == 0 else "WARNING"
        issues.append(f"{severity}: {label} — expected {expected} files, found {len(files)}")
    return issues


def validate_npy_dir(
    directory: Path,
    label: str,
    expected: int,
    expected_ndim: int | None = None,
    check_positive: bool = False,
) -> tuple[list[str], dict]:
    """Validate .npy files in a directory for shape/dtype/value consistency."""
    issues = []
    stats = {"count": 0, "shapes": set(), "dtypes": set(), "nan": 0, "inf": 0, "neg": 0}

    if not directory.exists():
        issues.append(f"CRITICAL: {label} — directory missing: {directory}")
        return issues, stats

    files = sorted(f for f in directory.glob("*.npy") if f.is_file() and not f.is_symlink())
    stats["count"] = len(files)

    if len(files) == 0:
        issues.append(f"CRITICAL: {label} — no .npy files found")
        return issues, stats

    if len(files) != expected:
        issues.append(f"WARNING: {label} — expected {expected}, found {len(files)}")

    for f in files:
        try:
            arr = np.load(f)
        except Exception as e:
            issues.append(f"CRITICAL: {label}/{f.name} — failed to load: {e}")
            continue

        stats["shapes"].add(arr.shape)
        stats["dtypes"].add(str(arr.dtype))

        if expected_ndim is not None and arr.ndim != expected_ndim:
            issues.append(f"WARNING: {label}/{f.name} — ndim={arr.ndim}, expected {expected_ndim}")

        nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
        inf_count = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
        if nan_count:
            stats["nan"] += 1
            issues.append(f"CRITICAL: {label}/{f.name} — {nan_count} NaN values")
        if inf_count:
            stats["inf"] += 1
            issues.append(f"CRITICAL: {label}/{f.name} — {inf_count} Inf values")

        if check_positive and np.issubdtype(arr.dtype, np.floating):
            neg_count = int((arr <= 0).sum())
            if neg_count > 0:
                stats["neg"] += 1
                issues.append(f"WARNING: {label}/{f.name} — {neg_count} non-positive depth values")

    return issues, stats


def validate_npz_dir(directory: Path, label: str, expected: int) -> tuple[list[str], dict]:
    """Validate .npz pointcloud files."""
    issues = []
    stats = {"count": 0, "keys": set()}

    if not directory.exists():
        issues.append(f"CRITICAL: {label} — directory missing: {directory}")
        return issues, stats

    files = sorted(f for f in directory.glob("*.npz") if f.is_file() and not f.is_symlink())
    stats["count"] = len(files)

    if len(files) != expected:
        severity = "CRITICAL" if len(files) == 0 else "WARNING"
        issues.append(f"{severity}: {label} — expected {expected}, found {len(files)}")

    for f in files[:3]:  # Spot-check first 3 files
        try:
            data = np.load(f)
            stats["keys"].update(data.files)
            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.floating):
                    if np.isnan(arr).any():
                        issues.append(f"CRITICAL: {label}/{f.name}[{key}] — contains NaN")
                    if np.isinf(arr).any():
                        issues.append(f"CRITICAL: {label}/{f.name}[{key}] — contains Inf")
        except Exception as e:
            issues.append(f"CRITICAL: {label}/{f.name} — failed to load: {e}")

    return issues, stats


def print_stage_report(title: str, issues: list[str], stats: dict | None = None):
    """Print a formatted report block for one stage."""
    critical = sum(1 for i in issues if i.startswith("CRITICAL"))
    warnings = sum(1 for i in issues if i.startswith("WARNING"))
    status = "PASS" if critical == 0 and warnings == 0 else ("FAIL" if critical > 0 else "WARN")

    print(f"\n{'='*60}")
    print(f"  [{status}] {title}")
    print(f"{'='*60}")

    if stats:
        print(f"  Files: {stats.get('count', '?')}")
        if "shapes" in stats and stats["shapes"]:
            print(f"  Shapes: {stats['shapes']}")
        if "dtypes" in stats and stats["dtypes"]:
            print(f"  Dtypes: {stats['dtypes']}")
        if "keys" in stats and stats["keys"]:
            print(f"  NPZ keys: {stats['keys']}")

    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  All checks passed.")


def main():
    parser = argparse.ArgumentParser(description="Validate MonoFusion preprocessing outputs")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Root data directory (e.g. /node_data/joon/data/monofusion/m5t2_poc)")
    args = parser.parse_args()

    root = args.data_root
    if not root.exists():
        print(f"CRITICAL: data_root does not exist: {root}")
        sys.exit(1)

    all_issues = []
    print(f"\nMonoFusion Preprocessing Validation")
    print(f"Data root: {root}")

    # --- Stage 1: dust3r ---
    for cam in CAMERAS:
        for subdir, ext, ndim, pos in [
            ("depth", ".npy", 2, True),
            ("confidence", ".npy", 2, False),
        ]:
            d = root / "dust3r" / "m5t2" / cam / subdir
            label = f"dust3r/m5t2/{cam}/{subdir}"
            iss, st = validate_npy_dir(d, label, EXPECTED_FRAMES, ndim, pos)
            all_issues.extend(iss)
            print_stage_report(label, iss, st)

        # Pointclouds
        pc_dir = root / "dust3r" / "m5t2" / cam / "pointcloud"
        label = f"dust3r/m5t2/{cam}/pointcloud"
        iss, st = validate_npz_dir(pc_dir, label, EXPECTED_FRAMES)
        all_issues.extend(iss)
        print_stage_report(label, iss, st)

    # --- Stage 2: raw_moge_depth ---
    for cam in MOGE_CAMERAS:
        d = root / "raw_moge_depth" / cam / "depth"
        label = f"raw_moge_depth/{cam}/depth"
        iss, st = validate_npy_dir(d, label, EXPECTED_FRAMES, expected_ndim=2, check_positive=True)
        all_issues.extend(iss)
        print_stage_report(label, iss, st)

    # --- Stage 3: aligned_moge_depth ---
    for cam in CAMERAS:
        d = root / "aligned_moge_depth" / "m5t2" / cam
        label = f"aligned_moge_depth/m5t2/{cam}"
        iss, st = validate_npy_dir(d, label, EXPECTED_FRAMES, expected_ndim=2, check_positive=True)
        all_issues.extend(iss)
        print_stage_report(label, iss, st)

    # --- Stage 4: dinov2_features ---
    for cam in MOGE_CAMERAS:
        d = root / "dinov2_features" / cam
        label = f"dinov2_features/{cam}"
        iss, st = validate_npy_dir(d, label, EXPECTED_FRAMES)
        all_issues.extend(iss)
        print_stage_report(label, iss, st)

    # --- Summary ---
    critical = sum(1 for i in all_issues if i.startswith("CRITICAL"))
    warnings = sum(1 for i in all_issues if i.startswith("WARNING"))

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {critical} critical, {warnings} warnings")
    print(f"{'='*60}")

    if critical > 0:
        print("\nCritical issues found. Preprocessing may be incomplete.")
        sys.exit(1)
    elif warnings > 0:
        print("\nWarnings found. Review above for details.")
        sys.exit(0)
    else:
        print("\nAll preprocessing stages validated successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
