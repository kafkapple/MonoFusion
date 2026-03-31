"""
Pre-training validation: 5 critical silent-bug checks for MonoFusion M5t2.

Checks identified from MoA deliberation (2026-03-29):
  1. Coordinate system consistency (camera reprojection round-trip)
  2. Frame index alignment (track file → dataset loader t mapping)
  3. occ_logit / expected_dist interpretation (parse_tapir_track_info)
  4. Scene scale normalization (camera translations vs Gaussian init scale)
  5. cam02 empty track handling (0% FG visible — must not crash loss)

Usage:
    python validate_training_inputs.py \
        --data_root /node_data/joon/data/monofusion/m5t2_poc
"""
import argparse
import sys
from pathlib import Path

from validate_checks import (
    check_coordinate_system,
    check_frame_index_alignment,
    check_track_interpretation,
    check_scene_scale,
    check_empty_camera,
)


def main():
    parser = argparse.ArgumentParser(description="Pre-training validation: 5 critical silent-bug checks")
    parser.add_argument("--data_root", type=Path, required=True)
    args = parser.parse_args()

    root = args.data_root
    if not root.exists():
        print(f"CRITICAL: data_root not found: {root}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  MonoFusion Pre-Training Validation")
    print(f"  Data root: {root}")
    print(f"{'='*60}\n")

    all_issues = []

    for label, fn in [
        ("1. Coordinate System", check_coordinate_system),
        ("2. Frame Index Alignment", check_frame_index_alignment),
        ("3. occ_logit / expected_dist", check_track_interpretation),
        ("4. Scene Scale", check_scene_scale),
        ("5. cam02 Empty Handling", check_empty_camera),
    ]:
        print(f"\n── {label} ──")
        iss = fn(root)
        all_issues.extend(iss)
        for i in iss:
            print(f"  {i}")

    critical = sum(1 for i in all_issues if i.startswith("CRITICAL"))
    warnings  = sum(1 for i in all_issues if i.startswith("WARNING"))

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {critical} critical, {warnings} warnings")
    print(f"{'='*60}")
    sys.exit(1 if critical > 0 else 0)


if __name__ == "__main__":
    main()
