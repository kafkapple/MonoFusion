from __future__ import annotations

import shutil
from pathlib import Path

import tyro


def prepare_images(
    raw_root: Path = Path("_raw_data"),
    output_root: Path = Path("data/images"),
    name_index: int = 1,
    overwrite: bool = False,
    cameras: list[str] | None = None,
) -> None:
    """Materialize undistorted frames into the expected data/images layout.

    The raw capture layout is assumed to follow the structure
        raw_root/SEQ_NAME/undist_processed_frames/undist_camXX/*.jpg

    For each SEQ_NAME we generate
        output_root/{SEQ_NAME.split('_')[name_index]}_undist_camXX/
    mirroring the files under each camera directory by copying files.
    """

    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root '{raw_root}' does not exist")

    output_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    if not seq_dirs:
        print(f"No sequences found under {raw_root}")
        return

    for seq_dir in seq_dirs:
        frames_root = seq_dir / "undist_processed_frames"
        if not frames_root.exists():
            print(f"Skipping {seq_dir.name}: missing 'undist_processed_frames'")
            continue

        parts = seq_dir.name.split("_")
        if name_index >= len(parts):
            raise ValueError(
                f"Cannot index into sequence name '{seq_dir.name}' with name_index={name_index}"
            )
        short_name = parts[name_index]

        if cameras is None:
            camera_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
        else:
            camera_dirs = [frames_root / cam for cam in cameras]

        for cam_dir in camera_dirs:
            if not cam_dir.exists():
                print(f"Skipping missing camera directory {cam_dir}")
                continue

            dest_dir = output_root / f"{short_name}_{cam_dir.name}"
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(cam_dir.glob("*")):
                if not img_path.is_file():
                    continue

                dest_path = dest_dir / img_path.name
                if dest_path.exists():
                    if overwrite:
                        dest_path.unlink()
                    else:
                        # Skip existing entries when not overwriting
                        continue

                shutil.copy2(img_path, dest_path)


if __name__ == "__main__":
    tyro.cli(prepare_images)
