from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import tyro


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ResizeConfig:
    raw_root: Path = Path("_raw_data")
    """Root directory containing per-sequence folders."""

    output_root: Path = Path("data/images_resized")
    """Destination directory for resized images."""

    name_index: int = 1
    """Index into the sequence name split on '_' to derive the short name."""

    width: int = 512
    height: int = 288
    overwrite: bool = False
    cameras: tuple[str, ...] | None = None


def _iter_sequences(raw_root: Path, sequences: Iterable[str] | None) -> list[Path]:
    if sequences:
        seq_paths = []
        for seq in sequences:
            seq_path = raw_root / seq
            if not seq_path.is_dir():
                raise FileNotFoundError(f"Sequence '{seq}' not found under {raw_root}")
            seq_paths.append(seq_path)
        return seq_paths

    return sorted([p for p in raw_root.iterdir() if p.is_dir()])


def _numeric_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return (int(stem), stem)
    except ValueError:
        return (10**9, stem)


def _resize_image(src: Path, dest: Path, size: tuple[int, int]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    resample = getattr(Image, "Resampling", Image).LANCZOS  # Pillow >=9 compatibility
    with Image.open(src) as img:
        resized = img.resize(size, resample=resample)
        if resized.mode in {"RGBA", "P", "LA"}:
            resized = resized.convert("RGB")
        resized.save(dest)


def resize_sequences(config: ResizeConfig, sequences: Iterable[str] | None = None) -> None:
    raw_root = config.raw_root.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve()

    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root '{raw_root}' does not exist")

    output_root.mkdir(parents=True, exist_ok=True)
    size = (config.width, config.height)

    seq_dirs = _iter_sequences(raw_root, sequences)
    if not seq_dirs:
        print(f"No sequences found under {raw_root}")
        return

    for seq_dir in seq_dirs:
        frames_root = seq_dir / "undist_processed_frames"
        if not frames_root.exists():
            print(f"Skipping {seq_dir.name}: missing 'undist_processed_frames'")
            continue

        parts = seq_dir.name.split("_")
        short_name = parts[config.name_index] if config.name_index < len(parts) else seq_dir.name

        if config.cameras is None:
            camera_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
        else:
            camera_dirs = [frames_root / cam for cam in config.cameras]

        for cam_dir in camera_dirs:
            if not cam_dir.exists():
                print(f"Skipping missing camera directory {cam_dir}")
                continue

            dest_dir = output_root / f"{short_name}_{cam_dir.name}"

            img_paths = sorted(
                [p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS],
                key=_numeric_key,
            )
            if not img_paths:
                print(f"No images found in {cam_dir}")
                continue

            print(f"Processing {cam_dir.relative_to(raw_root)} -> {dest_dir}")
            for img_path in img_paths:
                dest_path = dest_dir / img_path.name
                if dest_path.exists() and not config.overwrite:
                    continue
                _resize_image(img_path, dest_path, size)


def main(
    config: ResizeConfig = ResizeConfig(),
    sequences: tuple[str, ...] = (),
) -> None:
    resize_sequences(config, sequences)


if __name__ == "__main__":
    tyro.cli(main)
