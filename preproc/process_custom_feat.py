from __future__ import annotations

import glob
import shlex
import subprocess
from pathlib import Path

import tyro


def _quoted(path: str) -> str:
    return shlex.quote(path)


def _run_command(cmd: str) -> None:
    print(cmd)
    subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)


def _find_named_ancestor(path: Path, target_name: str) -> Path | None:
    """Return the nearest ancestor (including the path itself) matching target_name."""
    for candidate in (path,) + tuple(path.parents):
        if candidate.name == target_name:
            return candidate
    return None


def _short_name_from_img_dir(img_path: Path) -> str:
    stem = img_path.name
    if not stem:
        raise ValueError(f"Cannot derive short name from '{img_path}'")
    return stem.split("_")[0]


def _build_short_to_seq_map(raw_root: Path, name_index: int) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for seq_dir in sorted(raw_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        parts = seq_dir.name.split("_")
        if name_index >= len(parts):
            continue
        key = parts[name_index]
        mapping.setdefault(key, []).append(seq_dir.name)
    return mapping


def _build_dust3r_cli_args(
    *,
    seq_name: str,
    img_dir: str,
    mask_dir: str,
    dust3r_dir: str,
    img_name: str,
    mask_name: str,
    model_name: str,
    batch_size: int,
    schedule: str,
    niter: int,
    learning_rate: float,
    image_size: int,
    frame_step: int,
    frame_count: int,
    device: str,
    output_dirname: str,
    raw_root: Path,
    mask_root_override: Path | None,
    processing_root: Path,
) -> str:
    img_path = Path(img_dir).resolve()
    image_root = _find_named_ancestor(img_path, img_name)
    if image_root is None:
        raise ValueError(f"Could not locate '{img_name}' ancestor for {img_dir}")

    if mask_root_override is not None:
        mask_root = mask_root_override
    else:
        mask_root = _find_named_ancestor(Path(mask_dir).resolve(), mask_name)

    output_root = Path(dust3r_dir).resolve().parent

    args: list[str] = [
        f"--seq-name {_quoted(seq_name)}",
        f"--model-name {_quoted(model_name)}",
        f"--batch-size {batch_size}",
        f"--schedule {_quoted(schedule)}",
        f"--learning-rate {learning_rate}",
        f"--niter {niter}",
        f"--image-size {image_size}",
        f"--frame-step {frame_step}",
        f"--frame-count {frame_count}",
        f"--device {_quoted(device)}",
        f"--output-dirname {_quoted(output_dirname)}",
        f"--output-root {_quoted(str(output_root))}",
        f"--image-root {_quoted(str(image_root))}",
        f"--raw-root {_quoted(str(raw_root.resolve()))}",
        f"--processing-root {_quoted(str(processing_root.resolve()))}",
    ]

    if mask_root is not None:
        args.append(f"--mask-root {_quoted(str(mask_root.resolve()))}")

    return " ".join(args)


def _build_raw_moge_cli_args(
    *,
    seq_name: str,
    image_root: Path,
    output_root: Path,
    model_name: str,
    device: str,
    frame_step: int | None,
    max_frames: int | None,
    overwrite: bool,
    raw_root: Path,
) -> str:
    args: list[str] = [
        f"--seq-name {_quoted(seq_name)}",
        f"--image-root {_quoted(str(image_root.resolve()))}",
        f"--output-root {_quoted(str(output_root.resolve()))}",
        f"--raw-root {_quoted(str(raw_root.resolve()))}",
        f"--model-name {_quoted(model_name)}",
        f"--device {_quoted(device)}",
    ]
    if frame_step is not None:
        args.append(f"--frame-step {frame_step}")
    if max_frames is not None:
        args.append(f"--max-frames {max_frames}")
    if overwrite:
        args.append("--overwrite")
    return " ".join(args)


def _build_aligned_moge_cli_args(
    *,
    seq_name: str,
    raw_moge_root: Path,
    dust3r_root: Path,
    mask_root: Path | None,
    image_root: Path,
    output_root: Path,
    raw_root: Path,
    min_valid_pixels: int,
    store_pointcloud: bool,
    overwrite: bool,
) -> str:
    args: list[str] = [
        f"--seq-name {_quoted(seq_name)}",
        f"--raw-moge-root {_quoted(str(raw_moge_root.resolve()))}",
        f"--dust3r-root {_quoted(str(dust3r_root.resolve()))}",
        f"--image-root {_quoted(str(image_root.resolve()))}",
        f"--output-root {_quoted(str(output_root.resolve()))}",
        f"--raw-root {_quoted(str(raw_root.resolve()))}",
        f"--min-valid-pixels {min_valid_pixels}",
    ]
    if mask_root is not None:
        args.append(f"--mask-root {_quoted(str(mask_root.resolve()))}")
    if not store_pointcloud:
        args.append("--store-pointcloud False")
    if overwrite:
        args.append("--overwrite")
    return " ".join(args)


def main(
    img_dirs: list[str],
    gpus: list[int],
    img_name: str = "images",
    mask_name: str = "sam_v2_dyn_mask",
    metric_depth_name: str = "unidepth_disp",
    intrins_name: str = "unidepth_intrins",
    run_metric_depth: bool = False,
    mono_depth_model: str = "depth-anything",
    run_mono_depth: bool = False,
    slam_name: str = "droid_recon",
    track_model: str = "bootstapir",
    run_tracks: bool = True,
    tapir_torch: bool = True,
    run_dust3r: bool = True,
    dust3r_name: str = "dust3r",
    dust3r_model: str = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
    dust3r_scene_graph: str = "dust3r_scene_graph",
    dust3r_window_size: int = 3,
    dust3r_optimizer_iters: int = 700,
    dust3r_learning_rate: float = 1e-2,
    dust3r_min_confidence: float = 0.0,
    dust3r_optimize: bool = True,
    dust3r_batch_size: int = 1,
    dust3r_schedule: str = "cosine",
    dust3r_frame_step: int = 3,
    dust3r_frame_count: int = 300,
    dust3r_image_size: int = 512,
    dust3r_device: str = "cuda",
    dust3r_output_dirname: str = "dust_true_depth_conf3",
    dust3r_seq_names: list[str] | None = None,
    dust3r_name_index: int = 1,
    dust3r_raw_root: str | None = None,
    dust3r_mask_root: str | None = None,
    dust3r_processing_root: str | None = None,
    run_automask: bool = True,
    automask_name: str = "sam_v2_dyn_mask",
    automask_prompt: str = "person",
    automask_overwrite: bool = False,
    run_features: bool = True,
    feature_name: str = "dinov2_features",
    feature_model: str = "dinov2_vits14_reg",
    feature_height: int = 288,
    feature_width: int = 512,
    feature_dim: int = 32,
    feature_samples: int = 8192,
    feature_frame_step: int = 3,
    feature_max_frames: int | None = 300,
    feature_save_viz: bool = True,
    feature_viz_dirname: str = "pca_viz",
    feature_save_pca_model: bool = True,
    feature_num_crops_l0: int = 4,
    feature_crop_layers: int = 1,
    feature_crop_overlap_ratio: float = 512.0 / 1500.0,
    run_moge: bool = True,
    moge_name: str = "moge",
    moge_model: str = "Ruicheng/moge-vitl",
    moge_target_height: int | None = None,
    moge_target_width: int | None = None,
    moge_store_mask: bool = True,
    moge_store_pointcloud: bool = True,
    moge_store_intrinsics: bool = True,
    moge_overwrite: bool = True,
    moge_max_frames: int | None = 300,
    moge_frame_step: int = 3,
    run_raw_moge_depth: bool = True,
    raw_moge_name: str = "raw_moge_depth",
    raw_moge_frame_step: int | None = None,
    raw_moge_max_frames: int | None = None,
    raw_moge_overwrite: bool = True,
    run_aligned_moge_depth: bool = True,
    aligned_moge_name: str = "aligned_moge_depth",
    aligned_moge_min_valid_pixels: int = 500,
    aligned_moge_store_pointcloud: bool = True,
    aligned_moge_overwrite: bool = True,
) -> None:
    if not img_dirs:
        raise ValueError("img_dirs cannot be empty")

    expanded_dirs: list[str] = []
    for raw_dir in img_dirs:
        raw_dir = raw_dir.rstrip("/")
        if any(ch in raw_dir for ch in "*?[]"):
            matches = sorted(glob.glob(raw_dir))
            if not matches:
                raise FileNotFoundError(f"No directories matched pattern '{raw_dir}'")
            expanded_dirs.extend(str(Path(match).resolve()) for match in matches)
        else:
            path = Path(raw_dir).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Image directory '{raw_dir}' does not exist")
            expanded_dirs.append(str(path))

    if not expanded_dirs:
        raise ValueError("Resolved img_dirs cannot be empty")

    img_dirs = expanded_dirs

    sample_dir = Path(img_dirs[0])
    if img_name not in sample_dir.parts:
        raise ValueError(f"Expecting '{img_name}' in {sample_dir}")

    mono_depth_name = mono_depth_model.replace("-", "_")

    preproc_root = Path(__file__).resolve().parent
    repo_root = preproc_root.parent.resolve()
    parent_root = repo_root.parent.resolve()

    if dust3r_raw_root is None:
        raw_candidate = (repo_root / "_raw_data").resolve()
        raw_root = raw_candidate if raw_candidate.exists() else (parent_root / "raw_data").resolve()
    else:
        raw_root = Path(dust3r_raw_root).expanduser().resolve()

    if dust3r_processing_root is None:
        processing_candidate = (repo_root / "Processing").resolve()
        processing_root = processing_candidate if processing_candidate.exists() else repo_root
    else:
        processing_root = Path(dust3r_processing_root).expanduser().resolve()

    mask_root_override = (
        Path(dust3r_mask_root).expanduser().resolve() if dust3r_mask_root is not None else None
    )

    seq_names: list[str | None] = [None] * len(img_dirs)
    if run_dust3r:
        if (
            dust3r_scene_graph != "swin"
            or dust3r_window_size != 5
            or dust3r_min_confidence != 0.0
            or not dust3r_optimize
        ):
            print(
                "[DUSt3R] Warning: scene graph/window/min_confidence/optimize arguments "
                "are deprecated and ignored by the current DUSt3R integration."
            )

        if dust3r_seq_names is not None:
            if len(dust3r_seq_names) != len(img_dirs):
                raise ValueError("Length of dust3r_seq_names must match img_dirs")
            seq_names = list(dust3r_seq_names)
        else:
            if not raw_root.exists():
                raise FileNotFoundError(
                    f"DUSt3R raw_root '{raw_root}' does not exist. Provide --dust3r-raw-root or --dust3r-seq-names."
                )
            short_to_seq = _build_short_to_seq_map(raw_root, dust3r_name_index)
            for idx, img_dir in enumerate(img_dirs):
                short_name = _short_name_from_img_dir(Path(img_dir))
                candidates = short_to_seq.get(short_name)
                if not candidates:
                    raise ValueError(
                        f"Could not infer DUSt3R sequence for '{img_dir}'. Provide --dust3r-seq-names "
                        "or ensure raw data is available under the expected root."
                    )
                if len(candidates) > 1:
                    options = ", ".join(sorted(candidates))
                    raise ValueError(
                        f"Ambiguous DUSt3R sequence for '{img_dir}' (short name '{short_name}'). "
                        f"Candidates: {options}. Provide --dust3r-seq-names to disambiguate."
                    )
                seq_names[idx] = candidates[0]

    seq_names = [name if name is None else str(name) for name in seq_names]

    if not gpus:
        raise ValueError("gpus cannot be empty")
    if len(gpus) > 1:
        print(
            f"[Features] Multiple GPU ids provided ({gpus}); using the first one ({gpus[0]}) for all feature jobs."
        )
    gpu = gpus[0]

    submitted_dust3r: set[str] = set()
    submitted_raw_moge: set[str] = set()
    submitted_aligned_moge: set[str] = set()
    for idx, img_dir in enumerate(img_dirs):
        img_dir = img_dir.rstrip("/")
        mask_dir = img_dir.replace(img_name, mask_name)
        metric_depth_dir = img_dir.replace(img_name, metric_depth_name)
        intrins_path = img_dir.replace(img_name, intrins_name)
        mono_depth_dir = img_dir.replace(img_name, mono_depth_name)
        aligned_depth_dir = img_dir.replace(img_name, f"aligned_{mono_depth_name}")
        slam_dir = img_dir.replace(img_name, slam_name)
        track_dir = img_dir.replace(img_name, track_model)
        dust3r_dir = img_dir.replace(img_name, dust3r_name)
        automask_dir = img_dir.replace(img_name, automask_name)
        feature_dir = img_dir.replace(img_name, feature_name)
        moge_dir = img_dir.replace(img_name, moge_name)
        raw_moge_dir = img_dir.replace(img_name, raw_moge_name)
        aligned_moge_dir = img_dir.replace(img_name, aligned_moge_name)

        img_path = Path(img_dir).resolve()
        image_root_path = _find_named_ancestor(img_path, img_name)
        if image_root_path is None:
            raise ValueError(f"Could not locate '{img_name}' ancestor for {img_dir}")
        mask_root_path = (
            mask_root_override
            if mask_root_override is not None
            else _find_named_ancestor(Path(mask_dir).resolve(), mask_name)
        )

        seq_name = seq_names[idx]
        dust3r_cli_args = None
        if run_dust3r and seq_name is not None and seq_name not in submitted_dust3r:
            dust3r_cli_args = _build_dust3r_cli_args(
                seq_name=seq_name,
                img_dir=img_dir,
                mask_dir=mask_dir,
                dust3r_dir=dust3r_dir,
                img_name=img_name,
                mask_name=mask_name,
                model_name=dust3r_model,
                batch_size=dust3r_batch_size,
                schedule=dust3r_schedule,
                niter=dust3r_optimizer_iters,
                learning_rate=dust3r_learning_rate,
                image_size=dust3r_image_size,
                frame_step=dust3r_frame_step,
                frame_count=dust3r_frame_count,
                device=dust3r_device,
                output_dirname=dust3r_output_dirname,
                raw_root=raw_root,
                mask_root_override=mask_root_override,
                processing_root=processing_root,
            )
            submitted_dust3r.add(seq_name)

        raw_moge_cli_args = None
        if run_raw_moge_depth and seq_name is not None and seq_name not in submitted_raw_moge:
            eff_frame_step = raw_moge_frame_step if raw_moge_frame_step is not None else moge_frame_step
            eff_max_frames = raw_moge_max_frames if raw_moge_max_frames is not None else moge_max_frames
            raw_moge_cli_args = _build_raw_moge_cli_args(
                seq_name=seq_name,
                image_root=image_root_path,
                output_root=Path(raw_moge_dir).resolve().parent,
                model_name=moge_model,
                device="cuda",
                frame_step=eff_frame_step,
                max_frames=eff_max_frames,
                overwrite=raw_moge_overwrite,
                raw_root=raw_root,
            )
            submitted_raw_moge.add(seq_name)

        aligned_moge_cli_args = None
        if run_aligned_moge_depth and seq_name is not None and seq_name not in submitted_aligned_moge:
            aligned_moge_cli_args = _build_aligned_moge_cli_args(
                seq_name=seq_name,
                raw_moge_root=Path(raw_moge_dir).resolve().parent,
                dust3r_root=Path(dust3r_dir).resolve().parent,
                mask_root=mask_root_override if mask_root_override is not None else mask_root_path,
                image_root=image_root_path,
                output_root=Path(aligned_moge_dir).resolve().parent,
                raw_root=raw_root,
                min_valid_pixels=aligned_moge_min_valid_pixels,
                store_pointcloud=aligned_moge_store_pointcloud,
                overwrite=aligned_moge_overwrite,
            )
            submitted_aligned_moge.add(seq_name)

        process_sequence(
            gpu,
            img_dir,
            mask_dir,
            metric_depth_dir,
            intrins_path,
            mono_depth_dir,
            aligned_depth_dir,
            slam_dir,
            track_dir,
            run_metric_depth,
            run_mono_depth,
            mono_depth_model,
            run_tracks,
            track_model,
            tapir_torch,
            run_dust3r and dust3r_cli_args is not None,
            dust3r_dir,
            dust3r_model,
            dust3r_scene_graph,
            dust3r_window_size,
            dust3r_optimizer_iters,
            dust3r_learning_rate,
            dust3r_min_confidence,
            dust3r_optimize,
            run_automask,
            automask_dir,
            automask_prompt,
            automask_overwrite,
            run_features,
            feature_dir,
            feature_model,
            feature_height,
            feature_width,
            feature_dim,
            feature_samples,
            feature_frame_step,
            feature_max_frames,
            feature_save_viz,
            feature_viz_dirname,
            feature_save_pca_model,
            feature_num_crops_l0,
            feature_crop_layers,
            feature_crop_overlap_ratio,
            run_moge,
            moge_dir,
            moge_model,
            moge_target_height,
            moge_target_width,
            moge_store_mask,
            moge_store_pointcloud,
            moge_store_intrinsics,
            moge_overwrite,
            moge_max_frames,
            moge_frame_step,
            dust3r_cli_args,
            raw_moge_cli_args,
            aligned_moge_cli_args,
        )


def process_sequence(
    gpu: int,
    img_dir: str,
    mask_dir: str,
    metric_depth_dir: str,
    intrins_path: str,
    mono_depth_dir: str,
    aligned_depth_dir: str,
    slam_dir: str,
    track_dir: str,
    run_metric_depth: bool,
    run_mono_depth: bool,
    mono_depth_model: str,
    run_tracks: bool,
    track_model: str,
    tapir_torch: bool,
    run_dust3r: bool,
    dust3r_dir: str,
    dust3r_model: str,
    dust3r_scene_graph: str,
    dust3r_window_size: int,
    dust3r_optimizer_iters: int,
    dust3r_learning_rate: float,
    dust3r_min_confidence: float,
    dust3r_optimize: bool,
    run_automask: bool,
    automask_dir: str,
    automask_prompt: str,
    automask_overwrite: bool,
    run_features: bool,
    feature_dir: str,
    feature_model: str,
    feature_height: int,
    feature_width: int,
    feature_dim: int,
    feature_samples: int,
    feature_frame_step: int,
    feature_max_frames: int | None,
    feature_save_viz: bool,
    feature_viz_dirname: str,
    feature_save_pca_model: bool,
    feature_num_crops_l0: int,
    feature_crop_layers: int,
    feature_crop_overlap_ratio: float,
    run_moge: bool,
    moge_dir: str,
    moge_model: str,
    moge_target_height: int | None,
    moge_target_width: int | None,
    moge_store_mask: bool,
    moge_store_pointcloud: bool,
    moge_store_intrinsics: bool,
    moge_overwrite: bool,
    moge_max_frames: int | None,
    moge_frame_step: int,
    dust3r_cli_args: str | None,
    raw_moge_cli_args: str | None,
    aligned_moge_cli_args: str | None,
) -> None:
    dev_arg = f"CUDA_VISIBLE_DEVICES={gpu}"


    if run_features:
        feature_cmd = (
            f"{dev_arg} python compute_dinofeatures.py --img_dir {_quoted(img_dir)} "
            f"--out_dir {_quoted(feature_dir)} --device cuda "
            f"--model_name {_quoted(feature_model)} --output_height {feature_height} "
            f"--output_width {feature_width} --output_dim {feature_dim} "
            f"--num_crops_l0 {feature_num_crops_l0} --crop_n_layers {feature_crop_layers} "
            f"--crop_overlap_ratio {feature_crop_overlap_ratio} "
            f"--sample_per_image {feature_samples} --frame_step {feature_frame_step} "
            f"--mask_dir {_quoted(mask_dir)}"
        ) 
        if feature_max_frames is not None:
            feature_cmd += f" --max_frames {feature_max_frames}"
        feature_cmd += f" --viz_dirname {_quoted(feature_viz_dirname)}"
        if not feature_save_viz:
            feature_cmd += " --no-save_viz"
        if not feature_save_pca_model:
            feature_cmd += " --no-save_pca_model"
        _run_command(feature_cmd)


if __name__ == "__main__":
    tyro.cli(main)
