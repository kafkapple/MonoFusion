# MonoFusion
Sparse-view 4D reconstruction pipeline for monocular captures. This repo wraps preprocessing (depth, masks, camera tracks), training, and visualization into five short actions.

## Quick Start
1. **Set up the environment**
   ```bash
   git clone --recursive https://github.com/MonoFusion/MonoFusion.git
   cd MonoFusion
   conda create -n monofusion python=3.10
   conda activate monofusion
   pip install -r requirements.txt
   pip install git+https://github.com/nerfstudio-project/gsplat.git
   ```
   *Preprocessing extras:* `cd preproc && ./setup_dependencies.sh` installs Track-Anything, TapNet, and downloads all checkpoints listed in `preproc/requirements_extra.txt`.

2. **Stage raw data** – place every sequence under `./_raw_data/<SEQ_NAME>/`. Each folder should contain your source media, e.g. `images/frame_*.png` or `videos/<SEQ_NAME>.mp4`, plus optional masks (`sam_v2_dyn_mask`) if you already have them. The code also accepts `./raw_data/<SEQ_NAME>` if you prefer that name.

3. **Process data into priors**
   ```bash
   cd preproc
   python process_custom.py --img-dirs ../_raw_data/<SEQ_NAME>/images --gpus 0 1
   ```
   This single command runs depth, masks, TAPIR tracks, and DUSt3R alignment. When it finishes you should see `../data/<SEQ_NAME>/` with subfolders such as `aligned_depth_anything`, `dust3r_scene_graph`, `droid_recon`, and `bootstapir`.

4. **Train on a scene**
   ```bash
   # Edit opt.sh line 12 so SEQ_NAME matches your folder (e.g. "_indiana_piano_14_4").
   bash opt.sh <experiment_prefix>
   ```
   `opt.sh` wraps `dance_glb.py`, appends a timestamp to the prefix, and writes results to `./results_${SEQ_NAME}/<experiment_prefix>_<timestamp>/`. You can customize training indices or optimizer knobs by editing the script or calling `python dance_glb.py --seq_name <SEQ_NAME> --exp <NAME> [extra Tyro args]` directly.

5. **Visualize**
   ```bash
   bash vis.sh ./results_${SEQ_NAME}/<RUN_NAME> 7007
   ```
   Pass the exact work directory from step 4 and any open TCP port; `vis.sh` launches `run_rendering.py` for interactive inspection.

## Directory Expectations
- `_raw_data/<SEQ_NAME>` – raw captures plus metadata (`trajectory/Dy_train_meta.json` if available). Used for preprocessing and to fetch camera priors.
- `data/<SEQ_NAME>` – auto-generated priors consumed by `flow3d` during training.
- `results_${SEQ_NAME}/<RUN>` – experiment logs, checkpoints (`checkpoints/last.ckpt`), and video dumps.
- `preproc/` – third-party tools and scripts (`mask_app.py`, `process_custom.py`, DUSt3R, TAPIR, etc.). Run everything here inside the conda env you created above.

## Tips
- Make sure your GPU drivers match the CUDA wheels you select in `requirements.txt` (defaults target CUDA 11.2/12.x).
- `process_custom.py` accepts multiple `--img-dirs` in one call; mix and match GPUs via `--gpus` for throughput.
- If `_infer_raw_scene_dir` cannot locate your sequence during training, double-check the leading underscore convention (e.g. `_scene_01` instead of `scene_01`).
- `wandb` logging is enabled by default; export `WANDB_API_KEY` before launching training if you want the run to sync online.
