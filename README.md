# (ICCV 25)MonoFusion: Sparse-View 4D Reconstruction via Monocular Fusion
**[Project Page](https://ImNotPrepared.github.io/research/25_DSR/index.html) | [Arxiv](https://arxiv.org/abs/2507.23782) | [Data](https://drive.google.com/drive/folders/18H8OOOZLv7OmOen8pGbSLWwu8AvZAZro?usp=sharing)**

[Zihan Wang](https://imnotprepared.github.io/), [Jeff Tan](https://jefftan969.github.io/), [Tarasha Khurana](https://www.cs.cmu.edu/~tkhurana/)\*, [Neehar Peri](https://www.neeharperi.com)\*, [Deva Ramanan](https://www.cs.cmu.edu/~deva/)

Carnegie Mellon University

\* Equal Contribution

## Installation
```bash
git clone --recursive https://github.com/MonoFusion/MonoFusion.git
cd MonoFusion
conda create -n monofusion python=3.10
conda activate monofusion
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git
# extra deps for preprocessing
cd preproc && ./setup_dependencies.sh && cd -
```

## Usage
### 1. Environment
- Keep everything inside the `monofusion` conda env created above.
- GPU drivers must match the CUDA wheels declared in `requirements.txt` (defaults to CUDA 11.x).

### 2. Raw data via ExoRecon (`./raw_data/SEQ_NAME`)
- `cd preproc/ExoRecon` and follow `README.md` there:
  ```bash
  conda env create -f egorecon.yml
  conda activate egorecon
  python -m pip install -e projectaria_tools_pkg
  ./push_all_data.sh  # downloads + restructures Ego-Exo4D takes
  ```
- Each take should end up as `MonoFusion/raw_data/<SEQ_NAME>/` containing `aria01.vrs`, `frame_aligned_videos/`, `trajectory/Dy_train_meta.json`, and `timestep.txt`.

### 3. Priors (`./data/SEQ_NAME`)
```bash
cd preproc
python process_custom.py \
  --img-dirs ../raw_data/<SEQ_NAME>/images \
  --gpus 0 1
```
- Generates depth, masks, TAPIR tracks, and DUSt3R alignment into `../data/<SEQ_NAME>/`.
- Re-run whenever raw inputs change; outputs are overwritten safely.

### 4. Train (`bash opt.sh`)
```bash
# edit opt.sh so SEQ_NAME matches _<SEQ_NAME> used during preprocessing
bash opt.sh <experiment_prefix>
```
- The script appends a timestamp, calls `dance_glb.py`, logs to `./results_<SEQ_NAME>/<experiment_prefix>_<timestamp>/`, and saves checkpoints under `checkpoints/` inside that folder.
- Advanced runs can invoke `python dance_glb.py --seq_name <SEQ_NAME> --exp <NAME> [Tyro args]` directly.

### 5. Visualize (`bash vis.sh <WORK_DIR> <PORT>`)
```bash
bash vis.sh ./results_<SEQ_NAME>/<RUN_NAME> 7007
```
- `WORK_DIR` is the exact path produced in step 4.
- Pick any open TCP port; the script launches `run_rendering.py` for inspection.

## Directory Expectations
- `raw_data/<SEQ_NAME>`: unprocessed Ego-Exo4D take plus `Dy_train_meta.json` from ExoRecon.
- `data/<SEQ_NAME>`: priors from `process_custom.py` (depth, masks, DUSt3R, TAPIR).
- `results_<SEQ_NAME>/<RUN>`: training logs, checkpoints, and rendered videos.
- `preproc/ExoRecon`: official EgoRecon pipeline; always ensure it is up to date before step 2.

## Todo List
| Task | Status | Due Date |
|------|--------|----------|
| Drop data and environ build guide | ✅ Done | - |
| Preprocessing scripts | ⏳ Todo | in a week |
| Drop Code | ⏳ Todo | between ICLR and ICCV |

## Citation
If you find our data, code processing, or project useful, please kindly consider citing our work: 
```
@misc{wang2025monofusionsparseview4dreconstruction,
      title={MonoFusion: Sparse-View 4D Reconstruction via Monocular Fusion}, 
      author={Zihan Wang and Jeff Tan and Tarasha Khurana and Neehar Peri and Deva Ramanan},
      year={2025},
      eprint={2507.23782},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.23782}, 
}
```
