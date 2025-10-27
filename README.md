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
### 1. Raw data via [ExoRecon](preproc/ExoRecon/README.md) (`./raw_data/SEQ_NAME`)
- `cd preproc/ExoRecon` and follow `README.md` there:
  ```bash
  conda env create -f egorecon.yml
  conda activate egorecon
  python -m pip install -e projectaria_tools_pkg
  ./push_all_data.sh  # downloads + restructures Ego-Exo4D takes
  ```
- Each take should end up as `MonoFusion/raw_data/<SEQ_NAME>/` containing `aria01.vrs`, `frame_aligned_videos/`, `trajectory/Dy_train_meta.json`, and `timestep.txt`.

### 2. Get Priors (`./data/SEQ_NAME`)
```bash
cd preproc
python process_custom.py \
  --img-dirs ../raw_data/<SEQ_NAME>/images \
  --gpus 0 1
```
- Generates depth, masks, TAPIR tracks, and DUSt3R alignment into `../data/<SEQ_NAME>/`.

### 3. Train (`bash opt.sh`)
```bash
# edit opt.sh so SEQ_NAME matches _<SEQ_NAME> used during preprocessing
bash opt.sh <experiment_prefix>
```
- The script appends a timestamp, calls `dance_glb.py`, logs to `./results_<SEQ_NAME>/<experiment_prefix>_<timestamp>/`, and saves checkpoints under `checkpoints/` inside that folder.
- Advanced runs can invoke `python dance_glb.py --seq_name <SEQ_NAME> --exp <NAME> [Tyro args]` directly.

### 4. Visualize
```bash
bash vis.sh ./results_<SEQ_NAME>/<RUN_NAME> 7007
```
- `WORK_DIR` is the exact path produced in step 4.
- Pick any open TCP port; the script launches `run_rendering.py` for inspection.


## Citation
If you find our data, code processing, or project useful, please kindly consider citing our work: 
```
@InProceedings{Wang_2025_ICCV,
    author    = {Wang, Zihan and Tan, Jeff and Khurana, Tarasha and Peri, Neehar and Ramanan, Deva},
    title     = {MonoFusion: Sparse-View 4D Reconstruction via Monocular Fusion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {8252-8263}
}
```
