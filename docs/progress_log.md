# MonoFusion M5t2 PoC — Progress Log

## Phase 0: Analysis & Planning (COMPLETE)

### 0.1 Paper & Repo Analysis
- MonoFusion: 4D-GS for dynamic sparse-view scenes
- Architecture: canonical-frame 3D Gaussians + SE(3) motion bases
- Preprocessing: DUSt3R + MoGe + SAM2 + DINOv2 + TAPNet (11 submodules)
- Training core: gsplat (pip) + PyTorch — no custom CUDA compilation needed

### 0.2 MoA Deliberation (2 loops × 3 models)
- Loop 1: Feasibility assessment, risk matrix, execution plan
- Loop 2: Refined strategy — synthetic vs real preprocessing, GPU choice, minimal inputs
- Consensus: A6000 + cu118, 60 frames × 4 views, real preprocessing pipeline

### 0.3 Code Analysis (3 parallel agents)
- Camera views: 4 hardcoded but architecture-flexible
- Resolution: dynamic from image (512×512 OK)
- Scene scale: data-adaptive, no human-scale assumption
- Training inputs: RGB + masks + cameras + tracks + BG points (depth/features optional)

## Phase 1: Environment Setup (COMPLETE)

### 1.1 Conda Environment
- `monofusion` env: Python 3.10, PyTorch 2.1.0+cu118
- CUDA toolkit 11.8 (conda, env-isolated)
- gsplat 1.5.3 (pip, rasterization verified on A6000 sm_86)
- numpy 1.26.4, transformers 4.38.2

### 1.2 MonoFusion Clone
- Local: `/Users/joon/dev/MonoFusion/` (with all submodules)
- gpu03: `/home/joon/dev/MonoFusion/` (with all submodules)

### 1.3 DUSt3R Import
- Imported via PYTHONPATH (no setup.py in MonoFusion fork)
- Slow RoPE2D fallback (no CUDA compilation needed for inference)

## Phase 2: Data Conversion (COMPLETE)

### 2.1 Camera Selection
- 6 cameras available (cam_000 to cam_005)
- Selected 4 by angular spread: cam 0, 1, 2, 5
- Reprojection sanity check: 4/4 OK

### 2.2 Format Conversion
- Script: `mouse_m5t2/scripts/convert_m5t2.py`
- 60 frames (stride=10 from 3600), 4 cameras
- RGBA → RGB + alpha mask (FG=1, BG=-1, border=0)
- opencv_cameras.json → Dy_train_meta.json (per-camera)
- Output: `/node_data/joon/data/monofusion/m5t2_poc/`

## Phase 3: Preprocessing (REVERTED — see Phase 6 Audit)

> **⚠️ AUDIT FINDING**: RAFT tracks and Depth-Anything-V2 were judged as fundamentally
> wrong substitutions by 3-model audit (see `mouse_m5t2/envs/feedback_audit.md`).
> All preprocessing outputs from this phase must be regenerated using the original
> pipeline tools (TAPNet, DUSt3R+MoGe) in the new multi-env setup.

### 3.1 DINOv2 Feature Extraction
- Model: dinov2_vits14 (torch.hub)
- Output: 37×37×384 feature maps (fp16)
- Time: ~15s for 240 images (60 frames × 4 cams)
- Script: `mouse_m5t2/scripts/run_dinov2.py`

### 3.2 Depth Estimation
- Model: Depth-Anything-V2-Small (transformers pipeline)
- Output: 512×512 relative depth maps (float32)
- Note: relative depth, not metric — may need alignment
- Time: ~15s for 240 images
- Script: `mouse_m5t2/scripts/run_moge_depth.py --use_fallback`

### 3.3 Point Tracking
- Method: RAFT optical flow based tracking (fallback)
- 512 foreground query points per camera
- Visibility: cam00 51%, cam01 57%, cam02 46%, cam03 90%
- Note: RAFT flow tracking has drift over time — CoTracker/TAPNet would improve
- Time: ~4s per camera
- Script: `mouse_m5t2/scripts/run_tapnet_tracks.py --use_flow_fallback`

### 3.4 Masks
- Source: RGBA alpha channel from M5t2 dataset
- Processed: FG (alpha > 128) = 1, BG = -1, erosion border = 0
- FG coverage: 2.0-2.8% of image (mouse is small)

## Phase 4: Visualization & Verification (COMPLETE)

### 4.1 Diagnostic Visualizations (7 types)
All saved to `/node_data/joon/data/monofusion/m5t2_poc/viz/`:

1. **Multi-view RGB**: 4 cameras × 4 frames — different angles confirmed
2. **Mask Overlay**: FG (green) accurately covers mouse silhouette
3. **Depth Maps**: Relative depth shows scene structure, mouse region distinct
4. **DINOv2 PCA**: Semantic regions consistent across cameras
5. **Track Trajectories**: Query → tracked → trajectory visualization per camera
6. **Camera Poses 3D**: 4 cameras distributed around scene center
7. **Reprojection Check**: Scene center projects correctly into all 4 cameras

### 4.2 Issues Identified
- Depth: relative (not metric) — may need DUSt3R alignment for BG init
- Tracks: RAFT drift visible in cam02 (45% visibility) — acceptable for PoC

## Phase 5: Training Connection (IN PROGRESS)

### 5.1 casual_dataset.py Patches (DONE)
- Patch 1: Camera index extraction — regex `cam(\d+)` instead of `seq_name[-1]`
- Patch 2: Time loop — `len(md["k"])` instead of hardcoded `range(0, 300, 3)`
- Patch 3: Meta path — per-camera `Dy_train_meta_camXX.json` support

### 5.2 Data Loading Debug (DONE)
- tyro, loguru, mediapy etc. installed
- Camera JSON path: needed root_dir prefix (relative→absolute)
- glb_step: 3→1 for M5t2 (pre-sampled data)
- depth_type: "aligned_moge_depth"→"moge" (matches load_depth dispatcher)
- Track format: MonoFusion needs ALL query frames, not just frame 0
  - query_idcs = range(0, 60, 6) → 10 query frames
  - Regenerated tracks for all 10 query frames per camera

### 5.3 Training Wrapper (IN PROGRESS)
- `mouse_m5t2/train_m5t2.py` — creates CasualDataset per camera, wires to dance_glb.py
- Need to test data loading end-to-end before training

### 5.3 First Training Run (TODO)
- Target: 60 frames × 4 views, 5K FG + 10K BG Gaussians, 10 motion bases
- GPU: A6000 (GPU 4), ~48GB VRAM
- Expected: ~10-15 min training time

## Phase 6: Audit & Corrective Actions (2026-03-27)

### 6.1 3-Model Audit (MoA --audit)
- 6 substitutions audited: 4 Critical, 2 Major
- **Critical**: TAPNet→RAFT, DUSt3R→DA-V2, dummy tracks, single env
- **Major**: DINOv2 upsample, monkey-patches
- Verdict: "Not a MonoFusion PoC — a different broken pipeline"

### 6.2 Corrective Actions (DONE)
- All monkey-patches reverted to original (`git checkout`)
- 3-env architecture designed: `monofusion_jax` / `monofusion_pytorch` / `monofusion_raft`
- environment.yml × 3 + SETUP_GUIDE.md written
- `monofusion_jax` env creation started (background)

### 6.3 Next Session TODO
1. Verify `monofusion_jax` env + TAPNet GPU
2. Rename `monofusion` → `monofusion_pytorch` (or create fresh)
3. Run original pipeline: DUSt3R → MoGe → BootsTAPIR → DinoFeature
4. Write proper data adapter (not monkey-patches)
5. Training with correct preprocessing outputs

---

*Progress Log | MonoFusion M5t2 PoC | 2026-03-27*
