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

## Phase 7: Tracking Fix — RAFT+mask (2026-03-28 ~ 2026-03-29)

### 7.1 TAPNet Failure Root Cause Analysis
- TAPNet (BootsTAPIR) marked 80-89% FG tracks as occluded at F30
- Root cause: mouse moves ~6px/frame at 1080p, 30fps → violates TAPNet's slow-motion assumption
- "Visible" tracks after TAPNet = all background (slow-moving)
- Confirmed via Option A post-filter: 0% visible FG tracks after SAM2 mask filter (correct, but no signal)

### 7.2 Option B: RAFT+mask (COMPLETE, WORKING)
- Strategy: RAFT optical flow on original RGB (no masking to avoid distribution shift)
- Per-frame SAM2 mask used as validation (not input)
- Query points seeded from FG mask at query frame
- Bug #1 fixed: `dyn_mask.astype(bool)` → `raw > 0` (float32 -1.0 → True is wrong)
- Bug #2 fixed: RAFT uint8 input + `Raft_Small_Weights.DEFAULT.transforms()` required
- Results: cam00=45%, cam01=18%, cam02=0%(normal), cam03=15% at F30
- Script: `mouse_m5t2/scripts/generate_raft_tracks.py`

### 7.3 Multi-Query RAFT (COMPLETE)
- query_frames=[0,15,30,45,59], 512 pts/query → 300 files per camera
- cam01 F30 coverage: single-query 18% → multi-query ~30%
- `casual_dataset.py` confirmed to support multiple `query_idcs`
- All 4 cameras: ~1200-1500 files total

### 7.4 tapir/ Symlink Fix (COMPLETE)
- `train_m5t2.py:142` hardcodes check for "tapir" directory
- Fix: `mv tapir tapir_tapnet_backup && ln -sfn tapir_raft tapir`
- Backup: `tapir_tapnet_backup/` (original TAPNet tracks)

### 7.5 Visualizations Generated
- `viz/10_tapnet_vs_raft_cam01.png` — TAPNet(0%) vs RAFT(18%) at F30
- `viz/11_raft_all_cameras.png` — 4-camera RAFT results
- `viz/12_raft_all_cameras_legend.png` — with color legend
- `viz/13_raft_trajectory_analysis_cam01.png` — trajectories + visibility curve
- `viz/15_raft_4cam_video.mp4` — 60-frame tracking video
- `viz/16_multi_query_comparison_cam01.png` — single-Q vs multi-Q comparison

### 7.6 Documentation Added
- `docs/core_architecture.md` — full pipeline + RAFT fix strategy + training integration
- `docs/theory/scene_flow_and_tracking.md` — theory: 4D-GS, scene flow, RAFT vs TAPNet
- `docs/training_guide.md` — pre-training safeguards, monitoring metrics, improvement roadmap
- `docs/experiments/mf_001_notes.md` — updated with mf_001_raft pending config

## Phase 8: Architecture Audit & Doc Fix (2026-03-29)

### 8.1 Critical Architecture Correction: MLP → SE(3) Motion Bases (DONE)
- **Bug**: all docs incorrectly described "Deformation MLP" — MonoFusion uses SE(3) bases
- Root cause: reading 4D-GS survey papers instead of actual `flow3d/scene_model.py`
- Fix: updated `core_architecture.md`, `scene_flow_and_tracking.md` with correct architecture
- **Implication**: scene flow = `compute_poses_fg(t+1) - compute_poses_fg(t)` (no MLP forward)

### 8.2 Critical Bug Fix: expected_dist=0.0 → CONFIDENT_DIST=-2.0 (DONE)
- `parse_tapir_track_info`: `confidence = 1 - sigmoid(0.0) = 0.5` → valid_visible=0.44 < 0.5 → ALL visible RAFT tracks discarded
- Fix: added `CONFIDENT_DIST = -2.0` constant in `generate_raft_tracks.py`
- **Action required**: Regenerate RAFT tracks on gpu03 (existing tapir_raft/ is broken)

### 8.3 Validation & Viz Scripts (DONE)
- `validate_training_inputs.py` + `validate_checks.py` — 5 pre-training silent-bug checks
- `viz_scene_flow.py` — post-training scene flow visualization (matplotlib, 3 figures)
- Both split per 400L coding rule (validate: 68L+329L, viz: 387L)

### 8.4 New Documentation (DONE)
- `docs/theory/monofusion_architecture.md` — SE(3) motion bases, K selection, transform math
- All docs: backlinks added (`↑ MOC / ↔ Related`)
- `docs/README.md`: monofusion_architecture.md registered

## Phase 9: Execution (2026-03-29, IN PROGRESS)

### 9.1 Code Sync (DONE)
- rsync Mac → GPU03: scripts/ (16 files), train_m5t2.py, docs/ (16 files)
- Verified existing tapir_raft/ tracks: expected_dist=0.0 (confirmed broken)
- Deleted broken tapir_raft/ on GPU03

### 9.2 RAFT Track Regeneration (IN PROGRESS)
- Command: `generate_raft_tracks.py --query_frames 0 15 30 45 59 --n_points 512`
- GPU: Blackwell RTX PRO 6000 (GPU 0, 98GB)
- Fix: CONFIDENT_DIST=-2.0 (expected_dist column)
- tapir/ symlink → tapir_raft/ (already configured)

### 9.3 Pre-Training Checklist
- [x] Code sync to GPU03
- [x] Delete broken tracks
- [ ] RAFT track regeneration (running)
- [ ] Run validate_training_inputs.py — verify 5 checks pass
- [ ] Verify expected_dist=-2.0 in new tracks

### 9.4 First Training Run (PENDING)
- Experiment: `mf_001_raft`
- Plan: 1cam+20frame 축소 실험 먼저 (fast debug)
- Then: full 4cam×60frame, 30k iterations
- GPU: A6000 (CUDA_VISIBLE_DEVICES=4 or Blackwell GPU 0)

### 9.5 Temporal Consistency MoA Findings
- 3-model consensus: CoTracker > RAFT for long-range tracking
- Current RAFT: PoC 수준 OK, 고품질 필요시 CoTracker 전환 필수
- See `docs/audit_pretraining.md` for full MoA analysis

---

*Progress Log | MonoFusion M5t2 PoC | 2026-03-29*
