#!/bin/bash
# V8 Single-Variable Isolation Experiments
# Runs E1→E2→E3 sequentially after V8a completes
# Usage: nohup bash mouse_m5t2/scripts/run_v8_isolation.sh > /node_data/joon/data/monofusion/markerless_v7/v8_runner.log 2>&1 &

set -euo pipefail

DATA_ROOT="/node_data/joon/data/monofusion/markerless_v7"
PYTHON="${HOME}/anaconda3/envs/monofusion/bin/python"
TRAIN_SCRIPT="mouse_m5t2/train_m5t2.py"

# Environment: conda bin in PATH (for ninja), GPU selection, compiler
export PATH="${HOME}/anaconda3/envs/monofusion/bin:${PATH}"
export CC=x86_64-conda-linux-gnu-gcc
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=4

# Common args (V8a baseline config, only one variable changes per experiment)
COMMON_ARGS="--data_root ${DATA_ROOT} \
    --num_epochs 300 \
    --w_feat 1.5 --w_mask 7.0 --w_depth_reg 0.0 \
    --feat_ramp_start_epoch 5 --feat_ramp_end_epoch 50 \
    --max_gaussians 100000 --stop_densify_pct 0.6 \
    --densify_xys_grad_threshold 0.0002 \
    --feat_dir_name dinov2_features_pca32_norm \
    --seed 42"

echo "============================================================"
echo "V8 Isolation Experiments — $(date)"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

# V8a: Reproducible baseline (BG frozen, seed=42)
echo ""
echo "============================================================"
echo "[$(date)] Starting V8a: Baseline (BG frozen)"
echo "============================================================"
mkdir -p ${DATA_ROOT}/results_v8a
${PYTHON} -u ${TRAIN_SCRIPT} ${COMMON_ARGS} \
    --num_fg 5000 --num_bg 10000 --num_motion_bases 10 \
    --bg_lr_config frozen \
    --output_dir ${DATA_ROOT}/results_v8a \
    --wandb_name v8a_baseline_seed42 \
    2>&1 | tee ${DATA_ROOT}/results_v8a/train.log
echo "[$(date)] V8a completed."

# E1: BG LR unfrozen (only change: --bg_lr_config gt)
echo ""
echo "============================================================"
echo "[$(date)] Starting E1: BG LR unfrozen"
echo "============================================================"
mkdir -p ${DATA_ROOT}/results_e1
${PYTHON} -u ${TRAIN_SCRIPT} ${COMMON_ARGS} \
    --num_fg 5000 --num_bg 10000 --num_motion_bases 10 \
    --bg_lr_config gt \
    --output_dir ${DATA_ROOT}/results_e1 \
    --wandb_name e1_bg_lr_unfrozen \
    2>&1 | tee ${DATA_ROOT}/results_e1/train.log
echo "[$(date)] E1 completed."

# E2: FG 18K (only change: --num_fg 18000)
echo ""
echo "============================================================"
echo "[$(date)] Starting E2: FG 18K"
echo "============================================================"
mkdir -p ${DATA_ROOT}/results_e2
${PYTHON} -u ${TRAIN_SCRIPT} ${COMMON_ARGS} \
    --num_fg 18000 --num_bg 10000 --num_motion_bases 10 \
    --bg_lr_config frozen \
    --output_dir ${DATA_ROOT}/results_e2 \
    --wandb_name e2_fg_18k \
    2>&1 | tee ${DATA_ROOT}/results_e2/train.log
echo "[$(date)] E2 completed."

# E3: Motion bases 28 (only change: --num_motion_bases 28)
echo ""
echo "============================================================"
echo "[$(date)] Starting E3: bases 28"
echo "============================================================"
mkdir -p ${DATA_ROOT}/results_e3
${PYTHON} -u ${TRAIN_SCRIPT} ${COMMON_ARGS} \
    --num_fg 5000 --num_bg 10000 --num_motion_bases 28 \
    --bg_lr_config frozen \
    --output_dir ${DATA_ROOT}/results_e3 \
    --wandb_name e3_bases_28 \
    2>&1 | tee ${DATA_ROOT}/results_e3/train.log
echo "[$(date)] E3 completed."

echo ""
echo "============================================================"
echo "[$(date)] All V8 isolation experiments complete!"
echo "============================================================"
echo "Results:"
echo "  V8a: ${DATA_ROOT}/results_v8a/"
echo "  E1:  ${DATA_ROOT}/results_e1/"
echo "  E2:  ${DATA_ROOT}/results_e2/"
echo "  E3:  ${DATA_ROOT}/results_e3/"
