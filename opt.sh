#! /usr/bin/env python

# "_indiana_piano_14_4"
EXP_PREFIX=$1
ts=$(date '+%b_%I%P' | sed 's/^0//')

# Combine into full experiment name
EXP="${EXP_PREFIX}_${ts}"

# indiana_piano_14_14
# nus_cpr_08_1
SEQ_NAME="_indiana_piano_14_4"

python dance_glb.py \
   --seq_name  "$SEQ_NAME" \
   --depth_type 'moge' \
   --exp        "$EXP"
