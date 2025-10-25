#! /usr/bin/env python

# "_indiana_piano_14_4"
# _nus_cpr_08_1
# _cmu_bike_74_7
# _mit_dance_02_12
# _cmu_soccer_07_3
# _iiith_cooking_123_2
EXP_PREFIX=$1

# Get timestamp like May_7pm
ts=$(date '+%b_%I%P' | sed 's/^0//')

# Combine into full experiment name
EXP="${EXP_PREFIX}_${ts}"

# indiana_piano_14_14
# nus_cpr_08_1
SEQ_NAME="_indiana_piano_14_14"

python dance_glb.py \
   --seq_name  "$SEQ_NAME" \
   --depth_type 'moge' \
   --exp        "$EXP"
