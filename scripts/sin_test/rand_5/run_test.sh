#!/bin/bash
#GPU_ID='2'
# Choose GPU ID with CUDA_VISIBLE_DEVICES={id}
CONFIG_MAIN=avod/configs/simple/test.config
CONFIG_EVALSIN=avod/configs/simple/rand_5/evalsin_test.config
EVAL_CKPTS='2'

# Train model for clean data
python avod/experiments/run_training.py \
        --data_split='train' \
        --pipeline_config=${CONFIG_MAIN}
echo "CLEAN RUN INFERENCE"
# Eval data on validation set (Clean)
python avod/experiments/run_inference.py \
        --experiment_config=${CONFIG_MAIN} \
        --data_split='val' \
        --ckpt_indices ${EVAL_CKPTS}
# echo "SIN RUN INFERENCE"
# # Eval data on validation set (SIN)
# python avod/experiments/run_inference.py \
#         --experiment_config=${CONFIG_EVALSIN} \
#         --data_split='val' \
#         --ckpt_indices ${EVAL_CKPTS}
#         # --output_dir=${OUTPUT_DIR} \        
# echo "SIN CALC AVG KITTI EVAL"
# python ./utils_sin/sin_calc_avg_kitti_eval.py \
#         --experiment_config=${CONFIG_EVALSIN} \
#         --data_split='val'        