#!/bin/bash
#GPU_ID='2'
# Choose GPU ID with CUDA_VISIBLE_DEVICES={id}
CONFIG_MAIN=avod/configs/simple/rand_5/trainboth_pyramid_cars_with_aug_simple_rand_5.config
CONFIG_EVALSIN=avod/configs/simple/rand_5/trainboth_pyramid_cars_with_aug_simple_evalsin_rand_5.config
CONFIG_EVALAIN=avod/configs/simple/rand_5/trainboth_pyramid_cars_with_aug_simple_evalain_rand_5.config
OUTPUT_DIR=/data/kitti_avod/object/outputs
# EVAL_CKPTS='0 10 20'
EVAL_CKPTS='30'

# Fine-Tune model for noisy data
python avod/experiments/run_training.py \
        --pipeline_config=${CONFIG_MAIN} \
        --data_split='train' \
        --output_dir=${OUTPUT_DIR}

# Eval data on validation set (Clean)
python avod/experiments/run_inference.py \
        --experiment_config=${CONFIG_MAIN} \
        --data_split='val' \
        --output_dir=${OUTPUT_DIR} \
        --ckpt_indices ${EVAL_CKPTS}

# Eval data on validation set (SIN)
python avod/experiments/run_inference.py \
        --experiment_config=${CONFIG_EVALSIN} \
        --data_split='val' \
        --output_dir=${OUTPUT_DIR} \
        --ckpt_indices ${EVAL_CKPTS}

python ./utils_sin/sin_calc_avg_kitti_eval.py \
        --experiment_config=${CONFIG_EVALSIN} \
        --data_split='val'

# Eval data on validation set (AIN)
python avod/experiments/run_inference.py \
        --experiment_config=${CONFIG_EVALAIN} \
        --data_split='val' \
        --output_dir=${OUTPUT_DIR} \
        --ckpt_indices ${EVAL_CKPTS}

python ./utils_sin/sin_calc_avg_kitti_eval.py \
        --experiment_config=${CONFIG_EVALAIN} \
        --data_split='val'