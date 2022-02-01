#!/bin/bash
#GPU_ID='2'
# Choose GPU ID with CUDA_VISIBLE_DEVICES={id}
CONFIG_MAIN=avod/configs/simple/pyramid_cars_with_aug_simple.config
CONFIG_EVALSIN=avod/configs/simple/rand_5/pyramid_cars_with_aug_simple_evalsin_rand_5.config
CONFIG_EVALAIN=avod/configs/simple/rand_5/pyramid_cars_with_aug_simple_evalain_rand_5.config
# OUTPUT_DIR=data/kitti_avod/object/outputs
# EVAL_CKPTS='60 90 120' 
EVAL_CKPTS='120'

echo "TRAINING run_pyramid_cars_with_aug_simple"
# Train model for clean data
python avod/experiments/run_training.py \
        --pipeline_config=${CONFIG_MAIN} \
        --data_split='train' # \
        # --output_dir=${OUTPUT_DIR}
echo "INFERENCE CLEAN run_pyramid_cars_with_aug_simple"
# Eval data on validation set (Clean)
python avod/experiments/run_inference.py \
        --experiment_config=${CONFIG_MAIN} \
        --data_split='val' \
        --ckpt_indices ${EVAL_CKPTS}
        # --output_dir=${OUTPUT_DIR} \        
echo "INFERENCE SIN run_pyramid_cars_with_aug_simple"
# Eval data on validation set (SIN)
python avod/experiments/run_inference.py \
        --experiment_config=${CONFIG_EVALSIN} \
        --data_split='val' \
        --ckpt_indices ${EVAL_CKPTS}
        # --output_dir=${OUTPUT_DIR} \        
echo "EVAL SIN run_pyramid_cars_with_aug_simple"
python ./utils_sin/sin_calc_avg_kitti_eval.py \
        --experiment_config=${CONFIG_EVALSIN} \
        --data_split='val'
echo "EVAL VAL run_pyramid_cars_with_aug_simple"
# Eval data on validation set (AIN)
python avod/experiments/run_inference.py \
        --experiment_config=${CONFIG_EVALAIN} \
        --data_split='val' \
        --ckpt_indices ${EVAL_CKPTS}
        # --output_dir=${OUTPUT_DIR} \        
echo "EVAL VAL 2 run_pyramid_cars_with_aug_simple"
python ./utils_sin/sin_calc_avg_kitti_eval.py \
        --experiment_config=${CONFIG_EVALAIN} \
        --data_split='val'