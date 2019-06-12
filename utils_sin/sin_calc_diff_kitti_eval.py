# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2019-04-12 16:26:29
# @Last Modified by:   twankim
# @Last Modified time: 2019-05-21 15:49:23

import argparse
import os
import sys

import avod.builders.config_builder_util as config_builder
from utils_sin.sin_utils import SINFields
import numpy as np
from scipy.stats import sem

EVAL_NAMES = ['{}_detection AP',
              '{}_orientation AP',
              '{}_detection_BEV AP',
              '{}_heading_BEV AP',
              '{}_detection_3D AP',
              '{}_heading_3D AP']

CI_FACTOR = 1.96

def calc_diff_scores(output_dir, model_config, eval_config, 
                     dataset_config, data_split, iou05=False):
    if iou05:
        res_name = 'results_05_iou'
    else:
        res_name = 'results'

    score_threshold = np.round(eval_config.kitti_score_threshold,3)

    do_eval_sin = eval_config.do_eval_sin


    res_dir = os.path.join(output_dir,model_config.checkpoint_name,
                           'offline_eval',res_name+'_sin_{}_{}_{}/{}/'.format(
                                eval_config.sin_type,
                                eval_config.sin_level, 
                                eval_config.sin_repeat,
                                SINFields.SIN_INPUT_NAMES[0]))
    fname = '{}_{}_{}_{}_{}_rep.txt'.format(model_config.checkpoint_name,
                                            res_name,
                                            str(score_threshold),
                                            data_split,
                                            0)
    ckpt_list = get_ckpt_list(os.path.join(res_dir,fname))

    list_class = [class_name.lower() for class_name in dataset_config.classes]
    list_eval_names = []
    for eval_name in EVAL_NAMES:
        for class_name in list_class:
            list_eval_names.append(eval_name.format(class_name))

    # dict_res -> image/lidar -> sin_repeat x num_ckpt x num_eval_names x 3
    dict_res = {} # Save results 

    # dict_res_avg -> checkpoint -> image/lidar -> num_eval_names x 3(easy,moderate,hard)
    dict_res_avg = {} # Save average AP score
    # dict_res_sem -> checkpoint -> image/lidar -> num_eval_names x 3(easy,moderate,hard)
    dict_res_sem = {} # Save standard error of mean

    # Initialize dictionaries
    for ckpt in ckpt_list:
        dict_res_avg[ckpt] = {}
        dict_res_sem[ckpt] = {}
        dict_res_avg[SINFields.SIN_METRIC_NAME] = np.zeros((len(list_eval_names),3))
        dict_res_sem[SINFields.SIN_METRIC_NAME] = np.zeros((len(list_eval_names),3))

    for sin_input_name in SINFields.SIN_INPUT_NAMES:
        print('   ... Loading {}'.format(sin_input_name))
        res_dir = os.path.join(output_dir,model_config.checkpoint_name,
                               'offline_eval',res_name+'_sin_{}_{}_{}/{}/'.format(
                                    eval_config.sin_type,
                                    eval_config.sin_level, 
                                    eval_config.sin_repeat,
                                    sin_input_name))
        # Initialize result dictionary
        dict_res[sin_input_name] = np.zeros((eval_config.sin_repeat,
                                             len(ckpt_list),
                                             len(list_eval_names),
                                             3))
        # Read result files for differen idx_repeat
        for idx_repeat in range(eval_config.sin_repeat):
            fname = '{}_{}_{}_{}_{}_rep.txt'.format(model_config.checkpoint_name,
                                                    res_name,
                                                    str(score_threshold),
                                                    data_split,
                                                    idx_repeat)
            load_res_file(os.path.join(res_dir,fname),
                                       dict_res[sin_input_name][idx_repeat],
                                       list_eval_names)
    
        for idx_ckpt,ckpt in enumerate(ckpt_list):
            # Calcuate mean AP over repeats
            res_temp = dict_res[sin_input_name][:,idx_ckpt,:,:].mean(axis=0)
            dict_res_avg[ckpt][sin_input_name] = res_temp
            # Calculate confidence interval
            sem_temp = CI_FACTOR * sem(dict_res[sin_input_name][:,idx_ckpt,:,:],axis=0)
            dict_res_sem[ckpt][sin_input_name] = sem_temp
    
    for idx_ckpt,ckpt in enumerate(ckpt_list):
        stack_res = np.asarray([dict_res_avg[ckpt][sin_input_name] \
                                for sin_input_name in SINFields.SIN_INPUT_NAMES])
        min_idx = stack_res.argmin(axis=0)
        res_temp = stack_res.min(axis=0)
        idx2 = np.tile(np.arange(len(list_eval_names)), (3,1)).T
        idx3 = np.tile(np.arange(3), (len(list_eval_names),1))
        # res_temp = stack_res[min_idx,idx2,idx3]

        stack_sem = np.asarray([dict_res_sem[ckpt][sin_input_name] \
                                for sin_input_name in SINFields.SIN_INPUT_NAMES])
        sem_temp = stack_sem[min_idx,idx2,idx3] # sem of a corresponding min average value
        
        dict_res_avg[ckpt][SINFields.SIN_METRIC_NAME] = res_temp
        dict_res_sem[ckpt][SINFields.SIN_METRIC_NAME] = sem_temp

    # Find abs(AP_lidar-AP_rgb) and save
    print('   ... Writing final_diff file')
    fname_final_diff = os.path.join(output_dir,model_config.checkpoint_name,
                                    'offline_eval',res_name+'_sin_{}_{}_{}/{}_{}_{}_{}_diff.txt'.format(
                                        eval_config.sin_type,
                                        eval_config.sin_level, 
                                        eval_config.sin_repeat,
                                        model_config.checkpoint_name,
                                        res_name,
                                        str(score_threshold),
                                        data_split))
    with open(fname_final_diff,'w') as f_final_diff:
        for idx_ckpt,ckpt in enumerate(ckpt_list):
            stack_res = np.asarray([dict_res_avg[ckpt][sin_input_name] \
                                    for sin_input_name in SINFields.SIN_INPUT_NAMES])
            min_idx = stack_res.argmin(axis=0)
            max_idx = stack_res.argmax(axis=0)
            res_temp = stack_res.max(axis=0) - stack_res.min(axis=0)
            idx2 = np.tile(np.arange(len(list_eval_names)), (3,1)).T
            idx3 = np.tile(np.arange(3), (len(list_eval_names),1))
            # res_temp = stack_res[min_idx,idx2,idx3]

            stack_sem = np.asarray([dict_res_sem[ckpt][sin_input_name] \
                                    for sin_input_name in SINFields.SIN_INPUT_NAMES])
            sem_temp = stack_sem[max_idx,idx2,idx3] + stack_sem[min_idx,idx2,idx3]
            
            f_final_diff.write('checkpoint: '+ str(ckpt) + '(absolute max diff)\n')
            for idx_eval_name, eval_name in enumerate(list_eval_names):
                f_final_diff.write(eval_name + ":" + \
                              ','.join(['{:.6f} ({:.6f})'.format(val,se) for (val,se) \
                                        in zip(res_temp[idx_eval_name],sem_temp[idx_eval_name])]) +\
                              '\n')


def get_ckpt_list(res_file):
    ckpt_list = []
    with open(res_file,'r') as fid:
        for line in fid:
            num_candidate = line.split('\n')[0]
            if num_candidate.isdigit():
                ckpt_list.append(int(num_candidate))
    return ckpt_list

def load_res_file(res_file,res_mat,list_eval_names):
    first_eval_name = list_eval_names[0]
    idx_ckpt = -1
    
    with open(res_file,'r') as fid:
        for line in fid:
            line_items = line.split(':')
            if len(line_items)<2:
                continue
            eval_name = line_items[0]
            eval_scores = line_items[1].split('\n')[0].split(' ')[1:]
            eval_scores = [np.float(score) for score in eval_scores]

            assert eval_name in list_eval_names, \
                'Something is wrong. {} is not a valid metric'.format(eval_name)
            idx_eval_name = list_eval_names.index(eval_name)

            if eval_name == first_eval_name:
                idx_ckpt += 1
            
            # Store data
            res_mat[idx_ckpt][idx_eval_name] = eval_scores

def main(args):
    model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            args.experiment_config_path, is_training=False,
            output_dir=args.output_dir)

    assert eval_config.do_eval_sin, \
        "This code only supports repeated evaluation in eval_sin or eval_ain modes"

    print('..Working on {}'.format(args.experiment_config_path))
    calc_diff_scores(args.output_dir,model_config,eval_config,dataset_config,args.data_split)
    
    print('..Working on {} (iou05)'.format(args.experiment_config_path))
    calc_diff_scores(args.output_dir,model_config,eval_config,dataset_config,args.data_split,iou05=True)

def parse_args():
    default_output_dir = '/data/kitti_avod/object/outputs'

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir',
                        type=str,
                        dest='output_dir',
                        default=default_output_dir,
                        help='output dir to save checkpoints')

    parser.add_argument('--experiment_config',
                        type=str,
                        required=True,
                        dest='experiment_config_path',
                        help='Path to the experiment config must be specified')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        required=True,
                        help='Data split must be specified e.g. val or test')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print("Called with args:")
    print(args)
    sys.exit(main(args))
