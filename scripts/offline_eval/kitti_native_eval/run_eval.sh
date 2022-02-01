#!/bin/bash
set -e
cd $1 
# ../teams/DSC180A_FA21_A00/a15/avod_data/kitti_avod/object/outputs/test/predictions_sin_rand_5.0_5/image/kitti_native_eval/
# ./evaluate_object_3d_offline ~/Kitti/object/training/label_2/ $2/$3 | tee -a ./$4_results_$2.txt
~/avod_ssn/scripts/offline_eval/kitti_native_eval/evaluate_object_3d_offline ~/teams/DSC180A_FA21_A00/a15/avod_data/Kitti/object/training/label_2/ $2/$3 | tee -a ./$4_results_$2.txt
cwd=$(pwd)
cd ~/avod_ssn
echo $4_results_$2.txt
cp -R $cwd/$4_results_$2.txt $5
cd $cwd