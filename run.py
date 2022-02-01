import sys
import os
import json
import shutil

from avod.experiments import run_evaluation, run_inference, run_training
from utils_avod import avod_utils

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''
    if 'clean' in targets:
        shutil.rmtree("./outputs")
        os.mkdir("outputs")
        
    if 'test' in targets:
        with open('config/test.json') as fh:
            test_config = json.load(fh)

        # make the data target
        avod_utils.run_main_with_command_line_args(run_training, **(test_config['training']))
        avod_utils.run_main_with_command_line_args(run_inference, **(test_config['inference']))

    return


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)