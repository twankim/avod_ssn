import os
import sys

def root_dir():
    return "."
    # return os.path.dirname(os.path.realpath(__file__))


def top_dir():
    return "."
    # avod_root_dir = root_dir()
    # return os.path.split(avod_root_dir)[0]

sys.path.append(os.path.join(top_dir(),'wavedata'))