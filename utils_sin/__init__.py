# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2019-04-08 21:05:57
# @Last Modified by:   twankim
# @Last Modified time: 2019-04-15 15:03:53

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)

this_path = os.path.dirname(__file__)
top_path = os.path.split(this_path)[0]

PATH_AVOD = os.path.join(top_path,'avod')
PATH_WAVEDATA = os.path.join(top_path,'wavedata')

add_path(PATH_AVOD)
add_path(PATH_WAVEDATA)