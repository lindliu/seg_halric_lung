#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 12:17:55 2026

@author: dliu
"""
import numpy as np
import pydicom
import matplotlib.pylab as plt
import glob
import os
import imageio
import re

root_path = glob.glob('./data/Rat MIR/Rat 17')[0]
folder = 'Non-gated scan' #  'Resp-gated Scan' # 

num_re = re.compile(r'(\d+)(?!.*\d)')
overlap_path = glob.glob(os.path.join(root_path, folder+'_2_mask_overlap/*'))
path_list = sorted(overlap_path, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))


images = []
for filename in path_list[50:-50]:
    images.append(imageio.imread(filename))
imageio.mimsave(os.path.join(root_path, f'{folder}_gif.gif'), images, fps=10)

