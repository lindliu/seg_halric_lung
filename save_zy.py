#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:49:26 2026

@author: dliu
"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import re
import tifffile as tiff

num_re = re.compile(r'(\d+)(?!.*\d)')

data_name = 'Rat 19_during-VILI_19' # Rat 19_baseline_19 # Rat 4  # Rat 19_post-VILI_19 # Rat 19_during-VILI_19
root = f'./data/Rat MIR/{data_name}'  
paths = glob.glob(os.path.join(root, '2_tif/*.tif'))
paths = sorted(paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

Y, X = plt.imread(paths[0]).shape
Z = len(paths)


samples_idx = np.arange(0,X,20)
samples = np.zeros([len(samples_idx),Z,Y], dtype=np.uint8)
for i, path in enumerate(paths):
    img = plt.imread(path)
    for idx,j in enumerate(samples_idx):
        samples[idx,i,:] = img[:,j]


tif_zy_dir = os.path.join(root, '2_tif_zy')
os.makedirs(tif_zy_dir, exist_ok=True)
for idx, i in enumerate(samples_idx):
    tif_path = os.path.join(tif_zy_dir, data_name+f'_zy_{i}.tif')
    tiff.imwrite(tif_path, samples[idx])







samples_idx = np.arange(0,Y,20)
samples = np.zeros([len(samples_idx),Z,X], dtype=np.uint8)
for i, path in enumerate(paths):
    img = plt.imread(path)
    for idx,j in enumerate(samples_idx):
        samples[idx,i,:] = img[j,:]


tif_zx_dir = os.path.join(root, '2_tif_zx')
os.makedirs(tif_zx_dir, exist_ok=True)
for idx, i in enumerate(samples_idx):
    tif_path = os.path.join(tif_zx_dir, data_name+f'_zx_{i}.tif')
    tiff.imwrite(tif_path, samples[idx])