
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import re
import tifffile as tiff

num_re = re.compile(r'(\d+)(?!.*\d)')

root = './data/Rat MIR/Rat 19_baseline'
paths = glob.glob(os.path.join(root, '2_tif/*.tif'))

paths = sorted(paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

Y, X = plt.imread(paths[0]).shape
Z = len(paths)

j = 0

samples_idx = np.arange(0,1024,20)
samples = np.zeros([len(samples_idx),Z,Y], dtype=np.int8)
for i, path in enumerate(paths):
    img = plt.imread(path)
    for idx,j in enumerate(samples_idx):
        samples[idx,i,:] = img[:,j]


tif_zy_dir = os.path.join(root, '2_tif_zy')
os.makedirs(tif_zy_dir, exist_ok=True)
for idx, i in enumerate(samples_idx):
    tif_path = os.path.join(tif_zy_dir, f'tif_zy_{i}.tif')
    tiff.imwrite(tif_path, samples[idx])