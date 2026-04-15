
import numpy as np
# import torch
import pydicom
import matplotlib.pylab as plt
import glob
import os 
import re
import tifffile as tiff


def find_lo_hi_from_dcms(paths, percentiles=(1, 99), crop=None, dtype='int16', step_y=1, step_x=1):
    if dtype=='uint8':
        hist = np.zeros(256, dtype=np.uint8)
    elif dtype=='int16':
        hist = np.zeros(65536, dtype=np.int64)

    for path in paths:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array

        if crop is not None:
            arr = arr[crop[0]:crop[1],crop[2]:crop[3]] # [325:825,200:700] #
        arr = arr[::step_y, ::step_x]
        
        if dtype=='uint8':
            hist += np.bincount(arr.ravel(), minlength=256)
        if dtype=='int16':
            vals = arr.astype(np.int32).ravel() + 32768
            hist += np.bincount(vals, minlength=65536)

    cdf = np.cumsum(hist)
    total = cdf[-1]

    p_lo = total * (percentiles[0] / 100.0)
    p_hi = total * (percentiles[1] / 100.0)

    lo = np.searchsorted(cdf, p_lo)
    hi = np.searchsorted(cdf, p_hi)

    if dtype=='int16':
        lo = lo - 32768
        hi = hi - 32768
    return int(lo), int(hi)

num_re = re.compile(r'(\d+)(?!.*\d)')

root_paths = ['./data/Rat MIR/Rat 12']
# path_list = glob.glob('./data/Rat MIR/Rat 4/2_tif/*.tif')
path_list = glob.glob('./data/Rat MIR/Rat 19_post-VILI_19/2_tif/*.tif')

path_list = sorted(path_list, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

print(path_list)
#path_list = path_list[50:250]
Z = len(path_list)
Y, X = plt.imread(path_list[0]).shape

hist = np.zeros(256, dtype=np.int64)
### get 3d image data
# volume = np.zeros([Z, Y, X]).astype('float32')
for path in path_list:
    # volume[i] = plt.imread(path_list[i])[:,:]
    arr =  plt.imread(path)
    hist += np.bincount(arr.ravel(), minlength=256)

print(hist.max(), hist.min())
plt.plot(hist)
plt.savefig('hist_villipost_19.png')