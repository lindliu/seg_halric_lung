
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import re
import tifffile as tiff

# num_re = re.compile(r'(\d+)(?!.*\d)')

# root = './data/Rat MIR/Rat 19_baseline'
# paths = glob.glob(os.path.join(root, '2_tif/*.tif'))

# paths = sorted(paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

# Y, X = plt.imread(paths[0]).shape
# Z = len(paths)

# j = 0

# samples_idx = np.arange(0,1024,20)
# samples = np.zeros([len(samples_idx),Z,Y], dtype=np.int8)
# for i, path in enumerate(paths):
#     img = plt.imread(path)
#     for idx,j in enumerate(samples_idx):
#         samples[idx,i,:] = img[:,j]


# tif_zy_dir = os.path.join(root, '2_tif_zy')
# os.makedirs(tif_zy_dir, exist_ok=True)
# for idx, i in enumerate(samples_idx):
#     tif_path = os.path.join(tif_zy_dir, f'tif_zy_{i}.tif')
#     tiff.imwrite(tif_path, samples[idx])

mask = np.load('./data/Rat MIR/Rat 9_during-VILI_9/Rat 9_during-VILI_9_masks_10.npy')
mask = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)
mask = mask!=0

flow0 = np.load('./data/Rat MIR/Rat 9_during-VILI_9/Rat 9_during-VILI_9_flow010.npy')
flow1 = np.load('./data/Rat MIR/Rat 9_during-VILI_9/Rat 9_during-VILI_9_flow110.npy')
flow2 = np.load('./data/Rat MIR/Rat 9_during-VILI_9/Rat 9_during-VILI_9_flow210.npy')
flow3 = np.load('./data/Rat MIR/Rat 9_during-VILI_9/Rat 9_during-VILI_9_flow310.npy')

plt.imshow(flow0[60,:,:,:])
plt.savefig('flow0.png')
plt.imshow(flow1[0,60,:,:])
plt.savefig('flow1.png')
plt.imshow(flow2[60]>-10)
plt.savefig('flow2.png')
plt.imshow(flow3[60]>-10)
plt.savefig('flow3.png')

plt.imshow(mask[60])
plt.savefig('b.png')



# voxels eligible to become part of a mask
cp_mask = flow2 > -10
# inspect one slice
plt.imshow(cp_mask[60], cmap='gray')
plt.savefig('re .png')


from cellpose import dynamics
import numpy as np
import matplotlib.pyplot as plt

# raw outputs from model.eval(...)
dP = flow1          # 3D flow field
cellprob = flow2    # 3D cell probability

# run Cellpose dynamics and make masks
# niter can be tuned; this is a common default-style choice
mask_recomputed = dynamics.compute_masks(
    dP,
    cellprob,
    cellprob_threshold=-10,
    do_3D=True,
)

# inspect one slice
plt.imshow(mask_recomputed[60], cmap='gray')
plt.savefig('re.png')
