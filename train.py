#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 12:17:55 2026

@author: dliu
"""
import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt

io.logger_setup() # run this to get printing of progress

#Check if colab notebook instance has GPU access
if core.use_gpu()==False:
  raise ImportError("No GPU access, change your runtime")


# import gc, torch
# gc.collect()
# torch.cuda.empty_cache()      # releases cached blocks back to CUDA driver
# torch.cuda.ipc_collect()   


train_dir = './data/annotation/train/control_baseline'
# train_dir = './data/annotation/train/during_post'
# train_dir = './data/annotation/train/bleo'

test_dir = './data/annotation/test/Rat_17_Non_gated_scan'
masks_ext = "_seg.npy"



# model = models.CellposeModel(gpu=True)

new_model_path = './models/model_1'
model = models.CellposeModel(gpu=True,
                             pretrained_model=new_model_path)



from cellpose import train
import os.path

# model_name = "model_control_baseline"
# model_name = "model_during_post"
model_name = "model_"+os.path.split(train_dir)[1]

# default training params
n_epochs = 1000
learning_rate = 1e-5
weight_decay = 0.1
batch_size = 1

# get files
output = io.load_train_test_data(train_dir, test_dir, mask_filter=masks_ext)
train_data, train_labels, _, test_data, test_labels, _ = output
# (not passing test data into function to speed up training)

# if not os.path.isfile(f'./models/{model_name}'):
new_model_path, train_losses, test_losses = train.train_seg(model.net,
                                                                train_data=train_data,
                                                                train_labels=train_labels,
                                                                batch_size=batch_size,
                                                                n_epochs=n_epochs,
                                                                learning_rate=learning_rate,
                                                                weight_decay=weight_decay,
                                                                nimg_per_epoch=max(2, len(train_data)), # can change this
                                                                model_name=model_name,
                                                                min_train_masks=1)

# from cellpose import metrics

# new_model_path = './models/model_2'
# model = models.CellposeModel(gpu=True,
#                              pretrained_model=new_model_path)

# # run model on test images
# masks = model.eval(test_data, batch_size=1)[0]

# # check performance using ground truth labels
# ap = metrics.average_precision(test_labels, masks)[0]
# print('')
# print(f'>>> average precision at iou threshold 0.5 = {ap[:,0].mean():.3f}')


# i = 0

# fig, axe = plt.subplots(1, 2, figsize=(10,8), dpi=150)
# axe[0].imshow(test_data[i])
# axe[0].imshow(masks[i], alpha=.5)
# axe[0].axis('off')
# axe[0].set_title('prediction')

# axe[1].imshow(test_data[i])
# axe[1].imshow(test_labels[i], alpha=.5)
# axe[1].axis('off')
# axe[1].set_title('ground truth')
# plt.tight_layout()