#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 12:17:55 2026

@author: dliu
"""

from glob import glob
import pickle
import re
import os 
import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

io.logger_setup() # run this to get printing of progress

#Check if colab notebook instance has GPU access
if core.use_gpu()==False:
  raise ImportError("No GPU access, change your runtime")


def has_holes_3d(vol, connectivity=1):
    vol = vol.astype(bool)
    bg = ~vol

    # 3D 连通性：1=6邻域, 2=26邻域
    st = ndi.generate_binary_structure(3, connectivity)

    # 标记背景连通域
    lab, n = ndi.label(bg, structure=st)
    if n == 0:
        return False, 0, None

    # 边界上的背景标签 = 外部背景
    border = np.zeros_like(vol, dtype=bool)
    border[0,:,:] = border[-1,:,:] = True
    border[:,0,:] = border[:,-1,:] = True
    border[:,:,0] = border[:,:,-1] = True

    external_ids = np.unique(lab[border & bg])
    external = np.isin(lab, external_ids)

    holes = bg & (~external)
    holes_count = int(ndi.label(holes, structure=st)[1])
    return holes_count > 0, holes_count, holes


def keep_k_component(masks, top_k=1):
    # masks: shape (Z, X, Y), binary (0/1 or False/True)
    binary = masks > 0

    # 26-连通（最常用，3D 里算“接触”就连）
    structure = np.ones((3, 3, 3), dtype=int)

    labeled, num_components = ndi.label(binary, structure=structure)
    print("不连通区块数:", num_components)

    sizes = ndi.sum(binary, labeled, index=range(1, num_components+1))
    top_k = 1
    top_labels = np.argsort(sizes)[-top_k:] + 1   # label 从 1 开始
    mask_top = np.isin(labeled, top_labels)
    return mask_top
    

import open3d as o3d
def plot_3d_show(masks):
    points = np.argwhere(masks > 0)  ## masks (Z,X,Y), points (Z,X,Y)
    color = masks[points[:,0], points[:,1], points[:,2]]/np.max(masks)


    # 自定义颜色，每个点对应一个 RGB 值（范围 [0, 1]）
    colors = np.zeros_like(points, dtype=float)      # 所有点初始为黑色
    colors[:, 0] = color          # 红色通道 = x 坐标
    colors[:, 1] = color/1.5          # 绿色通道 = y 坐标
    colors[:, 2] = color/2          # 蓝色通道 = z 坐标

    # 构建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    o3d.visualization.draw_geometries([pcd])


import plotly.graph_objects as go
def plot_3d_save(masks, save_path=None):
    points_np = np.argwhere(masks)
    colors_np = [200,155,75]

    max_pts = 200000
    if points_np.shape[0] > max_pts:
        points_np = points_np[np.random.choice(points_np.shape[0], max_pts, replace=False)]

    fig = go.Figure(go.Scatter3d(x=points_np[:,2], y=points_np[:,1], z=points_np[:,0], mode="markers",
                                marker=dict(size=1, opacity=0.8, color=colors_np)))
    fig.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    if save_path is not None:
        fig.write_html(save_path)


### load dataset
num_re = re.compile(r'(\d+)(?!.*\d)')


type_data = 'model_during_post' #  'bleo'  # 'control_baseline' #  
### load model
if type_data == 'control_baseline':
    new_model_path = './models/model_control_baseline'
    model = models.CellposeModel(gpu=True, pretrained_model=new_model_path)
    #7,9,10,11,14,15,16,17,18,19,20,#1,2,4,5,6
    root_path_list = [
                      './data/Rat MIR/Rat 7',
                      './data/Rat MIR/Rat 9_baseline_9',
                      './data/Rat MIR/Rat 11_baseline_11',
                      './data/Rat MIR/Rat 14_baseline_14',
                      './data/Rat MIR/Rat 16_baseline_16',
                      './data/Rat MIR/Rat 17',
                      './data/Rat MIR/Rat 18',
                      './data/Rat MIR/Rat 19_baseline_19',
                      './data/Rat MIR/Rat 20']

if type_data == 'model_during_post':
    new_model_path = './models/model_during_post'
    model = models.CellposeModel(gpu=True, pretrained_model=new_model_path)
    # 9,10,11,14,16,19
    root_path_list = ['./data/Rat MIR/Rat 9_during-VILI_9',
                      './data/Rat MIR/Rat 9_post_VILI_9',
                      './data/Rat MIR/Rat 11_during-VILI_11',
                      './data/Rat MIR/Rat 11_post-VILI_11',
                      './data/Rat MIR/Rat 14_during-VILI_14',
                      './data/Rat MIR/Rat 14_post-VILI_14',
                      './data/Rat MIR/Rat 16_during-VILI_16',
                      './data/Rat MIR/Rat 19_during-VILI_19',
                      './data/Rat MIR/Rat 19_post-VILI_19']

if type_data == 'bleo':
    new_model_path = './models/bleo'
    model = models.CellposeModel(gpu=True, pretrained_model=new_model_path)
    # 1,2,3,4,12,13
    # root_path_list = glob('./data/Rat MIR/*') 
    root_path_list = ['./data/Rat MIR/Rat 4',
                    './data/Rat MIR//Rat 12',
                    './data/Rat MIR//Rat 13']

# cellprob_threshold = -1.5
for cellprob_threshold in [0,-0.5,-1,-2,-2.5]:
    for root_path in root_path_list:
        print(root_path)
        type_data = os.path.split(root_path)[1]

        path_list = glob(os.path.join(root_path, '2_tif/*.tif'))
        path_list = sorted(path_list, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))
        #path_list = path_list[50:250]

        Z = len(path_list)
        X, Y = plt.imread(path_list[0]).shape

        volume = np.zeros([Z, X, Y]).astype('float32')
        for i in range(Z):
            volume[i] = plt.imread(path_list[i])[:,:]

        # computes flows from 2D slices and combines into 3D flows to create masks
        masks_, flows, _ = model.eval(volume, z_axis=0, channel_axis=None,
                                        batch_size=32,
                                        do_3D=True, 
                                        cellprob_threshold=cellprob_threshold, 
                                        flow3D_smooth=1)
        masks_ = masks_!=0
        print(masks_.sum())
        # plot_3d_show(masks)
        np.save(os.path.join(root_path, type_data+f'_masks_{abs(cellprob_threshold)}.npy'), masks_)
        plot_3d_save(masks_, save_path=os.path.join(root_path, type_data+f'_masks_{abs(cellprob_threshold)}.html'))

        # ### keep top k components
        # masks = keep_k_component(masks_, top_k=1)
        
        # ### fill all holes
        # has, num, holes_mask = has_holes_3d(masks, connectivity=1)
        # print(f'the number of holes: {num}')
        # masks[holes_mask] = True

        # ### save results
        # np.save(os.path.join(root_path, type_data+'_1comp_masks.npy'), masks)
        # plot_3d_save(masks, save_path=os.path.join(root_path, type_data+'_1comp_masks.html'))

        # mask_dir = os.path.join(root_path, type_data+'_2_mask')
        # over_dir = os.path.join(root_path, type_data+'_2_mask_overlap')
        # os.makedirs(mask_dir, exist_ok=True)
        # os.makedirs(over_dir, exist_ok=True)

        # masks = ((masks!=0).astype(np.uint8) * 255)
        # for i in range(masks.shape[0]):
        #     tif_path = os.path.join(mask_dir, type_data+f'_mask_{i}.tif')
        #     tiff.imwrite(tif_path, masks[i])

        #     plt.figure()
        #     plt.imshow(volume[i])
        #     plt.imshow(masks[i], alpha=.25)
        #     plt.savefig(os.path.join(over_dir, type_data+f'_mask_overlap_{i}.png'))
        #     plt.close()


