import pydicom
import tifffile as tiff
from PIL import Image
import pandas as pd

import re, os, pickle, glob
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, remove_small_holes, ball, binary_opening, skeletonize
from cellpose import dynamics
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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

    if num_components == 0:
        return np.zeros_like(binary, dtype=bool)
    
    sizes = ndi.sum(binary, labeled, index=range(1, num_components+1))
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



structure = np.ones((3, 3, 3), dtype=bool)
### reduce influnce from z
# structure = np.zeros((3, 3, 3), dtype=bool)
# structure[1, :, :] = True
# structure[:, 1, 1] = True


### load dataset
num_re = re.compile(r'(\d+)(?!.*\d)')

# root_path_list = ['./data/Rat MIR/Rat 19',\
#                   './data/Rat MIR/Rat 17']
root_path_list = glob.glob('./data/Rat MIR/*')
# root_path_list = [
#                     './data/Rat MIR/Rat 9_during-VILI_9',
                    # './data/Rat MIR/Rat 9_post_VILI_9',
#                     './data/Rat MIR/Rat 11_during-VILI_11',
#                     './data/Rat MIR/Rat 11_post-VILI_11',
#                     './data/Rat MIR/Rat 14_during-VILI_14',
#                     './data/Rat MIR/Rat 14_post-VILI_14',
#                     './data/Rat MIR/Rat 16_during-VILI_16',
#                     './data/Rat MIR/Rat 19_during-VILI_19',
#                     './data/Rat MIR/Rat 19_post-VILI_19'
                    # ]



for root_path in root_path_list:
    type_data = os.path.split(root_path)[1]

    print(root_path)

    path_list = glob.glob(os.path.join(root_path, '2_tif/*.tif'))
    path_list = sorted(path_list, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))
    #path_list = path_list[50:250]
    Z = len(path_list)
    Y, X = plt.imread(path_list[0]).shape

    ### get 3d image data
    volume = np.zeros([Z, Y, X]).astype('float32')
    for i in range(Z):
        volume[i] = plt.imread(path_list[i])[:,:]

    ## get lung masks
    masks_ = np.load(os.path.join(root_path, type_data+'_masks_0.npy')) != 0
    # plot_3d_save(masks_, save_path=os.path.join(root_path, type_data+'_masks.html'))
    # plot_3d_show(masks_)

    # dP = np.load(os.path.join(root_path, type_data+'_flow_dP.npy'))          # 3D flow field
    # cellprob = np.load(os.path.join(root_path, type_data+'_flow_cellprob.npy'))   # 3D cell probability
    # print(dP.shape, cellprob.shape)
    for cellprob_threshold in [0]:#[-4,-2,-1,-.5,0]:
        print(cellprob_threshold)
        # run Cellpose dynamics and make masks
        # # niter can be tuned; this is a common default-style choice
        # # masks_ = dynamics.compute_masks(
        # #     dP,
        # #     cellprob,
        # #     cellprob_threshold=cellprob_threshold,
        # #     do_3D=True,
        # #     device=device
        # # )
        # shape = volume.shape
        # masks_ = dynamics.resize_and_compute_masks(
        #         dP, 
        #         cellprob, 
        #         cellprob_threshold=cellprob_threshold,
        #         do_3D=True,
        #         min_size=5, 
        #         max_size_fraction=.6, 
        #         resize=shape[:3] if (np.array(dP.shape[-3:])!=np.array(shape[:3])).sum() 
        #                 else None,
        #         device=device)
        # masks_ = masks_!=0
        # # np.save(os.path.join(root_path, type_data+f'_masks_{abs(cellprob_threshold)}.npy'), masks_)
        # plot_3d_save(masks_, save_path=os.path.join(root_path, type_data+f'_masks_{abs(cellprob_threshold)}.html'))

        
        ### keep top k components
        masks = keep_k_component(masks_, top_k=1)
        
        ### closing operation, fill small hole
        masks = ndi.binary_closing(masks, structure, iterations=3)
        
        ### fill all holes
        has, num, holes_mask = has_holes_3d(masks, connectivity=1)
        print(f'the number of holes: {num}')
        masks[holes_mask] = True

        ### save results
        np.save(os.path.join(root_path, type_data+f'_masks_{abs(cellprob_threshold)}_modified.npy'), masks)
        plot_3d_save(masks, save_path=os.path.join(root_path, type_data+f'_masks_{abs(cellprob_threshold)}_modified.html'))
        # masks = np.load(os.path.join(root_path, type_data+'_masks_modified.npy'))

        lung = volume*masks
        # thr = np.percentile(lung, 99.5)  # tune 95–99.5

        mask_dir = os.path.join(root_path, type_data+f'_2_mask_{abs(cellprob_threshold)}')
        over_dir = os.path.join(root_path, type_data+f'_2_mask_overlap_{abs(cellprob_threshold)}')
        lung_dir = os.path.join(root_path, type_data+f'_2_lung_{abs(cellprob_threshold)}')
        # lung1_dir = os.path.join(root_path, type_data+'_2_lung_bright')
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(over_dir, exist_ok=True)
        os.makedirs(lung_dir, exist_ok=True)
        # os.makedirs(lung1_dir, exist_ok=True)


        masks = ((masks!=0).astype(np.uint8) * 255)
        for i in range(masks.shape[0]):
            tif_path = os.path.join(mask_dir, f'mask_{i}.tif')
            tiff.imwrite(tif_path, masks[i])

            lung_path = os.path.join(lung_dir, f'lung_{i}.tif')
            tiff.imwrite(lung_path, lung[i])

            # lung1_path = os.path.join(lung1_dir, f'lung_{i}.tif')
            # tiff.imwrite(lung1_path, lung[i]*lung_bright_bin[i])

            plt.figure()
            plt.imshow(volume[i])
            plt.imshow(masks[i], alpha=.25)
            plt.savefig(os.path.join(over_dir, f'mask_overlap_{i}.png'))
            plt.close()
