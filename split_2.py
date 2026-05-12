import numpy as np
from scipy.ndimage import label, gaussian_filter1d


def largest_component(mask: np.ndarray) -> np.ndarray:
    cc, n = label(mask > 0)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    return cc == np.argmax(sizes)


def split_by_mid_valley(
    mask: np.ndarray,
    keep_largest=True,
    axis=2,                 # 2 表示 x 方向（你的 z,y,x 顺序里最后一维）
    smooth_sigma=2.0,
    margin_quantile=0.15,   # 只在中间 70% 范围找谷底
    min_balance_ratio=0.2,  # 防止切太偏
):
    """
    在指定轴上寻找“截面积曲线”的局部低谷作为切平面。
    
    返回:
        labels: 0背景, 1左, 2右
        info: 调试信息
        area_curve: 原始截面积
        area_smooth: 平滑后截面积
    """
    if mask.ndim != 3:
        raise ValueError("mask must be 3D")

    mask = mask.astype(bool)
    if keep_largest:
        mask = largest_component(mask)

    if mask.sum() == 0:
        raise ValueError("empty mask")

    # 统计每个切片面积
    other_axes = tuple(i for i in range(3) if i != axis)
    area_curve = mask.sum(axis=other_axes).astype(np.float64)

    # 只在 mask 实际存在的范围里找
    valid = np.where(area_curve > 0)[0]
    if len(valid) < 3:
        raise ValueError("valid slices too few")

    lo = valid[0]
    hi = valid[-1]

    # 平滑
    area_smooth = gaussian_filter1d(area_curve, sigma=smooth_sigma)

    # 只在中间范围找，避免跑到边缘
    q_lo = int(np.floor(lo + margin_quantile * (hi - lo)))
    q_hi = int(np.ceil(hi - margin_quantile * (hi - lo)))
    q_lo = max(q_lo, lo + 1)
    q_hi = min(q_hi, hi - 1)

    best_x = None
    best_score = None

    total = mask.sum()

    for x in range(q_lo, q_hi + 1):
        left = np.sum(mask.take(indices=range(lo, x), axis=axis))
        right = np.sum(mask.take(indices=range(x, hi + 1), axis=axis))

        if left == 0 or right == 0:
            continue

        balance = min(left, right) / max(left, right)
        if balance < min_balance_ratio:
            continue

        # 只优化“谷底最小”
        score = area_smooth[x]

        if best_score is None or score < best_score:
            best_score = score
            best_x = x
            best_balance = balance
            best_left = left
            best_right = right

    if best_x is None:
        raise RuntimeError("没有找到合适谷底；可把 min_balance_ratio 调低到 0.1")

    labels = np.zeros(mask.shape, dtype=np.uint8)

    slicer_left = [slice(None)] * 3
    slicer_right = [slice(None)] * 3
    slicer_left[axis] = slice(None, best_x)
    slicer_right[axis] = slice(best_x, None)

    labels[tuple(slicer_left)] = mask[tuple(slicer_left)] * 1
    labels[tuple(slicer_right)] = mask[tuple(slicer_right)] * 2

    info = {
        "split_index": int(best_x),
        "axis": int(axis),
        "valley_area_raw": float(area_curve[best_x]),
        "valley_area_smooth": float(area_smooth[best_x]),
        "left_voxels": int(best_left),
        "right_voxels": int(best_right),
        "balance_ratio": float(best_balance),
        "search_range": [int(q_lo), int(q_hi)],
    }

    return labels, info, area_curve, area_smooth


mask = np.load('./data/Rat MIR/Rat 9_during-VILI_9/Rat 9_during-VILI_9_masks_0_modified.npy')

labels, info, area_curve, area_smooth = split_by_mid_valley(
    mask[::8, ::8, ::8],
    keep_largest=True,
    axis=2,              # x方向
    smooth_sigma=2.0,
    margin_quantile=0.15,
    min_balance_ratio=0.15,
)

print(info)

np.save('Rat 9_during-VILI_9_plane_split.npy', labels)

import matplotlib.pyplot as plt
plt.imshow(labels[80])
plt.savefig('plane_split.png')
plt.close()

plt.figure()
plt.plot(area_curve, label='raw')
plt.plot(area_smooth, label='smooth')
plt.axvline(info["split_index"], color='r', linestyle='--', label='split')
plt.legend()
plt.savefig('area_curve.png')
plt.close()