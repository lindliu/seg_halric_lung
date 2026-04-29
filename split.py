import numpy as np
from scipy.ndimage import distance_transform_edt, label
from sklearn.decomposition import PCA
import maxflow


def largest_component(mask: np.ndarray) -> np.ndarray:
    cc, n = label(mask > 0)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    return cc == np.argmax(sizes)


def choose_seeds_by_pca(mask: np.ndarray, seed_quantile: float = 0.05):
    pts = np.argwhere(mask)
    pca = PCA(n_components=1)
    proj = pca.fit_transform(pts).ravel()

    q_low = np.quantile(proj, seed_quantile)
    q_high = np.quantile(proj, 1.0 - seed_quantile)

    left_seed = np.zeros_like(mask, dtype=bool)
    right_seed = np.zeros_like(mask, dtype=bool)

    left_pts = pts[proj <= q_low]
    right_pts = pts[proj >= q_high]

    left_seed[tuple(left_pts.T)] = True
    right_seed[tuple(right_pts.T)] = True

    overlap = left_seed & right_seed
    left_seed[overlap] = False
    right_seed[overlap] = False
    return left_seed, right_seed


def connected_cleanup(labels, mask):
    """
    labels: 0背景, 1左, 2右
    对左右各自只保留最大连通分量，其余碎片并回另一边。
    """
    out = labels.copy()

    for target in [1, 2]:
        comp, n = label(out == target)
        if n <= 1:
            continue
        sizes = np.bincount(comp.ravel())
        sizes[0] = 0
        keep = np.argmax(sizes)
        bad = (comp > 0) & (comp != keep)
        out[bad] = 3 - target  # 1->2, 2->1

    out[~mask] = 0
    return out


def extract_cut_surface(labels):
    """
    返回界面附近体素（布尔图）
    """
    left = labels == 1
    right = labels == 2

    cut = np.zeros_like(labels, dtype=bool)

    # 6 邻域接触处标记为 cut
    cut[:-1, :, :] |= left[:-1, :, :] & right[1:, :, :]
    cut[1:, :, :] |= left[1:, :, :] & right[:-1, :, :]

    cut[:, :-1, :] |= left[:, :-1, :] & right[:, 1:, :]
    cut[:, 1:, :] |= left[:, 1:, :] & right[:, :-1, :]

    cut[:, :, :-1] |= left[:, :, :-1] & right[:, :, 1:]
    cut[:, :, 1:] |= left[:, :, 1:] & right[:, :, :-1]

    return cut


def estimate_cut_area(labels, spacing=(1.0, 1.0, 1.0)):
    """
    估计切割面面积：统计左右标签相邻的 6 邻域面数 * 面积
    """
    dz, dy, dx = spacing
    area = 0.0

    # z 相邻 => 面积 dx*dy
    a = labels[:-1, :, :]
    b = labels[1:, :, :]
    area += np.sum(((a == 1) & (b == 2)) | ((a == 2) & (b == 1))) * (dx * dy)

    # y 相邻 => 面积 dx*dz
    a = labels[:, :-1, :]
    b = labels[:, 1:, :]
    area += np.sum(((a == 1) & (b == 2)) | ((a == 2) & (b == 1))) * (dx * dz)

    # x 相邻 => 面积 dy*dz
    a = labels[:, :, :-1]
    b = labels[:, :, 1:]
    area += np.sum(((a == 1) & (b == 2)) | ((a == 2) & (b == 1))) * (dy * dz)

    return area


def lung_split_graphcut(
    mask: np.ndarray,
    spacing=(1.0, 1.0, 1.0),   # (dz, dy, dx)
    beta=2.0,
    seed_quantile=0.05,
    seed_strength=1e6,
    keep_largest=True,
    cleanup=True,
):
    """
    输入:
        mask: 3D 二值肺图
    输出:
        labels: 0背景, 1左, 2右
        cut_surface: 界面附近体素
        info: 调试信息
    """
    if mask.ndim != 3:
        raise ValueError("mask 必须为 3D")

    mask = mask.astype(bool)
    if keep_largest:
        mask = largest_component(mask)

    if mask.sum() == 0:
        raise ValueError("mask 为空")

    dz, dy, dx = spacing
    dist = distance_transform_edt(mask, sampling=spacing)
    left_seed, right_seed = choose_seeds_by_pca(mask, seed_quantile)

    # 前景 voxel 编号
    pts = np.argwhere(mask)
    n = len(pts)

    g = maxflow.Graph[float](n, n * 3)
    nodes = g.add_nodes(n)

    # 坐标 -> node id
    coord2id = {tuple(p): i for i, p in enumerate(pts)}

    # 加 terminal edges
    for i, (z, y, x) in enumerate(pts):
        sc = seed_strength if left_seed[z, y, x] else 0.0
        tc = seed_strength if right_seed[z, y, x] else 0.0
        g.add_tedge(i, sc, tc)

    # 6 邻域
    neighbors = [
        ((1, 0, 0), dx * dy),  # z方向相邻，切开面积 dx*dy
        ((0, 1, 0), dx * dz),  # y方向相邻，切开面积 dx*dz
        ((0, 0, 1), dy * dz),  # x方向相邻，切开面积 dy*dz
    ]

    for i, (z, y, x) in enumerate(pts):
        for (oz, oy, ox), face_area in neighbors:
            zz, yy, xx = z + oz, y + oy, x + ox
            key = (zz, yy, xx)
            if key not in coord2id:
                continue

            j = coord2id[key]

            avg_dist = 0.5 * (dist[z, y, x] + dist[zz, yy, xx])

            # 越靠“粗”区域 dist 越大，切割应更不愿意经过
            # 所以这里让边权随 dist 增大而增大
            w = face_area * (1.0 + beta * avg_dist)

            # 无向边 => 双向同容量
            g.add_edge(i, j, w, w)

    flow = g.maxflow()

    labels = np.zeros(mask.shape, dtype=np.uint8)
    for i, (z, y, x) in enumerate(pts):
        seg = g.get_segment(i)  # 0 / 1，具体哪边对应 source 不用死记
        # PyMaxflow里通常 get_segment(i)==0 表示 source side
        labels[z, y, x] = 1 if seg == 0 else 2

    if cleanup:
        labels = connected_cleanup(labels, mask)

    cut_surface = extract_cut_surface(labels)

    vol1 = np.sum(labels == 1)
    vol2 = np.sum(labels == 2)
    cut_area = estimate_cut_area(labels, spacing)
    balance = min(vol1, vol2) / max(vol1, vol2)

    info = {
        "flow": float(flow),
        "vol_left": int(vol1),
        "vol_right": int(vol2),
        "balance_ratio": float(balance),
        "cut_area": float(cut_area),
        "left_seed_voxels": int(left_seed.sum()),
        "right_seed_voxels": int(right_seed.sum()),
    }

    return labels, cut_surface, info


mask = np.load('./data/Rat MIR/Rat 9_post_VILI_9/Rat 9_post_VILI_9_masks_0_modified.npy')
# mask: 3D numpy array, 肺=1, 背景=0
labels, cut_surface, info = lung_split_graphcut(
    mask,
    spacing=(1.0, 1.0, 1.0),   # 换成你的真实 spacing
    beta=1.5,
    seed_quantile=0.03,
    seed_strength=1e6
)

print(info)
# labels == 1 -> 一侧
# labels == 2 -> 另一侧
# cut_surface == True -> 分割界面附近

np.save('split.npy', labels)