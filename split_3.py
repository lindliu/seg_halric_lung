import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt


def largest_component(mask: np.ndarray) -> np.ndarray:
    cc, n = label(mask > 0)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    return cc == np.argmax(sizes)


def rotation_matrix_y(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float64)


def rotation_matrix_z(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float64)


def split_by_tilted_lr_plane(
    mask: np.ndarray,
    spacing=(1.0, 1.0, 1.0),   # (dz, dy, dx)
    keep_largest=True,
    angle_y_range=(-12, 12),   # 绕 y 轴微调，影响 z-x 平面里的倾斜
    angle_z_range=(-12, 12),   # 绕 z 轴微调，影响 x-y 平面里的倾斜
    angle_step=2,
    plane_half_thickness_vox=1.5,
    search_steps=200,
    min_balance_ratio=0.15,
    margin_quantile=0.08,
    w_balance=0.8,
    w_center=0.15,
    w_angle=0.05,
):
    """
    在“左右方向”为主的前提下，小角度自动微调切平面。

    返回:
        labels: 0背景, 1左, 2右
        plane_band: 平面附近体素
        info: 最优参数
    """
    if mask.ndim != 3:
        raise ValueError("mask must be 3D")

    mask = mask.astype(bool)
    if keep_largest:
        mask = largest_component(mask)

    pts = np.argwhere(mask)   # z,y,x
    if len(pts) < 10:
        raise ValueError("foreground too small")

    dz, dy, dx = spacing

    # 物理坐标，顺序保持为 [z, y, x]
    pts_phys = np.zeros((len(pts), 3), dtype=np.float64)
    pts_phys[:, 0] = pts[:, 0] * dz
    pts_phys[:, 1] = pts[:, 1] * dy
    pts_phys[:, 2] = pts[:, 2] * dx

    # 基准法向量：左右方向 x
    base_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    delta = plane_half_thickness_vox * min(dz, dy, dx)
    total_n = len(pts)

    best = None

    for ay in range(angle_y_range[0], angle_y_range[1] + 1, angle_step):
        Ry = rotation_matrix_y(ay)

        for az in range(angle_z_range[0], angle_z_range[1] + 1, angle_step):
            Rz = rotation_matrix_z(az)

            normal = Rz @ (Ry @ base_normal)
            normal = normal / (np.linalg.norm(normal) + 1e-12)

            proj = pts_phys @ normal

            q1 = np.quantile(proj, margin_quantile)
            q2 = np.quantile(proj, 1.0 - margin_quantile)

            center_proj = 0.5 * (np.min(proj) + np.max(proj))
            proj_span = np.max(proj) - np.min(proj) + 1e-12

            for t in np.linspace(q1, q2, search_steps):
                signed_dist = proj - t

                left_count = np.sum(signed_dist < -delta)
                right_count = np.sum(signed_dist > delta)
                band_count = np.sum(np.abs(signed_dist) <= delta)

                if left_count == 0 or right_count == 0:
                    continue

                balance = min(left_count, right_count) / max(left_count, right_count)
                if balance < min_balance_ratio:
                    continue

                # 归一化项
                band_term = band_count / total_n
                balance_penalty = 1.0 - balance
                center_penalty = abs(t - center_proj) / proj_span
                angle_penalty = (abs(ay) + abs(az)) / (
                    abs(angle_y_range[1]) + abs(angle_z_range[1]) + 1e-12
                )

                score = (
                    band_term
                    + w_balance * balance_penalty
                    + w_center * center_penalty
                    + w_angle * angle_penalty
                )

                if best is None or score < best["score"]:
                    best = {
                        "score": float(score),
                        "normal": normal.copy(),
                        "t": float(t),
                        "angle_y": int(ay),
                        "angle_z": int(az),
                        "band_count": int(band_count),
                        "left_count": int(left_count),
                        "right_count": int(right_count),
                        "balance": float(balance),
                        "center_penalty": float(center_penalty),
                        "angle_penalty": float(angle_penalty),
                    }

    if best is None:
        raise RuntimeError(
            "没有找到满足条件的平面。可尝试减小 min_balance_ratio，"
            "或扩大 angle_y_range / angle_z_range。"
        )

    final_signed = pts_phys @ best["normal"] - best["t"]

    labels = np.zeros(mask.shape, dtype=np.uint8)
    plane_band = np.zeros(mask.shape, dtype=bool)

    left_idx = final_signed < 0
    right_idx = ~left_idx
    band_idx = np.abs(final_signed) <= delta

    labels[tuple(pts[left_idx].T)] = 1
    labels[tuple(pts[right_idx].T)] = 2
    plane_band[tuple(pts[band_idx].T)] = True

    # 让“左侧=1，右侧=2”更稳定：按 x 坐标均值重排
    coords1 = np.argwhere(labels == 1)
    coords2 = np.argwhere(labels == 2)
    if len(coords1) > 0 and len(coords2) > 0:
        if coords1[:, 2].mean() > coords2[:, 2].mean():
            tmp = labels.copy()
            labels[tmp == 1] = 2
            labels[tmp == 2] = 1

    info = {
        "score": best["score"],
        "plane_normal_zyx": best["normal"],
        "plane_offset_t": best["t"],
        "angle_y_deg": best["angle_y"],
        "angle_z_deg": best["angle_z"],
        "band_voxels": best["band_count"],
        "left_voxels": int(np.sum(labels == 1)),
        "right_voxels": int(np.sum(labels == 2)),
        "balance_ratio": best["balance"],
        "center_penalty": best["center_penalty"],
        "angle_penalty": best["angle_penalty"],
        "delta": float(delta),
    }

    return labels, plane_band, info



mask = np.load('./data/Rat MIR/Rat 9_during-VILI_9/Rat 9_during-VILI_9_masks_0_modified.npy')

labels, plane_band, info = split_by_tilted_lr_plane(
    mask[::8, ::8, ::8],      # 建议先别 ::8，容易把狭窄连接采坏
    spacing=(1.0, 1.0, 1.0),
    keep_largest=True,
    angle_y_range=(-12, 12),
    angle_z_range=(-12, 12),
    angle_step=2,
    plane_half_thickness_vox=1.5,
    search_steps=200,
    min_balance_ratio=0.12,   # 若两肺大小差异很大，可继续降到 0.08
    margin_quantile=0.08,
    w_balance=0.8,
    w_center=0.15,
    w_angle=0.05,
)

print(info)

np.save('Rat_9_plane_split.npy', labels)

mid_z = labels.shape[0] // 2
plt.figure(figsize=(6, 6))
plt.imshow(labels[mid_z])
plt.contour(plane_band[mid_z], levels=[0.5], linewidths=1)
plt.savefig('split_plane.png', dpi=150)
plt.close()


