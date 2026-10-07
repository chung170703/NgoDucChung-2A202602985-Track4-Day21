"""Hàm dùng chung cho các thí nghiệm topic A (calibration drift).

Ý tưởng "ground truth gắn kết": điểm LiDAR nằm trong 3D box của label (box nằm trong camera
frame, dựng từ calib GỐC chưa perturb) được coi là điểm THUỘC object. Sau khi làm lệch calib,
ta chiếu đúng những điểm đó lên ảnh và kiểm tra chúng còn rơi vào 2D box của object không.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starter.datasets import dataset_type, load_frame
from starter.projection import perturb_extrinsic, project_velo_to_image, velo_to_cam

# Ngưỡng khoảng cách (z_cam của tâm object, mét) chia 3 nhóm gần / vừa / xa
DIST_BUCKETS = (("near_<15m", 0.0, 15.0), ("mid_15-30m", 15.0, 30.0), ("far_>=30m", 30.0, 1e9))
MIN_OBJ_POINTS = 8          # object có ít hơn số điểm này trong 3D box thì bỏ (metric quá nhiễu)
MIN_BASELINE_IN_BOX = 0.8   # object phải có >= 80% điểm rơi trong 2D box ở calib GỐC (loại object bị cắt ở mép ảnh)
BOX_MARGIN_M = 0.10         # nới box 10 cm để bù nhiễu đo


@dataclass
class ObjectPoints:
    frame_id: str
    obj_type: str
    bbox2d: np.ndarray       # (4,) x1, y1, x2, y2
    dist_m: float            # z_cam tâm object
    points_velo: np.ndarray  # (n, 3) điểm thuộc object, toạ độ velodyne
    truncated: float
    occluded: int


def bucket_of(dist_m: float) -> str:
    for name, lo, hi in DIST_BUCKETS:
        if lo <= dist_m < hi:
            return name
    return DIST_BUCKETS[-1][0]


def points_in_box_cam(points_cam: np.ndarray, obj, margin: float = BOX_MARGIN_M) -> np.ndarray:
    """Mask (N,) điểm camera-frame nằm trong 3D box KITTI (location = tâm đáy)."""
    h, w, l = obj.dimensions
    d = points_cam - obj.location
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    # xoay ngược quanh trục y: R(ry)^T @ d
    x = c * d[:, 0] - s * d[:, 2]
    z = s * d[:, 0] + c * d[:, 2]
    y = d[:, 1]
    return ((np.abs(x) <= l / 2 + margin) & (np.abs(z) <= w / 2 + margin) &
            (y <= margin) & (y >= -h - margin))


def _frac_in_box(points_velo: np.ndarray, bbox2d, calib, image_shape) -> float:
    uv, _, _ = project_velo_to_image(points_velo, calib, image_shape)
    x1, y1, x2, y2 = bbox2d
    inside = ((uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)).sum()
    return float(inside) / len(points_velo)


def extract_object_points(fr: dict, frame_id: str, min_points: int = MIN_OBJ_POINTS,
                          min_baseline: float = MIN_BASELINE_IN_BOX) -> list[ObjectPoints]:
    """Điểm thuộc từng object (theo 3D box GT). Loại object ít điểm, và object mà ngay cả với calib
    gốc cũng không khớp 2D box (bị cắt ở mép ảnh, label lệch), vì khi đó không quy được cho perturb."""
    pts = fr["points"][:, :3].astype(np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pc = velo_to_cam(pts, fr["calib"])
    out = []
    for obj in fr["labels"]:
        if obj.location[2] <= 1.0:
            continue
        m = points_in_box_cam(pc, obj)
        if m.sum() < min_points:
            continue
        if _frac_in_box(pts[m], obj.bbox, fr["calib"], fr["image"].shape) < min_baseline:
            continue
        out.append(ObjectPoints(frame_id, obj.type, obj.bbox.copy(), float(obj.location[2]),
                                pts[m], obj.truncated, obj.occluded))
    return out


def load(data_root: str, frame_id: str, **kw) -> dict:
    kwargs = dict(kw) if dataset_type(data_root) == "nuscenes" else {}
    return load_frame(data_root, frame_id, **kwargs)


def in_box_fraction(objs: list[ObjectPoints], calib, image_shape) -> dict[str, tuple[float, int]]:
    """Với calib (đã perturb): (tổng phần-trăm-trong-box của từng object, số object) theo nhóm khoảng cách.
    Điểm bị đẩy ra ngoài ảnh hoặc ra sau camera cũng tính là "rơi ra ngoài box"."""
    acc = {name: [0.0, 0] for name, *_ in DIST_BUCKETS}
    for o in objs:
        a = acc[bucket_of(o.dist_m)]
        a[0] += _frac_in_box(o.points_velo, o.bbox2d, calib, image_shape)
        a[1] += 1
    return {k: tuple(v) for k, v in acc.items()}


def make_perturbed(calib, axis: str, amount: float, sign: int):
    """axis in yaw/pitch/roll (độ) hoặc tx/ty/tz (cm). Chỉ đổi đúng 1 trục."""
    a = sign * amount
    if axis in ("yaw", "pitch", "roll"):
        return perturb_extrinsic(calib, **{f"{axis}_deg": a})
    t = [0.0, 0.0, 0.0]
    t["xyz".index(axis[1])] = a / 100.0
    return perturb_extrinsic(calib, t_xyz_m=tuple(t))
