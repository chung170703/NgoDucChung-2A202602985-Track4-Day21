"""Tạo các ảnh failure case (fail_*.png) và số liệu đi kèm cho topic A.

  fail_01  KITTI: cùng một lỗi yaw 1° nhưng object GẦN trông vẫn khớp, object XA lệch hẳn ra khỏi xe
           -> kiểm tra calib chỉ bằng vật gần sẽ bỏ sót drift (lớp Metric / Geometry).
  fail_02  nuScenes: yaw 3° (~66 px lệch ngang) mà phần lớn mẫu có edge-alignment score KHÔNG vượt ngưỡng
           -> score cần đủ điểm biên độ sâu; LiDAR 32 beam thưa chỉ có ~50-120 điểm (lớp Metric).
  ego_motion_time_offset.csv  nuScenes: bỏ bù chuyển động giữa LiDAR và camera (lớp Time).

Ví dụ:
    python -m src.failure_cases
    python -m src.failure_cases --help
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.alignment_score import EdgeScorer
from src.common import _frac_in_box, extract_object_points, in_box_fraction, load, make_perturbed
from starter.datasets import list_frames
from starter.projection import draw_box2d, overlay_points, project_velo_to_image


def _crop_overlay(fr: dict, calib, bbox, scale: int, margin: float = 0.5, max_depth: float = 60.0):
    """Cắt vùng quanh object, chiếu toàn bộ point cloud với calib cho trước, phóng to để nhìn rõ."""
    img = fr["image"]
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    cx1, cy1, cx2, cy2 = int(max(0, x1 - mw)), int(max(0, y1 - mh)), int(min(W, x2 + mw)), int(min(H, y2 + mh))
    uv, depth, _ = project_velo_to_image(fr["points"], calib, img.shape)
    keep = (uv[:, 0] >= cx1) & (uv[:, 0] < cx2) & (uv[:, 1] >= cy1) & (uv[:, 1] < cy2)
    crop = cv2.resize(img[cy1:cy2, cx1:cx2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    uv_c = (uv[keep] - [cx1, cy1]) * scale
    out = overlay_points(crop, uv_c, depth[keep], max_depth=max_depth, radius=max(2, scale // 2))
    return draw_box2d(out, (np.array(bbox) - [cx1, cy1, cx1, cy1]) * scale)


def fail_01(root: str, out_dir: Path, yaw_deg: float = 1.0) -> list[dict]:
    """Tìm frame KITTI có cả xe gần (<15 m) lẫn xe xa (>=30 m), vẽ 2x2: (gần, xa) x (calib gốc, yaw)."""
    best = None
    for fid in list_frames(root):
        fr = load(root, fid)
        objs = extract_object_points(fr, fid)
        near = [o for o in objs if o.obj_type == "Car" and o.dist_m < 15]
        far = [o for o in objs if o.obj_type == "Car" and 30 <= o.dist_m < 45]
        if near and far:
            n, f = max(near, key=lambda o: len(o.points_velo)), max(far, key=lambda o: len(o.points_velo))
            best = (fid, fr, n, f)
            break
    fid, fr, near, far = best
    calib0, calib1 = fr["calib"], make_perturbed(fr["calib"], "yaw", yaw_deg, 1)
    rows, fig, axes = [], plt.figure(figsize=(12, 7)), None
    axes = fig.subplots(2, 2)
    for r, (obj, scale, name) in enumerate([(near, 3, "GẦN"), (far, 8, "XA")]):
        for c, (cal, tag) in enumerate([(calib0, "calib gốc"), (calib1, f"yaw +{yaw_deg:g}°")]):
            pct = 100 * _frac_in_box(obj.points_velo, obj.bbox2d, cal, fr["image"].shape)
            axes[r, c].imshow(cv2.cvtColor(_crop_overlay(fr, cal, obj.bbox2d, scale), cv2.COLOR_BGR2RGB))
            axes[r, c].set_title(f"Xe {name} ({obj.dist_m:.0f} m) – {tag}: {pct:.0f}% điểm trong box", fontsize=10)
            axes[r, c].axis("off")
            rows.append(dict(case="fail_01", frame=fid, obj=name, dist_m=round(obj.dist_m, 1), calib=tag, pct_in_box=round(pct, 1)))
    fig.suptitle(f"KITTI {fid}: cùng một lỗi calib, vật gần vẫn 'trông ổn', vật xa lệch hẳn khỏi xe", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fail_01_yaw1deg_far_car_slides_off.png", dpi=100)
    plt.close(fig)
    return rows


def fail_02(root: str, out_dir: Path, frames_csv: str, yaw_deg: float = 3.0) -> list[dict]:
    """nuScenes: chọn frame yaw 3° có score cao nhất nhưng vẫn lệch rõ; vẽ overlay + phân phối score."""
    d = pd.read_csv(frames_csv)
    d = d[d.dataset == Path(root).name]
    thr = float(np.nanpercentile(d[d.level == 0].score, 5))
    und = d[(d.axis == "yaw") & (d.level == yaw_deg) & (d.score >= thr)]
    row = und.sort_values("score", ascending=False).iloc[0]
    fr = load(root, row.frame)
    cal = make_perturbed(fr["calib"], "yaw", yaw_deg, int(row.sign))
    uv, depth, _ = project_velo_to_image(fr["points"], cal, fr["image"].shape)
    vis = overlay_points(fr["image"], uv, depth, radius=3)
    for o in fr["labels"]:
        vis = draw_box2d(vis, o.bbox)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax[0].axis("off")
    ax[0].set_title(f"nuScenes {row.frame}, yaw {int(row.sign) * yaw_deg:+g}° (frame có score CAO NHẤT trong các mẫu yaw 3°): "
                    f"score = {row.score:.2f} > ngưỡng {thr:.2f}, không báo drift", fontsize=9)
    base = d[d.level == 0].score.dropna(); pert = d[(d.axis == "yaw") & (d.level == yaw_deg)].score.dropna()
    bins = np.linspace(0, 0.5, 26)
    ax[1].hist(base, bins, alpha=.6, density=True, label=f"calib gốc (n={len(base)})"); ax[1].hist(pert, bins, alpha=.6, density=True, label=f"yaw {yaw_deg:g}° (n={len(pert)})")
    ax[1].axvline(thr, color="r", ls="--", label="ngưỡng FPR 5%"); ax[1].legend(fontsize=8)
    ax[1].set_xlabel("edge-alignment score"); ax[1].set_ylabel("mật độ"); ax[1].set_title("hai phân phối chồng lên nhau", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "fail_02_nuscenes_yaw3deg_score_blind.png", dpi=100); plt.close(fig)
    n_und = 100 * float((pert >= thr).mean())
    return [dict(case="fail_02", frame=row.frame, score=round(float(row.score), 3), thr=round(thr, 3),
                 pct_yaw3_undetected=round(n_und, 1), n_edge_pts=int(row.n_edge_pts))]


def ego_motion_table(root: str, out_csv: Path) -> pd.DataFrame:
    """Đo độ lệch tương đương khi tắt bù chuyển động xe giữa LiDAR và camera (nuScenes)."""
    rows = []
    for f in list_frames(root):
        on, off = load(root, f, use_ego_motion=True), load(root, f, use_ego_motion=False)
        dt = (on["timestamp_camera_us"] - on["timestamp_lidar_us"]) / 1e6
        shift = float(np.linalg.norm(on["calib"].T_cam_velo[:3, 3] - off["calib"].T_cam_velo[:3, 3]))
        objs = extract_object_points(on, f, min_baseline=0.0)
        n = len(objs)
        frac = lambda cal: np.mean([_frac_in_box(o.points_velo, o.bbox2d, cal, on["image"].shape) for o in objs]) if n else np.nan
        sc = EdgeScorer(on["image"])
        pts = on["points"][:, :3].astype(float); pts = pts[np.isfinite(pts).all(1)]
        rows.append(dict(frame=f, dt_lidar_to_cam_ms=round(dt * 1e3, 2), equiv_translation_m=round(shift, 3),
                         ego_speed_mps=round(abs(shift / dt), 2), n_objects=n,
                         pct_in_box_compensated=100 * frac(on["calib"]), pct_in_box_uncompensated=100 * frac(off["calib"]),
                         score_compensated=sc.score(pts, on["calib"])[0], score_uncompensated=sc.score(pts, off["calib"])[0]))
    df = pd.DataFrame(rows).round(3)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Sinh ảnh failure case fail_01, fail_02 và bảng ego-motion cho topic A")
    ap.add_argument("--kitti", default="data/kitti_mini")
    ap.add_argument("--nuscenes", default="data/nuscenes_mini_subset")
    ap.add_argument("--frames-csv", default="results/alignment_score_frames.csv", help="đầu ra của src.alignment_score")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--out-csv", default="results/failure_cases.csv")
    ap.add_argument("--ego-csv", default="results/ego_motion_time_offset.csv")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = fail_01(args.kitti, out_dir) + fail_02(args.nuscenes, out_dir, args.frames_csv)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    ego = ego_motion_table(args.nuscenes, Path(args.ego_csv))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))
    print(ego.describe().loc[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
