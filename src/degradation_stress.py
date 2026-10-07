"""Stress test suy giảm dữ liệu (bonus B2): dữ liệu LiDAR xấu đi thì phép kiểm tra calib bị ảnh hưởng thế nào?

Ba loại suy giảm (mỗi loại >= 3 mức, seed cố định): random dropout, nhiễu Gaussian, motion smear.
Với mỗi mức đo:
  - pts_per_object:   số điểm TB còn lại trong 3D box GT của object (object lấy từ lần chạy dữ liệu sạch)
  - pct_in_box_gt:    % điểm của object rơi trong 2D box khi calib ĐÚNG (độ suy giảm tự nó làm hỏng metric đến đâu)
  - false_alarm:      % frame bị score báo drift dù calib đúng (ngưỡng = phân vị 5% của score trên dữ liệu SẠCH)
  - detect_yaw1:      % mẫu yaw +-1° vẫn bị phát hiện (cùng ngưỡng; mẫu không tính được score là NaN thì coi là không phát hiện)
  - score_unavailable: % mẫu không tính được score (quá ít điểm biên độ sâu) -> monitor 'mù' chứ không phải 'an toàn'

Ví dụ:
    python -m src.degradation_stress --data-root data/kitti_mini data/nuscenes_mini_subset
    python -m src.degradation_stress --help
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.alignment_score import EdgeScorer
from src.common import _frac_in_box, extract_object_points, load, make_perturbed, points_in_box_cam
from starter import perturb
from starter.datasets import dataset_type, list_frames
from starter.projection import velo_to_cam

SEED = 0
CONFIGS = [("clean", 0.0)] + [("random_dropout", k) for k in (0.9, 0.7, 0.5, 0.3)] + \
          [("gaussian_noise", s) for s in (0.02, 0.05, 0.1)] + [("motion_smear", v) for v in (10.0, 20.0, 30.0)]


def degrade(points: np.ndarray, kind: str, level: float) -> np.ndarray:
    if kind == "clean":
        return points
    if kind == "random_dropout":
        return perturb.random_dropout(points, level, seed=SEED)
    if kind == "gaussian_noise":
        return perturb.gaussian_noise(points, level, seed=SEED)
    if kind == "motion_smear":
        return perturb.motion_smear(points, level)
    raise ValueError(kind)


def run_dataset(data_root: str, frames: list[str] | None = None) -> pd.DataFrame:
    name = Path(data_root).name
    is_nusc = dataset_type(data_root) == "nuscenes"
    rows = []
    for fid in frames or list_frames(data_root):
        fr = load(data_root, fid)
        clean = fr["points"].astype(np.float32)
        shape, calib = fr["image"].shape, fr["calib"]
        keep_boxes = [o.bbox2d for o in extract_object_points(fr, fid)]   # object hợp lệ ở dữ liệu sạch
        sc = EdgeScorer(fr["image"])
        for kind, level in CONFIGS:
            if is_nusc and kind == "motion_smear":      # motion_smear giả định x = phía trước; LiDAR nuScenes có x = sang phải
                continue
            pts = degrade(clean, kind, level)
            pts = pts[np.isfinite(pts).all(axis=1)]
            pc = velo_to_cam(pts[:, :3], calib)
            n_pts, fracs = [], []
            for o in fr["labels"]:
                if not any(np.array_equal(o.bbox, b) for b in keep_boxes):
                    continue
                m = points_in_box_cam(pc, o)
                n_pts.append(int(m.sum()))
                if m.sum():
                    fracs.append(_frac_in_box(pts[m, :3].astype(np.float64), o.bbox, calib, shape))
            xyz = pts[:, :3].astype(np.float64)
            s0 = sc.score(xyz, calib)[0]
            for sg in (1, -1):
                rows.append(dict(dataset=name, frame=fid, kind=kind, level=level, drift_yaw_deg=sg,
                                 score=sc.score(xyz, make_perturbed(calib, "yaw", 1.0, sg))[0]))
            rows.append(dict(dataset=name, frame=fid, kind=kind, level=level, drift_yaw_deg=0, score=s0,
                             pts_per_object=np.mean(n_pts) if n_pts else np.nan,
                             pct_in_box_gt=100 * np.mean(fracs) if fracs else np.nan))
    return pd.DataFrame(rows)


def summarise(raw: pd.DataFrame, fpr: float = 0.05) -> pd.DataFrame:
    out = []
    for ds, d in raw.groupby("dataset"):
        thr = np.nanpercentile(d[(d.kind == "clean") & (d.drift_yaw_deg == 0)].score, 100 * fpr)
        for (kind, level), g in d.groupby(["kind", "level"], sort=False):
            ok, drift = g[g.drift_yaw_deg == 0], g[g.drift_yaw_deg != 0]
            out.append(dict(dataset=ds, kind=kind, level=level, n_frames=len(ok), threshold=thr,
                            pts_per_object=ok.pts_per_object.mean(), pct_in_box_gt=ok.pct_in_box_gt.mean(),
                            mean_score=ok.score.mean(),
                            false_alarm=100 * (ok.score < thr).mean(),
                            detect_yaw1=100 * (drift.score < thr).mean(),            # NaN (không tính được) KHÔNG tính là phát hiện
                            score_unavailable=100 * g.score.isna().mean()))
    return pd.DataFrame(out)


def plot(s: pd.DataFrame, out: Path) -> None:
    metrics = [("pts_per_object", "điểm TB trên object"), ("pct_in_box_gt", "% điểm trong box (calib đúng)"),
               ("false_alarm", "% báo drift nhầm (calib đúng)"), ("detect_yaw1", "% phát hiện yaw 1°"), ("score_unavailable", "% không tính được score")]
    for ds, d in s.groupby("dataset"):
        kinds = [k for k in d.kind.unique() if k != "clean"]
        fig, axes = plt.subplots(len(kinds), 5, figsize=(20, 2.8 * len(kinds)), squeeze=False)
        clean = d[d.kind == "clean"].iloc[0]
        for r, k in enumerate(kinds):
            g = d[d.kind == k].sort_values("level")
            for c, (m, lab) in enumerate(metrics):
                ax = axes[r, c]
                xs = [0] + list(g.level) if k != "random_dropout" else [1.0] + list(g.level)
                pts = sorted(zip(xs, [clean[m]] + list(g[m])))
                ax.plot([a for a, _ in pts], [b for _, b in pts], marker="o")
                ax.set_title(f"{ds}: {k}", fontsize=8); ax.set_ylabel(lab, fontsize=8); ax.grid(alpha=.3)
                ax.set_xlabel({"random_dropout": "tỉ lệ giữ điểm", "gaussian_noise": "σ (m)", "motion_smear": "tốc độ (m/s)"}[k], fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"degradation_stress_{ds}.png", dpi=100); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stress test suy giảm dữ liệu LiDAR lên metric calibration (B2)")
    ap.add_argument("--data-root", nargs="+", default=["data/kitti_mini"])
    ap.add_argument("--out", default="results/degradation_stress.csv")
    ap.add_argument("--fig-dir", default="results/figures")
    args = ap.parse_args()
    raw = pd.concat([run_dataset(r) for r in args.data_root], ignore_index=True)
    s = summarise(raw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    s.round(3).to_csv(args.out, index=False)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    plot(s, Path(args.fig_dir))
    pd.set_option("display.width", 220)
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
