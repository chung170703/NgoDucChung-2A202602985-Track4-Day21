"""Alignment score (không cần label): biên độ sâu của LiDAR có nằm trên biên ảnh không?

Cách tính:
  1. Chiếu điểm LiDAR lên ảnh, đặt vào một depth map thưa.
  2. "Điểm biên độ sâu" = điểm mà độ sâu trong cửa sổ lân cận chênh nhau lớn
     ((max-min)/depth > ratio), tức là điểm nằm ở mép vật thể.
  3. Canny trên ảnh xám -> distance transform: khoảng cách (pixel) từ mỗi pixel tới biên ảnh gần nhất.
  4. score = mean( exp(-dist / sigma) ) trên các điểm biên độ sâu.  Calib đúng -> mép vật khớp biên ảnh
     -> score cao; calib lệch -> mép vật lệch khỏi biên ảnh -> score thấp.

Ngưỡng phát hiện drift = phân vị thấp (mặc định 5%) của score ở calib GỐC trên cùng dataset
(tức false-alarm khoảng 5%). Baseline đối chứng: "% điểm trong FOV" (metric rẻ nhất, không cần ảnh).

Ví dụ:
    python -m src.alignment_score --data-root data/kitti_mini data/nuscenes_mini_subset
    python -m src.alignment_score --data-root data/kitti_mini --latency
    python -m src.alignment_score --help
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calib_sweep import AXES
from src.common import load, make_perturbed
from starter.datasets import list_frames
from starter.projection import project_velo_to_image

BIG = 1000.0


class EdgeScorer:
    """Tiền xử lý ảnh một lần (Canny + distance transform), rồi chấm điểm nhiều calib khác nhau."""

    def __init__(self, image: np.ndarray, sigma_px: float = 4.0, ratio: float = 0.3, win: int | None = None):
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        self.dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
        self.shape, self.sigma, self.ratio = image.shape, sigma_px, ratio
        # LiDAR thưa hơn trên ảnh lớn (nuScenes 1600x900): cửa sổ phải rộng hơn để thấy lân cận
        self.win = win or (11 if image.shape[1] < 1400 else 21)

    def score(self, points: np.ndarray, calib) -> tuple[float, int]:
        uv, depth, _ = project_velo_to_image(points, calib, self.shape)
        H, W = self.shape[:2]
        if len(uv) < 50:
            return float("nan"), 0
        x = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
        y = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
        far = np.zeros((H, W), np.float32)             # độ sâu lớn nhất (0 = trống)
        near = np.zeros((H, W), np.float32)            # BIG - độ sâu nhỏ nhất (0 = trống)
        np.maximum.at(far, (y, x), depth.astype(np.float32))
        np.maximum.at(near, (y, x), (BIG - depth).astype(np.float32))
        k = np.ones((self.win, self.win), np.uint8)
        local_max = cv2.dilate(far, k)
        local_min = BIG - cv2.dilate(near, k)
        sel = (local_max[y, x] - local_min[y, x]) / depth > self.ratio
        if sel.sum() < 20:
            return float("nan"), int(sel.sum())
        return float(np.exp(-self.dist[y[sel], x[sel]] / self.sigma).mean()), int(sel.sum())


def collect(data_root: str, frames: list[str] | None, kw: dict) -> pd.DataFrame:
    name = Path(data_root).name
    rows = []
    for fid in frames or list_frames(data_root):
        fr = load(data_root, fid)
        pts = fr["points"][:, :3].astype(np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
        sc = EdgeScorer(fr["image"], **kw)
        calib, shape = fr["calib"], fr["image"].shape
        for axis, levels in AXES.items():
            for lvl in levels:
                for sg in ((1,) if lvl == 0 else (1, -1)):
                    cp = make_perturbed(calib, axis, lvl, sg)
                    s, n_edge = sc.score(pts, cp)
                    _, _, m = project_velo_to_image(pts, cp, shape)
                    rows.append(dict(dataset=name, frame=fid, axis=axis, level=lvl, sign=sg,
                                     score=s, n_edge_pts=n_edge, fov_ratio=float(m.mean())))
    return pd.DataFrame(rows)


def agg_detect(vals: np.ndarray, base: np.ndarray, k: int, fpr: float, rng) -> float:
    """% tập k frame (lấy ngẫu nhiên, seed cố định) có điểm trung bình thấp hơn ngưỡng FPR của các tập baseline."""
    def means(a):
        a = a[np.isfinite(a)]
        return np.array([a[rng.choice(len(a), min(k, len(a)), replace=False)].mean() for _ in range(300)])
    thr = np.percentile(means(base), 100 * fpr)
    return 100 * float((means(vals) < thr).mean())


def summarise(frames_df: pd.DataFrame, fpr: float, k: int = 10, seed: int = 0) -> pd.DataFrame:
    out = []
    for ds, d in frames_df.groupby("dataset"):
        base = d[d.level == 0]
        thr_edge = np.nanpercentile(base.score, 100 * fpr)
        thr_fov = np.nanpercentile(base.fov_ratio, 100 * fpr)
        rng = np.random.default_rng(seed)
        for (axis, lvl), g in d.groupby(["axis", "level"]):
            out.append(dict(dataset=ds, axis=axis, level=lvl, n_samples=len(g),
                            mean_score=g.score.mean(), std_score=g.score.std(),
                            thr_edge=thr_edge, detect_rate_edge=100 * (g.score < thr_edge).mean(),
                            mean_fov_ratio=g.fov_ratio.mean(), thr_fov=thr_fov,
                            detect_rate_fov=100 * (g.fov_ratio < thr_fov).mean(),
                            detect_rate_edge_mean_of_k=agg_detect(g.score.values, base.score.values, k, fpr, rng), k=k))
    return pd.DataFrame(out)


def plot(summary: pd.DataFrame, out_dir: Path) -> None:
    for ds, d in summary.groupby("dataset"):
        fig, axes = plt.subplots(1, 6, figsize=(21, 3.6), sharey=True)
        for ax, axis in zip(axes, AXES):
            s = d[d.axis == axis].sort_values("level")
            ax.errorbar(s.level, s.mean_score, yerr=s.std_score, marker="o", capsize=3)
            ax.axhline(s.thr_edge.iloc[0], color="r", ls="--", label="ngưỡng (FPR 5%)")
            ax.set_title(f"{ds}: {axis}"); ax.set_xlabel("độ" if axis in ("yaw", "pitch", "roll") else "cm")
            ax.grid(alpha=.3)
        axes[0].set_ylabel("edge-alignment score"); axes[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out_dir / f"alignment_score_{ds}.png", dpi=110); plt.close(fig)


def latency(data_root: str, n: int, kw: dict) -> dict:
    fid = list_frames(data_root)[0]
    fr = load(data_root, fid)
    pts = fr["points"][:, :3].astype(np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    sc = EdgeScorer(fr["image"], **kw)
    sc.score(pts, fr["calib"])                              # lần chạy đầu (khởi tạo) bỏ đi
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); sc.score(pts, fr["calib"]); t.append((time.perf_counter() - t0) * 1e3)
    t_pre = []
    for _ in range(n):
        t0 = time.perf_counter(); EdgeScorer(fr["image"], **kw); t_pre.append((time.perf_counter() - t0) * 1e3)
    return dict(dataset=Path(data_root).name, frame=fid, n_points=len(pts), n_runs=n,
                score_p50_ms=np.percentile(t, 50), score_p95_ms=np.percentile(t, 95),
                image_prep_p50_ms=np.percentile(t_pre, 50), image_prep_p95_ms=np.percentile(t_pre, 95),
                hardware=f"{platform.processor() or platform.machine()} / {platform.system()} {platform.release()} / "
                         f"OpenCV {cv2.__version__} (CPU, 1 luồng Python)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Edge-alignment score + ngưỡng phát hiện calibration drift")
    ap.add_argument("--data-root", nargs="+", default=["data/kitti_mini"])
    ap.add_argument("--sigma-px", type=float, default=4.0, help="độ rộng exp(-d/sigma)")
    ap.add_argument("--ratio", type=float, default=0.3, help="ngưỡng (max-min)/depth để coi là điểm biên độ sâu")
    ap.add_argument("--win", type=int, default=None, help="cửa sổ lân cận (px); mặc định 11 (ảnh KITTI) / 21 (ảnh lớn)")
    ap.add_argument("--fpr", type=float, default=0.05, help="tỉ lệ báo động giả khi chọn ngưỡng")
    ap.add_argument("--agg-k", type=int, default=10, help="số frame gộp khi tính detect_rate_edge_mean_of_k")
    ap.add_argument("--latency", action="store_true", help="đo latency p50/p95 (bỏ lần đầu, 20 lần) thay vì chạy sweep")
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--out-prefix", default="results/alignment_score")
    ap.add_argument("--fig-dir", default="results/figures")
    args = ap.parse_args()
    kw = dict(sigma_px=args.sigma_px, ratio=args.ratio, win=args.win)

    if args.latency:
        df = pd.DataFrame([latency(r, args.n_runs, kw) for r in args.data_root])
        Path("results").mkdir(exist_ok=True)
        df.round(2).to_csv("results/alignment_latency.csv", index=False)
        print(df.drop(columns="hardware").round(2).to_string(index=False)); print(df.hardware.iloc[0])
        return

    frames_df = pd.concat([collect(r, None, kw) for r in args.data_root], ignore_index=True)
    summary = summarise(frames_df, args.fpr, args.agg_k)
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    frames_df.round(4).to_csv(f"{args.out_prefix}_frames.csv", index=False)
    summary.round(3).to_csv(f"{args.out_prefix}.csv", index=False)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    plot(summary, Path(args.fig_dir))
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
