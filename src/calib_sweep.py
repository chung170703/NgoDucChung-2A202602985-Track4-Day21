"""Thí nghiệm chính topic A: calibration drift -> % điểm rơi trong 2D box / % điểm trong FOV.

Mỗi lần chỉ đổi MỘT trục (yaw/pitch/roll theo độ, tx/ty/tz theo cm), đối xứng hai dấu (+/-) rồi
gộp. Thí nghiệm hoàn toàn xác định (không có yếu tố ngẫu nhiên), nên chạy lại ra đúng số.

Ví dụ:
    python -m src.calib_sweep --data-root data/kitti_mini data/nuscenes_mini_subset
    python -m src.calib_sweep --help
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import DIST_BUCKETS, extract_object_points, in_box_fraction, load, make_perturbed
from starter.datasets import list_frames
from starter.projection import project_velo_to_image

ROT_LEVELS = (0.0, 0.5, 1.0, 2.0, 3.0)          # độ
TRANS_LEVELS = (0.0, 2.0, 5.0, 10.0)            # cm
AXES = {"yaw": ROT_LEVELS, "pitch": ROT_LEVELS, "roll": ROT_LEVELS,
        "tx": TRANS_LEVELS, "ty": TRANS_LEVELS, "tz": TRANS_LEVELS}


def run_dataset(data_root: str, max_frames: int | None, frames: list[str] | None) -> list[dict]:
    name = Path(data_root).name
    frame_ids = frames or list_frames(data_root)
    if max_frames:
        frame_ids = frame_ids[:max_frames]
    rows = []
    # acc[(axis, level, bucket)] = [tổng %-trong-box của các object, số (object x dấu)]; fov[(axis, level)] = [kept_in_fov, base_in_fov, all_in_fov, all]
    acc: dict = {}
    fov: dict = {}
    for fid in frame_ids:
        fr = load(data_root, fid)
        objs = extract_object_points(fr, fid)
        calib, shape = fr["calib"], fr["image"].shape
        _, _, base_mask = project_velo_to_image(fr["points"], calib, shape)
        for axis, levels in AXES.items():
            for lvl in levels:
                signs = (1,) if lvl == 0 else (1, -1)
                for sg in signs:
                    cp = make_perturbed(calib, axis, lvl, sg)
                    for b, (fsum, n) in in_box_fraction(objs, cp, shape).items():
                        a = acc.setdefault((axis, lvl, b), [0.0, 0])
                        a[0] += fsum; a[1] += n
                    _, _, m = project_velo_to_image(fr["points"], cp, shape)
                    f = fov.setdefault((axis, lvl), [0, 0, 0, 0])
                    f[0] += int((m & base_mask).sum()); f[1] += int(base_mask.sum())
                    f[2] += int(m.sum());               f[3] += len(m)
    n_obj = {(axis, b): acc[(axis, 0.0, b)][1] for (axis, lvl, b) in acc if lvl == 0}
    for (axis, lvl, b), (fsum, n) in acc.items():
        kept, base, allin, allp = fov[(axis, lvl)]
        rows.append(dict(dataset=name, axis=axis, level=lvl, unit="deg" if axis in ("yaw", "pitch", "roll") else "cm",
                         bucket=b, n_frames=len(frame_ids), n_objects=n_obj[(axis, b)],
                         pct_in_box=100.0 * fsum / n if n else np.nan,
                         pct_points_in_fov=100.0 * allin / allp,
                         pct_baseline_fov_kept=100.0 * kept / base))
    return rows


def plot(df: pd.DataFrame, out: Path) -> None:
    for ds, d in df.groupby("dataset"):
        fig, axes = plt.subplots(1, 6, figsize=(21, 3.6), sharey=True)
        for ax, axis in zip(axes, AXES):
            s = d[d.axis == axis]
            for b, g in s.groupby("bucket"):
                ax.plot(g.level, g.pct_in_box, marker="o", label=b)
            ax.set_title(f"{ds}: {axis}"); ax.set_xlabel("độ" if axis in ("yaw", "pitch", "roll") else "cm")
            ax.grid(alpha=.3)
        axes[0].set_ylabel("% điểm object còn trong 2D box"); axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"calib_sweep_{ds}.png", dpi=110)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep calibration drift (yaw/pitch/roll/tx/ty/tz) và đo % điểm object còn trong 2D box")
    ap.add_argument("--data-root", nargs="+", default=["data/kitti_mini"], help="một hoặc nhiều thư mục dataset")
    ap.add_argument("--max-frames", type=int, default=None, help="chỉ dùng N frame đầu mỗi dataset (debug)")
    ap.add_argument("--out", default="results/calib_sweep.csv")
    ap.add_argument("--fig-dir", default="results/figures")
    args = ap.parse_args()

    rows = []
    for root in args.data_root:
        rows += run_dataset(root, args.max_frames, None)
    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(out, index=False)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    plot(df, Path(args.fig_dir))
    pd.set_option("display.width", 200)
    print(df[df.axis.isin(["yaw", "tx"])].round(1).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
