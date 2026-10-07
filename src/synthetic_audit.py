"""Tự động tìm lỗi cài sẵn trong data/synthetic (bonus B6): NaN, khoảng trống góc quét, khoảng hở timestamp.

Mỗi luật chạy độc lập trên mọi frame, không đọc trước đáp án:
  - nan:      tỉ lệ điểm NaN/Inf > 0
  - sector:   so mật độ điểm theo từng bin azimuth (5°) với trung vị của cùng bin qua các frame;
              cờ khi >= 3 bin liên tiếp có mật độ < 60% trung vị (loại được cấu trúc cảnh cố định)
  - time_gap: khoảng cách timestamp giữa 2 frame liên tiếp lệch > 50% so với trung vị

Ví dụ:
    python -m src.synthetic_audit
    python -m src.synthetic_audit --help
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_points


def audit(data_root: str, bin_deg: float = 5.0, ratio: float = 0.6, min_bins: int = 3,
          gap_tol: float = 0.5) -> pd.DataFrame:
    frames = list_frames(data_root)
    edges = np.arange(-180, 180 + bin_deg, bin_deg)
    hists, rows = {}, []
    for f in frames:
        p = load_points(data_root, f)
        bad = ~np.isfinite(p).all(axis=1)
        q = p[~bad]
        az = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
        hists[f] = np.histogram(az, bins=edges)[0]
        if bad.any():
            rows.append(dict(defect="NaN/Inf trong point cloud", frame=f,
                             evidence=f"{int(bad.sum())}/{len(p)} điểm = {bad.mean():.2%} không hữu hạn", layer="I/O"))
    med = np.median(np.stack(list(hists.values())), axis=0)
    for f, h in hists.items():
        low = (h < ratio * np.maximum(med, 1))
        run, runs = [], []
        for i, v in enumerate(low):
            if v:
                run.append(i)
            elif run:
                runs.append(run); run = []
        if run:
            runs.append(run)
        for r in runs:
            if len(r) >= min_bins:
                a, b = edges[r[0]], edges[r[-1] + 1]
                rows.append(dict(defect="Mất một sector góc quét", frame=f,
                                 evidence=f"azimuth {a:.0f}° … {b:.0f}°: mật độ {h[r].sum() / med[r].sum():.0%} so với trung vị các frame "
                                          f"(frame có {int(h.sum())} điểm hợp lệ)", layer="I/O / Preprocess"))
    ts_path = Path(data_root) / "training" / "timestamps.txt"
    if ts_path.exists():
        ts = np.loadtxt(ts_path)
        dt = np.diff(ts)
        for i, d in enumerate(dt):
            if abs(d - np.median(dt)) > gap_tol * np.median(dt):
                rows.append(dict(defect="Khoảng hở timestamp", frame=frames[i + 1],
                                 evidence=f"{frames[i]} -> {frames[i + 1]}: Δt = {d:.2f}s, trung vị {np.median(dt):.2f}s", layer="Time"))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tìm lỗi cài sẵn trong data/synthetic bằng các luật độc lập")
    ap.add_argument("--data-root", default="data/synthetic")
    ap.add_argument("--out", default="results/synthetic_defects.csv")
    ap.add_argument("--bin-deg", type=float, default=5.0)
    ap.add_argument("--ratio", type=float, default=0.6, help="bin bị coi là thiếu nếu mật độ < ratio x trung vị")
    args = ap.parse_args()
    df = audit(args.data_root, args.bin_deg, args.ratio)
    # NaN xuất hiện ở mọi frame: gộp thành 1 dòng cho dễ đọc
    nan = df[df.defect.str.startswith("NaN")]
    rest = df.drop(nan.index)
    if len(nan):
        rest = pd.concat([pd.DataFrame([dict(defect=nan.defect.iloc[0], frame=", ".join(nan.frame),
                                             evidence="; ".join(f"{f}: {e.split(' =')[0]}" for f, e in zip(nan.frame, nan.evidence)),
                                             layer="I/O")]), rest], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rest.to_csv(args.out, index=False)
    pd.set_option("display.max_colwidth", 120, "display.width", 250)
    print(rest.to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
