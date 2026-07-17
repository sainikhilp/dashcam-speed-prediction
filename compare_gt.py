"""
Compare pipeline speed predictions against the GPS ground truth.

Works on every video folder under --outputs whose <name>_speeds.csv contains
both the predictions (from predict_speed.py) and the gt_speed_kmh columns
(from extract_gt.py):

    python compare_gt.py --outputs ../outputs

What it does per video:
1. aggregates the 20 fps predictions to 1 Hz (mean within each second),
   matching the 1 Hz resolution of the GPS overlay
2. groups both signals into --interval second intervals (default 5) and
   computes: mean GT, mean prediction, absolute error, percent error
   (percent only where GT > 10 km/h - percentages are meaningless near 0)
3. writes  <name>_gt_comparison.csv  (the interval table)
   and      <name>_gt_comparison.png (GT vs prediction + error bars)
4. prints a per-clip and overall summary (MAE, moving-only MAE, mean %err,
   and MSE in m/s for the comma.ai benchmark scale)

All three prediction variants are scored: FILTERED is the system's actual
output; RAW isolates the model's own calibration bias (the honest baseline
that fine-tuning must improve); SMOOTHED shows what plain averaging does.
"""

import argparse
import csv
import glob
import os

import numpy as np

VARIANTS = [("raw", "speed_kmh"), ("smooth", "speed_kmh_smooth"),
            ("filtered", "speed_kmh_filtered")]
PCT_MIN_KMH = 10.0        # below this GT speed a percent error is not meaningful

# chart colors: GT = dark neutral ink, filtered = blue, raw = light blue
C_GT, C_FILT, C_RAW = "#374151", "#2563eb", "#93c5fd"


def load_speeds_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    need = ["time_s", "gt_speed_kmh"] + [c for _, c in VARIANTS]
    for c in need:
        if c not in rows[0]:
            return None
    return {c: np.array([float(r[c]) for r in rows]) for c in need}


def per_second(t, values, n_seconds):
    """Mean of `values` within each whole second [n, n+1)."""
    sec = np.minimum(t.astype(int), n_seconds - 1)
    return np.array([values[sec == n].mean() for n in range(n_seconds)])


def analyze(name, folder, data, interval):
    t = data["time_s"]
    n_seconds = int(np.floor(t.max())) + 1
    gt = per_second(t, data["gt_speed_kmh"], n_seconds)
    preds = {v: per_second(t, data[col], n_seconds) for v, col in VARIANTS}

    # ---- interval table ----
    starts = np.arange(0, n_seconds, interval)
    table = []
    for s in starts:
        e = min(s + interval, n_seconds)
        row = {"start_s": s, "end_s": e, "gt_kmh": gt[s:e].mean()}
        for v, _ in VARIANTS:
            p = preds[v][s:e].mean()
            row[f"{v}_kmh"] = p
            row[f"{v}_abs_err_kmh"] = abs(p - row["gt_kmh"])
            row[f"{v}_pct_err"] = (100.0 * abs(p - row["gt_kmh"]) / row["gt_kmh"]
                                   if row["gt_kmh"] > PCT_MIN_KMH else None)
        table.append(row)

    csv_path = os.path.join(folder, f"{name}_gt_comparison.csv")
    cols = list(table[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in table:
            w.writerow(["" if row[c] is None else
                        (row[c] if isinstance(row[c], (int, np.integer)) else round(row[c], 2))
                        for c in cols])

    # ---- per-clip summary (from the 1 Hz series, not the interval means) ----
    moving = gt > PCT_MIN_KMH
    summary = {"clip": name, "seconds": n_seconds, "moving_s": int(moving.sum())}
    for v, _ in VARIANTS:
        err = np.abs(preds[v] - gt)
        summary[f"{v}_mae_kmh"] = err.mean()
        summary[f"{v}_mae_moving_kmh"] = err[moving].mean() if moving.any() else np.nan
        summary[f"{v}_pct_moving"] = (100.0 * (err[moving] / gt[moving]).mean()
                                      if moving.any() else np.nan)
        summary[f"{v}_mse_ms"] = (((preds[v] - gt) / 3.6) ** 2).mean()

    # ---- graph ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]})
    sec_t = np.arange(n_seconds) + 0.5

    ax1.plot(sec_t, preds["raw"], color=C_RAW, lw=1.2, label="model raw")
    ax1.plot(sec_t, preds["filtered"], color=C_FILT, lw=2.2, label="model filtered")
    ax1.plot(sec_t, gt, color=C_GT, lw=2.2, ls="--", label="GPS ground truth")
    ax1.set_ylabel("speed (km/h)")
    ax1.set_title(f"Prediction vs GPS ground truth - {name}")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(alpha=0.25)

    mid = starts + interval / 2.0
    w = interval * 0.36
    err_f = [row["filtered_abs_err_kmh"] for row in table]
    err_r = [row["raw_abs_err_kmh"] for row in table]
    ax2.bar(mid - w / 2, err_r, width=w, color=C_RAW, label="raw error")
    ax2.bar(mid + w / 2, err_f, width=w, color=C_FILT, label="filtered error")
    ax2.set_ylabel("abs error (km/h)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    png_path = os.path.join(folder, f"{name}_gt_comparison.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    return summary, csv_path, png_path


def main():
    p = argparse.ArgumentParser(description="Score predictions against the GPS ground truth.")
    p.add_argument("--outputs", default="outputs",
                   help="Folder containing one subfolder per video.")
    p.add_argument("--interval", type=int, default=5,
                   help="Comparison interval in seconds (default 5).")
    args = p.parse_args()

    found = []
    for path in sorted(glob.glob(os.path.join(args.outputs, "*", "*_speeds.csv"))):
        folder = os.path.dirname(path)
        name = os.path.basename(path)[:-len("_speeds.csv")]
        data = load_speeds_csv(path)
        if data is None:
            print(f"skipping {path} (no ground-truth columns - run extract_gt.py first)")
            continue
        summary, csv_path, png_path = analyze(name, folder, data, args.interval)
        found.append(summary)
        print(f"{name}\n  table -> {csv_path}\n  graph -> {png_path}")

    if not found:
        raise SystemExit(f"No comparable *_speeds.csv found under {args.outputs}/*/")

    # ---- printed summary ----
    hdr = f"{'clip':42s} {'variant':9s} {'MAE':>7s} {'MAE mov':>8s} {'%err mov':>9s} {'MSE m/s':>8s}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for s in found:
        for v, _ in VARIANTS:
            print(f"{s['clip']:42s} {v:9s} {s[f'{v}_mae_kmh']:7.2f} "
                  f"{s[f'{v}_mae_moving_kmh']:8.2f} {s[f'{v}_pct_moving']:8.1f}% "
                  f"{s[f'{v}_mse_ms']:8.2f}")
    if len(found) > 1:
        print("-" * len(hdr))
        for v, _ in VARIANTS:
            mae = np.mean([s[f"{v}_mae_kmh"] for s in found])
            maem = np.nanmean([s[f"{v}_mae_moving_kmh"] for s in found])
            pct = np.nanmean([s[f"{v}_pct_moving"] for s in found])
            mse = np.mean([s[f"{v}_mse_ms"] for s in found])
            print(f"{'OVERALL (' + str(len(found)) + ' clips)':42s} {v:9s} "
                  f"{mae:7.2f} {maem:8.2f} {pct:8.1f}% {mse:8.2f}")
    print("=" * len(hdr))
    print("(MAE in km/h; 'mov' = seconds where GT > "
          f"{PCT_MIN_KMH:.0f} km/h; MSE on the comma.ai m/s scale)")


if __name__ == "__main__":
    main()
