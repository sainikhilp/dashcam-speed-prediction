"""
Command-line entry point.

Basic use:
    python predict_speed.py --video mydashcam.mp4

With ground-truth speeds (one value per frame, comma.ai format) to score accuracy:
    python predict_speed.py --video mydashcam.mp4 --gt speeds.txt

Outputs (written to --outdir, default ./outputs):
    <name>_speeds.csv   frame, time_s, speed_ms, speed_mph, speed_ms_smooth
    <name>_speed.png    predicted speed over time (and ground truth if given)
"""

import argparse
import csv
import os

import numpy as np

from pipeline import load_model, predict_speed, moving_average, MS_TO_MPH


def main():
    p = argparse.ArgumentParser(description="Dashcam ego-speed from video (optical flow + CNN).")
    p.add_argument("--video", required=True, help="Path to the dashcam video file.")
    p.add_argument("--weights", default="weights/Model.pt", help="Path to model weights.")
    p.add_argument("--fps", type=int, default=20, help="Target frame rate to resample to (comma.ai used 20).")
    p.add_argument("--gt", default=None, help="Optional ground-truth speeds file (one value per frame, m/s).")
    p.add_argument("--outdir", default="outputs", help="Where to write results.")
    p.add_argument("--smooth", type=int, default=5, help="Moving-average window for smoothing (1 = off).")
    args = p.parse_args()

    if not os.path.isfile(args.video):
        raise SystemExit(f"Video not found: {args.video}")
    if not os.path.isfile(args.weights):
        raise SystemExit(f"Weights not found: {args.weights}")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[1/3] Loading model  ({args.weights})")
    net = load_model(args.weights)

    print(f"[2/3] Running pipeline on {args.video}  (resampling to {args.fps} fps)")
    speeds = predict_speed(args.video, net, target_fps=args.fps)
    if not speeds:
        raise SystemExit("No frames processed - is the video valid?")

    ms = [s["speed_ms"] for s in speeds]
    ms_smooth = moving_average(ms, args.smooth)

    name = os.path.splitext(os.path.basename(args.video))[0]
    csv_path = os.path.join(args.outdir, f"{name}_speeds.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "speed_ms", "speed_mph", "speed_ms_smooth"])
        for s, sm in zip(speeds, ms_smooth):
            w.writerow([s["frame"], s["time_s"], s["speed_ms"], s["speed_mph"], round(sm, 4)])

    # ---- optional accuracy scoring ----
    gt = None
    if args.gt:
        gt = np.loadtxt(args.gt).ravel()
        n = min(len(gt), len(ms_smooth))
        pred = np.asarray(ms_smooth[:n])
        gtn = gt[:n]
        mse = float(np.mean((pred - gtn) ** 2))
        mae = float(np.mean(np.abs(pred - gtn)))
        bar = ("EXCEPTIONAL" if mse < 3 else "competitive" if mse < 5
               else "acceptable" if mse < 10 else "needs fine-tuning")
        print(f"      Accuracy vs ground truth:  MSE={mse:.2f}  MAE={mae:.2f} m/s  ->  {bar}")

    # ---- plot ----
    print(f"[3/3] Writing results to {args.outdir}/")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = [s["time_s"] for s in speeds]
        plt.figure(figsize=(11, 4))
        plt.plot(t, ms, color="#9ec5fe", lw=1, label="raw")
        plt.plot(t, ms_smooth, color="#1f6feb", lw=2, label=f"smoothed (w={args.smooth})")
        if gt is not None:
            plt.plot(t[:len(gt)], gt[:len(t)], color="#d1242f", lw=1.5, ls="--", label="ground truth")
        plt.xlabel("time (s)"); plt.ylabel("speed (m/s)")
        plt.title(f"Estimated ego-speed - {name}")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        png_path = os.path.join(args.outdir, f"{name}_speed.png")
        plt.savefig(png_path, dpi=120)
        print(f"      plot -> {png_path}")
    except Exception as e:
        print(f"      (skipped plot: {e})")

    mean_ms = float(np.mean(ms_smooth))
    print("\n==================  RESULT  ==================")
    print(f"  frames processed : {len(speeds)}")
    print(f"  mean speed       : {mean_ms:6.2f} m/s   ({mean_ms * MS_TO_MPH:6.2f} mph)")
    print(f"  min / max        : {min(ms_smooth):6.2f} / {max(ms_smooth):6.2f} m/s")
    print(f"  full results     : {csv_path}")
    print("=============================================")


if __name__ == "__main__":
    main()
