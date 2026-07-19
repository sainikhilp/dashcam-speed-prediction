"""
Score model weights against the OCR ground truth for every video in a split.

Walks outputs/<split>/<video>/ folders (as produced by batch_extract_gt.py),
runs the speed model on each video, aggregates predictions and GT to 1 Hz
(the GPS overlay rate), and reports MAE / MSE per clip and overall.

    python eval_split.py --split test  --tag baseline
    python eval_split.py --split both  --tag baseline
    python eval_split.py --split test  --tag finetuned --weights weights/finetuned.pt

Per video (written into its folder):
    <name>_<tag>_pred.csv    second, gt_kmh, pred_kmh (smoothed model output)
    <name>_<tag>_pred.png    GT vs prediction over time
Per run (written to --outputs):
    eval_<tag>_<split>.csv   one row per clip: MAE, moving-MAE, MSE (m/s)

MSE is on the comma.ai m/s scale (<10 acceptable, <5 competitive, <3 exceptional).
"""

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pipeline import load_model, predict_speed, moving_average

PCT_MIN_KMH = 10.0    # a second counts as "moving" above this GT speed


def load_gt_kmh(gt_csv):
    with open(gt_csv, newline="") as f:
        return np.array([float(r["speed_kmh"]) for r in csv.DictReader(f)])


def per_second(t, values, n_seconds):
    sec = np.minimum(np.asarray(t).astype(int), n_seconds - 1)
    v = np.asarray(values)
    return np.array([v[sec == n].mean() for n in range(n_seconds)])


def eval_video(folder, weights, fps, smooth):
    name = os.path.basename(folder)
    video = os.path.join(folder, f"{name}.MP4")
    gt_csv = os.path.join(folder, f"{name}_gt.csv")
    if not (os.path.isfile(video) and os.path.isfile(gt_csv)):
        return None
    net = load_model(weights)
    speeds = predict_speed(video, net, target_fps=fps)
    t = [s["time_s"] for s in speeds]
    ms_smooth = moving_average([s["speed_ms"] for s in speeds], smooth)

    gt = load_gt_kmh(gt_csv)
    n = min(len(gt), int(np.floor(max(t))) + 1)
    pred = per_second(t, ms_smooth, n) * 3.6      # km/h
    gt = gt[:n]

    err = np.abs(pred - gt)
    moving = gt > PCT_MIN_KMH
    return {
        "name": name, "folder": folder, "seconds": n,
        "gt": gt, "pred": pred,
        "mae_kmh": float(err.mean()),
        "mae_moving_kmh": float(err[moving].mean()) if moving.any() else float("nan"),
        "mse_ms": float((((pred - gt) / 3.6) ** 2).mean()),
    }


def plot_video(r, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sec = np.arange(r["seconds"]) + 0.5
    plt.figure(figsize=(11, 4))
    plt.plot(sec, r["gt"], color="#374151", lw=2, ls="--", label="OCR ground truth")
    plt.plot(sec, r["pred"], color="#2563eb", lw=2, label=f"model ({tag})")
    plt.xlabel("time (s)"); plt.ylabel("speed (km/h)")
    plt.title(f"{r['name']}  -  MAE {r['mae_kmh']:.1f} km/h, MSE {r['mse_ms']:.2f} m/s")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(r["folder"], f"{r['name']}_{tag}_pred.png"), dpi=110)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Evaluate weights against OCR GT per split.")
    p.add_argument("--outputs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "..", "outputs"))
    p.add_argument("--split", default="test", choices=["train", "test", "both"])
    p.add_argument("--weights", default="weights/Model.pt")
    p.add_argument("--tag", default="baseline", help="Label used in output filenames.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    folders = []
    for s in splits:
        base = os.path.join(args.outputs, s)
        folders += [(s, os.path.join(base, d)) for d in sorted(os.listdir(base))
                    if os.path.isdir(os.path.join(base, d))]
    print(f"Evaluating {len(folders)} videos ({'+'.join(splits)}) "
          f"with {args.weights} [{args.tag}]", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(eval_video, f, args.weights, args.fps, args.smooth): (s, f)
                for s, f in folders}
        done = 0
        for fut in futs:
            r = fut.result()
            done += 1
            if r is None:
                print(f"[{done}/{len(folders)}] skipped {futs[fut][1]} (missing files)",
                      flush=True)
                continue
            r["split"] = futs[fut][0]
            results.append(r)
            print(f"[{done}/{len(folders)}] {r['split']:<5} {r['name']}  "
                  f"MAE {r['mae_kmh']:5.1f} km/h  MSE {r['mse_ms']:6.2f} m/s", flush=True)

    import pandas as pd
    for r in results:   # matplotlib is not thread-safe -> plot sequentially
        plot_video(r, args.tag)
        # per-second csv next to the plot
        with open(os.path.join(r["folder"], f"{r['name']}_{args.tag}_pred.csv"),
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["second", "gt_kmh", "pred_kmh"])
            for sec in range(r["seconds"]):
                w.writerow([sec, round(r["gt"][sec], 1), round(r["pred"][sec], 2)])
        # add prediction columns to the video's Excel sheet (one row per second)
        df = pd.read_csv(os.path.join(r["folder"], f"{r['name']}_gt.csv"))
        pred = np.full(len(df), np.nan)
        pred[:r["seconds"]] = r["pred"]
        df[f"{args.tag}_pred_kmh"] = np.round(pred, 2)
        df[f"{args.tag}_error_kmh"] = np.round(np.abs(pred - df["speed_kmh"]), 2)
        df.to_excel(os.path.join(r["folder"], f"{r['name']}_gt.xlsx"), index=False)

    summary = os.path.join(args.outputs, f"eval_{args.tag}_{args.split}.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "split", "seconds", "mae_kmh", "mae_moving_kmh", "mse_ms"])
        for r in sorted(results, key=lambda r: r["name"]):
            w.writerow([r["name"], r["split"], r["seconds"], round(r["mae_kmh"], 2),
                        round(r["mae_moving_kmh"], 2), round(r["mse_ms"], 2)])

    print("\n================  OVERALL  ================")
    for s in splits:
        rs = [r for r in results if r["split"] == s]
        if not rs:
            continue
        mse = np.mean([r["mse_ms"] for r in rs])
        bar = ("EXCEPTIONAL" if mse < 3 else "competitive" if mse < 5
               else "acceptable" if mse < 10 else "needs fine-tuning")
        print(f"  {s:<5} ({len(rs)} clips): MAE {np.mean([r['mae_kmh'] for r in rs]):5.2f} km/h"
              f"   moving-MAE {np.nanmean([r['mae_moving_kmh'] for r in rs]):5.2f} km/h"
              f"   MSE {mse:6.2f} m/s  -> {bar}")
    print(f"  summary -> {summary}")
    print("===========================================")


if __name__ == "__main__":
    main()
