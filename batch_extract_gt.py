"""
Batch ground-truth extraction over every dashcam video, with a train/test split.

Split strategy: clips are grouped into DRIVE SESSIONS (consecutive clips whose
start times are <= 3 min apart belong to the same drive). Within each session
the last ~20% of clips go to test, the rest to train. This avoids leakage:
a random split would put adjacent minutes of the same road in both sets.
Single-clip sessions go to train.

Output layout (under --outdir, default ../outputs):
    train/<video name>/   <video>.MP4, <name>_gt.csv, <name>_gt.xlsx, <name>_gt_ms.txt
    test/<video name>/    same
    split_manifest.csv    one row per video: split, session, seconds, speed stats
    split_manifest.xlsx

Resumable: videos whose gt files already exist are skipped, so re-running
after an interruption only processes what is missing.

Use:
    python batch_extract_gt.py                 # from dashcam-speed-prediction/
    python batch_extract_gt.py --workers 2     # fewer parallel extractions
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(HERE, "extract_gt.py")

SESSION_GAP_S = 180      # new drive session if start-to-start gap exceeds this
TEST_FRAC = 0.2          # share of each session's clips held out for test


def start_time(path):
    m = re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})", os.path.basename(path))
    if not m:
        raise SystemExit(f"Cannot parse start time from {path}")
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s)


def split_videos(video_paths):
    """Group into drive sessions, hold out the tail of each session for test."""
    vids = sorted(video_paths, key=start_time)
    sessions, cur = [], [vids[0]]
    for prev, v in zip(vids, vids[1:]):
        if (start_time(v) - start_time(prev)).total_seconds() > SESSION_GAP_S:
            sessions.append(cur)
            cur = [v]
        else:
            cur.append(v)
    sessions.append(cur)

    assign = {}   # path -> (split, session_id)
    for sid, sess in enumerate(sessions):
        n_test = int(round(TEST_FRAC * len(sess))) if len(sess) > 1 else 0
        for v in sess[:len(sess) - n_test]:
            assign[v] = ("train", sid)
        for v in sess[len(sess) - n_test:]:
            assign[v] = ("test", sid)
    return assign, sessions


def process_one(video, split, outbase):
    """Run extract_gt.py, then copy the video in and add an .xlsx of the csv."""
    name = os.path.splitext(os.path.basename(video))[0]
    outdir = os.path.join(outbase, split, name)
    csv_path = os.path.join(outdir, f"{name}_gt.csv")
    txt_path = os.path.join(outdir, f"{name}_gt_ms.txt")

    if not (os.path.isfile(csv_path) and os.path.isfile(txt_path)):
        r = subprocess.run(
            [sys.executable, EXTRACT, "--video", video, "--outdir", outdir,
             "--template-cache", os.path.join(outbase, "digit_templates.npz")],
            capture_output=True, text=True)
        if r.returncode != 0:
            return name, split, None, (r.stderr or r.stdout).strip().splitlines()[-1]

    dst_video = os.path.join(outdir, os.path.basename(video))
    if not os.path.isfile(dst_video):
        shutil.copy2(video, dst_video)

    df = pd.read_csv(csv_path)
    df.to_excel(os.path.join(outdir, f"{name}_gt.xlsx"), index=False)

    kmh = df["speed_kmh"].to_numpy()
    stats = dict(seconds=len(df), mean_kmh=round(float(np.mean(kmh)), 2),
                 min_kmh=round(float(np.min(kmh)), 2), max_kmh=round(float(np.max(kmh)), 2),
                 ocr_misses=int((df["source"] != "ocr").sum()))
    return name, split, stats, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default=os.path.join(HERE, "..", "Dashcam Videos"))
    p.add_argument("--outdir", default=os.path.join(HERE, "..", "outputs"))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    videos = [os.path.join(args.videos, f) for f in sorted(os.listdir(args.videos))
              if f.upper().endswith(".MP4")]
    if not videos:
        raise SystemExit(f"No videos found in {args.videos}")
    assign, sessions = split_videos(videos)
    n_test = sum(1 for s, _ in assign.values() if s == "test")
    print(f"{len(videos)} videos, {len(sessions)} drive sessions "
          f"-> {len(videos) - n_test} train / {n_test} test\n", flush=True)

    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, v, assign[v][0], args.outdir): v for v in videos}
        for i, fut in enumerate(as_completed(futs), 1):
            name, split, stats, err = fut.result()
            if err:
                failures.append((futs[fut], err))
                print(f"[{i}/{len(videos)}] FAIL  {name}: {err}", flush=True)
            else:
                sid = assign[futs[fut]][1]
                rows.append({"video": name, "split": split, "session": sid, **stats})
                print(f"[{i}/{len(videos)}] {split:<5} {name}  "
                      f"mean {stats['mean_kmh']} km/h", flush=True)

    # retry pass: clips too short to learn templates succeed once a longer
    # clip has populated the shared template cache
    retry, failures = failures, []
    if retry:
        print(f"\nRetrying {len(retry)} failed video(s) with the template cache...",
              flush=True)
    for video, _ in retry:
        name, split, stats, err = process_one(video, assign[video][0], args.outdir)
        if err:
            failures.append((name, err))
            print(f"RETRY FAIL  {name}: {err}", flush=True)
        else:
            rows.append({"video": name, "split": split,
                         "session": assign[video][1], **stats})
            print(f"RETRY OK    {split:<5} {name}  mean {stats['mean_kmh']} km/h",
                  flush=True)

    rows.sort(key=lambda r: r["video"])
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(args.outdir, "split_manifest.csv"), index=False)
    man.to_excel(os.path.join(args.outdir, "split_manifest.xlsx"), index=False)
    print(f"\nDone: {len(rows)} ok, {len(failures)} failed. "
          f"Manifest: {os.path.join(args.outdir, 'split_manifest.csv')}")
    for name, err in failures:
        print(f"  FAILED {name}: {err}")


if __name__ == "__main__":
    main()
