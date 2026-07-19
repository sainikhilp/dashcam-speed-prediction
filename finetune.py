"""
OPTIONAL Phase B: fine-tune the model on your own labeled dashcam data.

Only needed if the pretrained weights are not accurate enough on your camera
(check with `predict_speed.py --gt` first). Fine-tuning adapts the model to your
camera's mounting / lens / frame rate.

Inputs (one or more videos, each paired with its own labels):
  --video   one or more dashcam videos
  --gt      one text file per video, each with ONE ground-truth speed (m/s) per
            frame of that video (the comma.ai label format). The Nth --gt file
            corresponds to the Nth --video.

    # single video
    python finetune.py --video train.mp4 --gt train.txt --epochs 15

    # a handful of videos trained together into ONE model
    python finetune.py --video v1.mp4 v2.mp4 ... v10.mp4 \
                       --gt    v1.txt  v2.txt  ... v10.txt  --epochs 15

    # MANY videos (e.g. 100): list them in a manifest file instead of on the CLI
    python finetune.py --manifest train_set.txt --epochs 15
    # -> saves weights/finetuned.pt (a single model that has seen ALL videos)
    python predict_speed.py --video heldout.mp4 --weights weights/finetuned.pt

Manifest format (one 'video , labels' pair per line; '#' comments allowed;
relative paths are resolved against the manifest file's own directory):
    videos/clip001.mp4 , labels/clip001.txt
    videos/clip002.mp4 , labels/clip002.txt
    ...

Frames from every video are pooled, then split once into train/validation, so
the model learns from all of them at once. Ground-truth speeds are per ORIGINAL
frame, so this script does NOT resample - it processes every frame to keep
labels aligned.
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from model import Model
from pipeline import _flow_image, _to_tensor, COMMA_SIZE, MODEL_INPUT


def build_flow_cache(video_path, gt, cache_dir, stride=1):
    """Compute a 128x128 flow image per used frame pair; return (paths, labels).

    With stride > 1 only every stride-th frame is used, so the flow magnitude
    matches a lower effective frame rate (e.g. a 60 fps dashcam with stride 3
    gives 20 fps flow - the conditions the pretrained weights and
    predict_speed.py's default resampling expect). Labels stay aligned because
    they are indexed by ORIGINAL frame number.

    Writes into its own `cache_dir` (one per video) so frames from different
    videos never overwrite each other by index.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"Could not read: {video_path}")
    prev = cv2.resize(prev, COMMA_SIZE, interpolation=cv2.INTER_AREA)
    paths, labels = [], []
    i = 0
    while True:
        ok, curr = cap.read()
        if not ok:
            break
        i += 1
        if i % stride:
            continue
        curr = cv2.resize(curr, COMMA_SIZE, interpolation=cv2.INTER_AREA)
        flow = _flow_image(curr, prev)
        flow = cv2.resize(flow, MODEL_INPUT, interpolation=cv2.INTER_AREA)
        path = os.path.join(cache_dir, f"{i}.png")
        cv2.imwrite(path, flow)
        paths.append(path)
        labels.append(float(gt[i]) if i < len(gt) else float(gt[-1]))
        prev = curr
    cap.release()
    return paths, labels


def read_manifest(manifest_path):
    """Parse a manifest into a list of (video, labels) pairs.

    One pair per line, `video , labels` (comma OR whitespace separated). Blank
    lines and lines starting with '#' are ignored. Relative paths are resolved
    against the manifest file's own directory, so the manifest is portable.
    """
    base = os.path.dirname(os.path.abspath(manifest_path))
    pairs = []
    with open(manifest_path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [c.strip() for c in (line.split(",") if "," in line else line.split())]
            if len(parts) != 2:
                raise SystemExit(
                    f"{manifest_path}:{lineno}: expected 'video , labels', got: {raw.rstrip()}")
            video, gt = (p if os.path.isabs(p) else os.path.join(base, p) for p in parts)
            pairs.append((video, gt))
    if not pairs:
        raise SystemExit(f"No video/labels pairs found in {manifest_path}")
    return pairs


def main():
    p = argparse.ArgumentParser(description="Fine-tune the speed model on your data.")
    p.add_argument("--video", nargs="+",
                   help="One or more dashcam videos. (Use --manifest instead for many videos.)")
    p.add_argument("--gt", nargs="+",
                   help="One labels file per video (one speed in m/s per frame), same order as --video.")
    p.add_argument("--manifest",
                   help="Text file of 'video , labels' pairs, one per line "
                        "(the scalable alternative to --video/--gt for many videos).")
    p.add_argument("--base", default="weights/Model.pt", help="Weights to start from ('' = from scratch).")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val", type=float, default=0.2, help="Validation fraction.")
    p.add_argument("--out", default="weights/finetuned.pt")
    p.add_argument("--cache", default="outputs/flow_cache")
    p.add_argument("--stride", type=int, default=1,
                   help="Use every Nth frame (3 for a 60 fps camera -> 20 fps flow, "
                        "matching the pretrained weights and predict_speed.py).")
    args = p.parse_args()

    # Resolve the list of (video, labels) pairs from either source.
    if args.manifest:
        if args.video or args.gt:
            raise SystemExit("Use either --manifest OR --video/--gt, not both.")
        pairs = read_manifest(args.manifest)
    else:
        if not args.video or not args.gt:
            raise SystemExit("Provide --video and --gt, or a --manifest file.")
        if len(args.video) != len(args.gt):
            raise SystemExit(
                f"Got {len(args.video)} --video but {len(args.gt)} --gt files; "
                "each video needs exactly one matching labels file, in the same order.")
        pairs = list(zip(args.video, args.gt))

    print(f"Fine-tuning on {len(pairs)} video(s).")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pool frames from every video into one training set.
    paths, labels = [], []
    for vi, (video_path, gt_path) in enumerate(pairs):
        if not os.path.isfile(video_path):
            raise SystemExit(f"Video not found: {video_path}")
        if not os.path.isfile(gt_path):
            raise SystemExit(f"Labels not found: {gt_path}")
        gt = np.loadtxt(gt_path).ravel()
        cache_dir = os.path.join(args.cache, f"vid{vi:03d}")
        cached = sorted(glob.glob(os.path.join(cache_dir, "*.png")),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        if cached:   # reuse a cache from a previous run (delete the dir to rebuild)
            idxs = [int(os.path.splitext(os.path.basename(c))[0]) for c in cached]
            vpaths = cached
            vlabels = [float(gt[i]) if i < len(gt) else float(gt[-1]) for i in idxs]
            print(f"[{vi + 1}/{len(pairs)}] Reusing {len(cached)} cached flow frames "
                  f"for {video_path}", flush=True)
        else:
            print(f"[{vi + 1}/{len(pairs)}] Building optical-flow cache from {video_path} ...",
                  flush=True)
            vpaths, vlabels = build_flow_cache(video_path, gt, cache_dir, args.stride)
            print(f"      {len(vpaths)} flow frames cached.", flush=True)
        paths.extend(vpaths)
        labels.extend(vlabels)

    labels = np.asarray(labels, dtype=np.float32)
    n = len(paths)
    print(f"Total: {n} flow frames from {len(pairs)} video(s).")

    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    n_val = int(n * args.val)
    val_idx, train_idx = set(idx[:n_val].tolist()), idx[n_val:]

    net = Model().to(device)
    if args.base and os.path.isfile(args.base):
        net.load_state_dict(torch.load(args.base, map_location=device))
        print(f"  started from {args.base}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def load_batch(indices):
        xs, ys = [], []
        for j in indices:
            img = cv2.imread(paths[j])
            for _ in range(3):           # transient read failures (AV file locks)
                if img is not None:
                    break
                time.sleep(0.2)
                img = cv2.imread(paths[j])
            if img is None:
                print(f"  warning: skipping unreadable cache file {paths[j]}", flush=True)
                continue
            xs.append(_to_tensor(img))
            ys.append([labels[j]])
        if not xs:
            return None, None
        return (torch.cat(xs).to(device),
                torch.tensor(ys, dtype=torch.float32).to(device))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(1, args.epochs + 1):
        net.train()
        np.random.shuffle(train_idx)
        tot, n_tot = 0.0, 0
        for b in range(0, len(train_idx), args.batch):
            x, y = load_batch(train_idx[b:b + args.batch])
            if x is None:
                continue
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * y.shape[0]
            n_tot += y.shape[0]
        # validation
        net.eval()
        with torch.no_grad():
            vi = np.array(sorted(val_idx))
            vloss, n_val_used = 0.0, 0
            for b in range(0, len(vi), args.batch):
                x, y = load_batch(vi[b:b + args.batch])
                if x is None:
                    continue
                vloss += loss_fn(net(x), y).item() * y.shape[0]
                n_val_used += y.shape[0]
            vloss = vloss / n_val_used if n_val_used else float("nan")
        # checkpoint every epoch so a crash never loses the training done so far
        torch.save(net.state_dict(), args.out)
        print(f"  epoch {ep:2d}/{args.epochs}  train_mse={tot/max(n_tot,1):6.3f}  "
              f"val_mse={vloss:6.3f}  (saved -> {args.out})", flush=True)

    print(f"\nSaved fine-tuned weights -> {args.out}")


if __name__ == "__main__":
    main()
