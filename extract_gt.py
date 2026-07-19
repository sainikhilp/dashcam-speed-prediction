"""
Extract ground-truth ego-speed from a 70mai dashcam's burned-in GPS overlay.

No OCR engine needed: the overlay uses a fixed bitmap font, and the timestamp
text is fully predictable (the start time is encoded in the filename and the
clock ticks one second per second of video). So the script first LEARNS the
ten digit shapes from the timestamp, then reads the speed field ("<n>km/h")
by template matching. On a fixed font this is essentially exact.

Basic use:
    python extract_gt.py --video ../dashcam_videos/NO20260508-132543-004975F.MP4

Outputs (written to <outdir>/<video name>/):
    <name>_gt.csv        per second: clock time, OCR'd km/h, cleaned km/h, m/s
    <name>_gt_ms.txt     one speed (m/s) per ORIGINAL video frame - the format
                         finetune.py and predict_speed.py --gt expect
    gt_debug/            (with --debug) speed-field crops named with their
                         reading, for visual spot checks
"""

import argparse
import csv
import os
import re
from datetime import datetime, timedelta

import cv2
import numpy as np

# 70mai 1080p overlay layout (full-frame pixel coordinates).
TEXT_Y = (1005, 1075)     # horizontal band containing the overlay text
TS_X = (0, 620)           # timestamp "YYYY-MM-DD HH:MM:SS"
SPEED_X = (880, 1026)     # the speed DIGITS only; "km/h" always starts at x=1030
CROP_SIZE = (24, 36)      # (w, h) all glyphs are normalized to before matching
MATCH_MARGIN = 0.03       # best digit must beat the runner-up by this much
VOTES_PER_SEC = 5         # GPS speed updates at 1 Hz, so several frames within
                          # the same second must agree -> majority vote
MAX_JUMP_KMH = 25         # per-second change beyond this is treated as an OCR miss
MAX_CHAR_GAP = 20         # px; real inter-digit gaps are <=14, glare blobs sit further
DIGIT_W = (10, 30)        # px; plausible digit widths ('1' is ~12, others ~20)


def start_time_from_name(video_path):
    """70mai filenames look like NO20260508-132543-004975F.MP4."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})", os.path.basename(video_path))
    if not m:
        raise SystemExit(
            "Could not parse the recording start time from the filename; "
            "pass it explicitly with --start-time 'YYYY-MM-DD HH:MM:SS'.")
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s)


def _binary(region_bgr):
    """White overlay text -> 1, everything else -> 0."""
    return (np.min(region_bgr, axis=2) > 180).astype(np.uint8)


def _char_boxes(bin_img, min_w=3, merge_gap=1):
    """Split a binary text line into per-character column ranges."""
    cols = bin_img.sum(axis=0)
    runs, start = [], None
    for i, c in enumerate(cols):
        if c > 0 and start is None:
            start = i
        elif c == 0 and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(cols)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_w]


def _norm(img):
    """Zero-mean, unit-norm float image (so a dot product = NCC)."""
    f = img.astype(np.float32)
    f -= f.mean()
    n = np.linalg.norm(f)
    return f / n if n > 0 else None


def _norm_crop(gray, bin_img, x0, x1):
    """Cut one glyph, tighten its rows, normalize size and contrast.

    Returns (gray_feature, binary_feature); matching on both makes the score
    robust to bright road showing through the semi-transparent overlay.
    """
    rows = np.where(bin_img[:, x0:x1].sum(axis=1) > 0)[0]
    if len(rows) == 0:
        return None
    g = cv2.resize(gray[rows[0]:rows[-1] + 1, x0:x1].astype(np.float32),
                   CROP_SIZE, interpolation=cv2.INTER_AREA)
    b = cv2.resize(bin_img[rows[0]:rows[-1] + 1, x0:x1].astype(np.float32),
                   CROP_SIZE, interpolation=cv2.INTER_AREA)
    g, b = _norm(g), _norm(b)
    return None if g is None or b is None else (g, b)


def _score(crop, tmpl):
    """Combined NCC over the gray and binary features."""
    return 0.5 * float(np.sum(crop[0] * tmpl[0])) + 0.5 * float(np.sum(crop[1] * tmpl[1]))


def _grab(cap, t):
    """Frame at video time t seconds (None past the end)."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def _median_template(crops):
    """Per-pixel median of several (gray, binary) samples -> one clean template."""
    return (_norm(np.median(np.stack([c[0] for c in crops]), axis=0)),
            _norm(np.median(np.stack([c[1] for c in crops]), axis=0)))


def learn_digit_templates(cap, start_time, n_seconds, samples_per_digit=6):
    """Learn the ten digit glyphs from the overlay itself, in two stages.

    Stage 1: the date+hour part of the timestamp ("2026-05-08 13") is CONSTANT
    for the whole clip, so those glyphs can be labeled with total certainty.
    (The overlay clock may be +/-1 s off the filename-derived time, so glyphs
    in the CHANGING part of the timestamp must never be trusted for labels.)

    Stage 2: the digits the constant part doesn't contain are learned from the
    seconds-units position, which cycles 0..9: whatever that position shows
    one second after a confident '3' is by definition a '4', and so on.
    Candidates must be mutually consistent before they become a template.
    """
    UNITS = 17                       # seconds-units glyph index (of 18 chars)
    per_sec = {}                     # sec -> list of 18 normalized crops
    for sec in range(n_seconds):
        frame = _grab(cap, sec + 0.5)
        if frame is None:
            continue
        band = frame[TEXT_Y[0]:TEXT_Y[1], TS_X[0]:TS_X[1]]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        binimg = _binary(band)
        boxes = _char_boxes(binimg)
        if len(boxes) != 18:         # scene glare broke segmentation; skip
            continue
        crops = [_norm_crop(gray, binimg, x0, x1) for x0, x1 in boxes]
        # separators ('-', ':') have zero binary variance and come back None;
        # only the digit positions we actually harvest need to be valid
        if crops[UNITS] is not None:
            per_sec[sec] = crops
    if len(per_sec) < 10:
        raise SystemExit("Could not segment the timestamp reliably "
                         "(is the overlay layout different on this camera?)")

    # ---- stage 1: constant chars (date, and hour if it doesn't roll over) ----
    end_time = start_time + timedelta(seconds=n_seconds + 2)
    n_const = 12 if end_time.strftime("%Y%m%d%H") == start_time.strftime("%Y%m%d%H") else 10
    if end_time.date() != start_time.date():
        raise SystemExit("Clip crosses midnight; constant-part learning not supported.")
    const_chars = start_time.strftime("%Y-%m-%d%H")[:n_const]
    samples = {}
    for sec in sorted(per_sec):
        for i in range(n_const):
            ch = const_chars[i]
            if (ch.isdigit() and per_sec[sec][i] is not None
                    and len(samples.setdefault(ch, [])) < samples_per_digit):
                samples[ch].append(per_sec[sec][i])
    templates = {d: _median_template(v) for d, v in samples.items() if v}

    # ---- stage 2: chain through the 0..9 cycle for the missing digits ----
    def classify(crop):
        sc = sorted(((_score(crop, t), d) for d, t in templates.items()), reverse=True)
        return sc[0][1], sc[0][0], (sc[0][0] - sc[1][0] if len(sc) > 1 else 1.0)

    for _ in range(10):
        missing = [str(d) for d in range(10) if str(d) not in templates]
        if not missing:
            break
        progressed = False
        for m in missing:
            anchor = str((int(m) - 1) % 10)
            if anchor not in templates:
                continue
            cands = []
            for sec in sorted(per_sec):
                if sec + 1 not in per_sec:
                    continue
                d, s, margin = classify(per_sec[sec][UNITS])
                if d == anchor and s > 0.9 and margin > 0.05:
                    cand = per_sec[sec + 1][UNITS]
                    if classify(cand)[1] < 0.9:      # not a digit we already know
                        cands.append(cand)
            # keep only candidates that agree with most other candidates
            good = [c for c in cands
                    if sum(float(np.sum(c[1] * o[1])) > 0.9 for o in cands) > len(cands) / 2]
            if len(good) >= 2:
                templates[m] = _median_template(good[:samples_per_digit])
                progressed = True
        if not progressed:
            break
    missing = [str(d) for d in range(10) if str(d) not in templates]
    if missing:
        raise SystemExit(f"Could not learn digit templates for: {missing} "
                         "(clip may be too short for the clock cycle to cover them).")
    return templates


def save_templates(path, templates):
    """Cache learned templates so short clips can reuse them (atomic write)."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{f"g{d}": t[0] for d, t in templates.items()},
             **{f"b{d}": t[1] for d, t in templates.items()})
    os.replace(tmp, path)


def load_templates(path):
    z = np.load(path)
    return {str(d): (z[f"g{d}"], z[f"b{d}"]) for d in range(10)}


def verify_templates(cap, templates, start_time, n_seconds, min_acc=0.9):
    """Check cached templates against this clip's KNOWN date digits."""
    const = start_time.strftime("%Y-%m-%d")
    digit_pos = [i for i, ch in enumerate(const) if ch.isdigit()]
    total = correct = 0
    for sec in range(min(n_seconds, 10)):
        frame = _grab(cap, sec + 0.5)
        if frame is None:
            continue
        band = frame[TEXT_Y[0]:TEXT_Y[1], TS_X[0]:TS_X[1]]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        binimg = _binary(band)
        boxes = _char_boxes(binimg)
        if len(boxes) != 18:
            continue
        for i in digit_pos:
            crop = _norm_crop(gray, binimg, *boxes[i])
            if crop is None:
                continue
            best = max(templates, key=lambda d: _score(crop, templates[d]))
            total += 1
            correct += best == const[i]
    return total >= 8 and correct / total >= min_acc


def read_speed_once(frame, templates):
    """Read the speed digits from ONE frame. Returns int km/h or None.

    The crop contains only the digits (km/h is outside SPEED_X), so every
    segmented box must classify confidently or the whole read is rejected -
    a rejected read just loses one vote out of VOTES_PER_SEC.
    """
    band = frame[TEXT_Y[0]:TEXT_Y[1], SPEED_X[0]:SPEED_X[1]]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    binimg = _binary(band)
    boxes = _char_boxes(binimg)
    # keep only the right-aligned digit group: bright scenery bleeding through
    # the overlay can leave blobs LEFT of the digits, separated by a wide gap
    kept = []
    for box in reversed(boxes):
        if kept and kept[-1][0] - box[1] > MAX_CHAR_GAP:
            break
        kept.append(box)
    boxes = kept[::-1]
    if not boxes or len(boxes) > 3:      # speed is always 1-3 digits
        return None
    if any(not (DIGIT_W[0] <= x1 - x0 <= DIGIT_W[1]) for x0, x1 in boxes):
        return None                      # a blob that size is not a digit
    digits = ""
    for x0, x1 in boxes:
        crop = _norm_crop(gray, binimg, x0, x1)
        if crop is None:
            return None
        scored = sorted(((_score(crop, t), d) for d, t in templates.items()), reverse=True)
        if scored[0][0] - scored[1][0] < MATCH_MARGIN:   # ambiguous glyph
            return None
        digits += scored[0][1]
    return int(digits)


def read_speed(cap, sec, templates, debug_dir=None):
    """Majority vote over several frames within one second (GPS is 1 Hz)."""
    votes = {}
    for k in range(VOTES_PER_SEC):
        t = sec + (k + 1.0) / (VOTES_PER_SEC + 1.0)
        frame = _grab(cap, t)
        if frame is None:
            continue
        v = read_speed_once(frame, templates)
        if v is not None:
            votes[v] = votes.get(v, 0) + 1
        if debug_dir is not None and k == 2:
            band = frame[TEXT_Y[0]:TEXT_Y[1], SPEED_X[0] - 20:1180]
            cv2.imwrite(os.path.join(debug_dir, f"sec{sec:03d}_read_{v}.png"), band)
    if not votes:
        return None
    value, count = max(votes.items(), key=lambda kv: kv[1])
    return value if count >= 2 else None     # a single lone vote is not enough


def clean_readings(vals):
    """Interpolate misses and physically impossible jumps. Returns (clean, sources)."""
    v = np.array([np.nan if x is None else float(x) for x in vals])
    src = ["ocr"] * len(v)
    for i in range(len(v)):              # spike = disagrees with BOTH neighbours
        if i > 0 and i < len(v) - 1 and not np.isnan(v[i - 1: i + 2]).any():
            if (abs(v[i] - v[i - 1]) > MAX_JUMP_KMH
                    and abs(v[i] - v[i + 1]) > MAX_JUMP_KMH
                    and abs(v[i + 1] - v[i - 1]) <= MAX_JUMP_KMH):
                v[i] = np.nan
                src[i] = "fixed"
    if np.isnan(v).all():
        raise SystemExit("No speed readings succeeded - check the overlay region.")
    idx = np.arange(len(v))
    good = ~np.isnan(v)
    v[~good] = np.interp(idx[~good], idx[good], v[good])
    for i in np.where(~good)[0]:
        if src[i] == "ocr":
            src[i] = "interp"
    return v, src


def main():
    p = argparse.ArgumentParser(description="Ground-truth speed from the 70mai GPS overlay.")
    p.add_argument("--video", required=True, help="Path to the dashcam video file.")
    p.add_argument("--outdir", default="outputs",
                   help="Base results folder; each video gets its own subfolder in it.")
    p.add_argument("--start-time", default=None,
                   help="Recording start 'YYYY-MM-DD HH:MM:SS' (default: parsed from filename).")
    p.add_argument("--debug", action="store_true",
                   help="Also save every speed-field crop, named with its reading.")
    p.add_argument("--template-cache", default=None,
                   help="Path to a .npz digit-template cache. Written after a "
                        "successful learn; used as a verified fallback when a clip "
                        "is too short to learn all ten digits itself.")
    args = p.parse_args()

    if not os.path.isfile(args.video):
        raise SystemExit(f"Video not found: {args.video}")
    name = os.path.splitext(os.path.basename(args.video))[0]
    outdir = args.outdir
    if os.path.basename(os.path.normpath(outdir)) != name:
        outdir = os.path.join(outdir, name)
    os.makedirs(outdir, exist_ok=True)
    debug_dir = None
    if args.debug:
        debug_dir = os.path.join(outdir, "gt_debug")
        os.makedirs(debug_dir, exist_ok=True)

    start_time = (datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
                  if args.start_time else start_time_from_name(args.video))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_seconds = max(1, int(np.floor(n_frames / fps)))
    print(f"[1/3] Learning digit templates from the timestamp ({name}, start {start_time})")
    # Prefer verified cached templates: they come from a full 0-9 clock cycle on
    # a long clip, whereas a short clip can mislabel digits it never saw and
    # still read the speed field "confidently" (but wrong).
    templates = None
    if args.template_cache and os.path.isfile(args.template_cache):
        cached = load_templates(args.template_cache)
        if verify_templates(cap, cached, start_time, n_seconds):
            print("      using cached digit templates (verified on this clip)")
            templates = cached
        else:
            print("      cached templates failed verification; learning fresh")
    if templates is None:
        templates = learn_digit_templates(cap, start_time, n_seconds)
        if args.template_cache and not os.path.isfile(args.template_cache):
            save_templates(args.template_cache, templates)

    print(f"[2/3] Reading the speed field ({n_seconds} s, "
          f"{VOTES_PER_SEC}-frame majority vote per second)")
    readings = [read_speed(cap, sec, templates, debug_dir) for sec in range(n_seconds)]
    cap.release()
    n_miss = sum(r is None for r in readings)
    clean, src = clean_readings(readings)
    n_fixed = sum(s == "fixed" for s in src)
    print(f"      {len(readings) - n_miss}/{len(readings)} frames read"
          f" ({n_miss} interpolated, {n_fixed} spike-fixed)")

    print(f"[3/3] Writing results to {outdir}/")
    csv_path = os.path.join(outdir, f"{name}_gt.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["second", "clock", "speed_kmh_ocr", "speed_kmh", "speed_ms", "source"])
        for sec, (raw, val, s) in enumerate(zip(readings, clean, src)):
            clock = (start_time + timedelta(seconds=sec)).strftime("%H:%M:%S")
            w.writerow([sec, clock, "" if raw is None else raw,
                        round(val, 1), round(val / 3.6, 4), s])

    # one label per ORIGINAL frame (readings sit at second midpoints), m/s
    sec_t = np.arange(n_seconds) + 0.5
    frame_t = np.arange(n_frames) / fps
    per_frame_ms = np.interp(frame_t, sec_t, clean) / 3.6
    txt_path = os.path.join(outdir, f"{name}_gt_ms.txt")
    np.savetxt(txt_path, per_frame_ms, fmt="%.4f")

    kmh = clean
    print("\n==============  GROUND TRUTH  ==============")
    print(f"  seconds read     : {n_seconds}   ({n_frames} frame labels at {fps:.0f} fps)")
    print(f"  mean speed       : {kmh.mean():6.2f} km/h")
    print(f"  min / max        : {kmh.min():6.2f} / {kmh.max():6.2f} km/h")
    print(f"  per-second csv   : {csv_path}")
    print(f"  per-frame labels : {txt_path}")
    print("============================================")


if __name__ == "__main__":
    main()
