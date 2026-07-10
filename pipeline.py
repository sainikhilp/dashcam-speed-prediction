"""
Core pipeline: dashcam video  ->  optical flow  ->  CNN  ->  speed.

This reproduces EXACTLY the preprocessing the bundled pretrained weights were
trained on (Farneback optical flow, HSV encoding, 128x128), and normalizes any
input video to comma.ai conditions (640x480 @ 20 fps) so the pretrained model
gets its fairest possible shot without any fine-tuning.
"""

import cv2
import numpy as np
import torch

from model import Model

# comma.ai training conditions the weights expect.
COMMA_FPS = 20
COMMA_SIZE = (640, 480)   # (width, height)
MODEL_INPUT = (128, 128)
MS_TO_MPH = 2.2369362920544


def load_model(weights_path="weights/Model.pt", device="cpu"):
    """Load the CNN and the pretrained weights."""
    net = Model()
    state = torch.load(weights_path, map_location=torch.device(device))
    net.load_state_dict(state)
    net.eval()
    net.to(device)
    return net


def _flow_image(curr_bgr, prev_bgr):
    """Farneback optical flow encoded as an HSV-in-BGR image (matches training)."""
    gray_c = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    gray_n = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    rgb_flow = np.zeros_like(curr_bgr)
    flow = cv2.calcOpticalFlowFarneback(
        gray_c, gray_n, None, 0.5, 1, 15, 2, 5, 1.3, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    rgb_flow[:, :, 0] = ang * (180 / np.pi / 2)
    rgb_flow[:, :, 2] = (mag * 15).astype(int)
    rgb_flow[:, :, 1] = 255
    return rgb_flow


def _to_tensor(flow_img_128):
    """uint8 HxWxC [0,255] -> float CHW [0,1] with a batch dim (matches ToTensor)."""
    t = flow_img_128.astype(np.float32) / 255.0
    t = np.transpose(t, (2, 0, 1))
    return torch.from_numpy(t).unsqueeze(0)


def _sampled_frames(video_path, target_fps=COMMA_FPS, size=COMMA_SIZE):
    """
    Yield frames resampled to `target_fps` and resized to `size`.

    Uses real presentation timestamps so it works even when the source frame
    rate is variable or different from comma.ai's 20 fps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    dt = 1.0 / target_fps
    next_t = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = t_ms / 1000.0 if t_ms and t_ms > 0 else None
        if t is None:
            # No timestamps available; fall back to keeping every frame.
            yield cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            continue
        if t + 1e-6 >= next_t:
            yield cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            next_t += dt
    cap.release()


def predict_speed(video_path, net, target_fps=COMMA_FPS, device="cpu"):
    """
    Run the full pipeline on a video.

    Returns a list of dicts: {frame, time_s, speed_ms, speed_mph}.
    Speed for frame 0 is copied from frame 1 (there is no flow before frame 0),
    matching the reference implementation.
    """
    speeds = []
    prev = None
    dt = 1.0 / target_fps
    idx = 0
    with torch.no_grad():
        for frame in _sampled_frames(video_path, target_fps, COMMA_SIZE):
            if prev is not None:
                flow = _flow_image(frame, prev)
                flow = cv2.resize(flow, MODEL_INPUT, interpolation=cv2.INTER_AREA)
                out = net(_to_tensor(flow).to(device)).item()
                speeds.append({
                    "frame": idx,
                    "time_s": round(idx * dt, 4),
                    "speed_ms": round(float(out), 4),
                    "speed_mph": round(float(out) * MS_TO_MPH, 4),
                })
                idx += 1
            prev = frame
    if speeds:
        # frame 0 mirrors frame 1
        first = dict(speeds[0])
        first.update(frame=0, time_s=0.0)
        speeds.insert(0, first)
        for i, s in enumerate(speeds):
            s["frame"] = i
            s["time_s"] = round(i * dt, 4)
    return speeds


def moving_average(values, window=5):
    """Simple centered-ish smoothing to reduce per-frame jitter."""
    if window <= 1:
        return list(values)
    v = np.asarray(values, dtype=np.float64)
    pad = window // 2
    # edge-pad so the first/last frames aren't dragged toward zero
    vp = np.pad(v, pad, mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(vp, kernel, mode="same")[pad:pad + len(v)]
    return list(smoothed)
