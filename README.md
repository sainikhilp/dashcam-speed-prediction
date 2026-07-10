# Dashcam Speed Estimator (optical flow + CNN)

This estimates the speed of the vehicle that a **dashcam is mounted on**, straight
from the video. No vanishing points, no camera calibration, no 3D geometry.

How it works: for each pair of consecutive frames we measure how the pixels move
(that's *optical flow*), and a small neural network turns that motion into a speed
number. It's the same idea as the [comma.ai Speed Challenge](https://github.com/commaai/speedchallenge).
Pretrained weights are already included (`weights/Model.pt`), so you can run it
right away without training anything.

## Setup (once)

You need Python 3.9 or newer. Copy-paste the block for your machine.

Windows (PowerShell):
```powershell
cd new_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:
```bash
cd new_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it on a real dashcam video

```bash
python predict_speed.py --video path/to/your_dashcam.mp4
```

You get two files in `outputs/`:

- `<name>_speeds.csv`, the speed at every frame (in m/s and mph)
- `<name>_speed.png`, a speed-over-time chart

### Check accuracy (if you have ground-truth speeds)

If you have a text file with one true speed (m/s) per frame:

```bash
python predict_speed.py --video your_dashcam.mp4 --gt your_speeds.txt
```

It prints an error score on the comma.ai scale: MSE under 10 is acceptable, under
5 is competitive, under 3 is exceptional.

## If the pretrained model isn't accurate enough on your camera

That's expected if your dashcam is very different from comma.ai's (different lens,
mounting, or frame rate). Fine-tune it on some labeled clips from **your** camera.
You can pass one video or several. All of their frames get pooled into a single
model.

```bash
# one video
python finetune.py --video train.mp4 --gt train_speeds.txt --epochs 15

# a few videos trained together into ONE model
python finetune.py --video v1.mp4 v2.mp4 v3.mp4 --gt v1.txt v2.txt v3.txt --epochs 15

# many videos (say 100): list them in a manifest file instead of on the command line
python finetune.py --manifest data/train/train_set.txt --epochs 15

# then test on a held-out video the model never trained on
python predict_speed.py --video test.mp4 --weights weights/finetuned.pt --gt test_speeds.txt
```

A couple of things to keep in mind:

- The Nth `--gt` file has to match the Nth `--video`.
- Don't loop `finetune.py` once per video. Each run overwrites the output and
  starts over from the base model, so you'd only keep the last video. Fine-tune
  **once** over all the videos, which gives you a single `weights/finetuned.pt`,
  then predict with that one file.

### Where to put your videos and the manifest

Keep the training videos, their labels, and the manifest together under `data/`,
and keep a few videos aside in `heldout/` that you never train on so you can
measure real accuracy. A layout that works well:

```
new_pipeline/
├── data/
│   ├── train/
│   │   ├── videos/          your fine-tuning clips
│   │   │   ├── clip001.mp4
│   │   │   └── ...
│   │   ├── labels/          one .txt per clip (speed in m/s, one per frame)
│   │   │   ├── clip001.txt
│   │   │   └── ...
│   │   └── train_set.txt    the manifest (lives here)
│   └── heldout/             videos you do NOT train on, for honest testing
│       ├── videos/test01.mp4
│       └── labels/test01.txt
├── weights/
│   ├── Model.pt             pretrained, shipped with the repo
│   └── finetuned.pt         produced when you fine-tune
├── outputs/                 predictions, plots, flow cache (created automatically)
├── finetune.py
└── predict_speed.py
```

The manifest is one `video , labels` pair per line. Lines starting with `#` are
ignored, and relative paths are resolved against the manifest's own folder, so if
`train_set.txt` sits in `data/train/` the paths stay short:

```
# my 100-clip training set
videos/clip001.mp4 , labels/clip001.txt
videos/clip002.mp4 , labels/clip002.txt
...
```

One note on git: the `.gitignore` already skips `*.mp4`, so the video files stay
on your machine and are not committed. The small `.txt` labels and the manifest
are tracked, which is usually what you want.

## Files

| File | What it is |
|------|------------|
| `predict_speed.py` | main script, run a video and get speeds |
| `pipeline.py` | optical-flow and inference core |
| `model.py` | the CNN definition |
| `weights/Model.pt` | included pretrained weights |
| `finetune.py` | optional, adapt the model to your camera |

## Notes and limits

- Speeds are in m/s. The CSV also gives mph.

## Credits

Model architecture and pretrained weights adapted from Yash Shah's comma.ai
speed-challenge solution: https://github.com/shahyash10/speedchallenge
