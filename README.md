# Final Project: Does Predicted Depth Help Separate Shading from Reflectance?

**Team:** Geethanjali Ravichandra, Bhavya Teja Vejandla
**Course:** CS 7170 Advanced Perception

## What this is

In Project 2 we split an image into reflectance (surface color) and shading (lighting) by labeling
every image derivative as a reflectance edge or a shading edge (Tappen, Freeman and Adelson, NIPS 2002).
Our best rule, the color cue, fails when brightness changes but the color doesn't, like white print
on the tea bags.

For the final project we add a geometry cue from Depth Anything 3. The starting rule:

- color changes across the edge -> reflectance (same as Project 2)
- else, strong edge but the predicted surface is smooth there -> reflectance
- otherwise -> shading

We compare it against Retinex, the color cue, and a no-depth control (color cue + Retinex)
on the MIT dataset using LMSE.

This repo starts from our Project 2 code.

## Files

| File | What's in it |
|---|---|
| `depth.py` | Runs Depth Anything 3 on MIT objects and caches depth + camera intrinsics |
| `geometry.py` | Surface normals from depth, and the angle between neighboring normals ("is the surface smooth here") |
| `intrinsic.py` | Log derivatives, Retinex and color-cue classifiers, Poisson reconstruction, `decompose()` |
| `metrics.py` | LMSE error (Grosse et al. 2009) |
| `io_utils.py` | Loading photos and MIT data, sRGB/linear conversion, saving figures |
| `evaluate_mit.py` | Runs the methods on MIT, saves figures and a score table, optional threshold sweep |
| `synthetic.py` | 3 synthetic test scenes with known answers |
| `run_photos.py` | Runs the methods on your own photos (no ground truth) |

## Setup

Depth Anything 3 is not on pip, so clone it into `third_party/` (it's gitignored).
We use Python 3.14 because PyTorch and DA3's dependencies have wheels for it.
DA3's full package list includes 3D export tools we don't need (xformers, open3d,
pycolmap), so `requirements.txt` only has what inference needs.

```bash
/opt/homebrew/bin/python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git third_party/Depth-Anything-3
```

The DA3-BASE weights (`depth-anything/da3-base` on Hugging Face) download automatically the
first time `depth.py` runs.

Get the MIT Intrinsic Images dataset from
http://www.cs.toronto.edu/~rgrosse/intrinsic/downloads.html (`intrinsic-data.tar.gz`, about 100 MB)
and unzip it so the objects are in `mit_data/MIT-intrinsic/data/` (one folder per object,
each with `diffuse.png`, `shading.png`, `reflectance.png` and `mask.png`).

`.venv/`, `mit_data/` and `results/` are in `.gitignore`, so don't commit them.

## How to run

```bash
.venv/bin/python depth.py        # DA3 depth for the 5 pilot objects -> depth_cache/
.venv/bin/python geometry.py     # normals + smoothness signal -> results/pilot/geometry.png
.venv/bin/python synthetic.py
.venv/bin/python evaluate_mit.py --data mit_data/MIT-intrinsic/data --sweep
.venv/bin/python run_photos.py photos/
```

Output goes to `results/`.

## Starting point (Project 2 results, MIT, 20 objects, lower is better)

| Method | Mean LMSE |
|---|---|
| Retinex (threshold 0.3) | 0.0375 |
| Color cue (threshold 1 degree) | 0.0319 |

These thresholds were picked on the same 20 objects. For the final project we tune on half
the objects and test on the other half.

## To do

- [x] Run Depth Anything 3 (DA3-BASE) on the 5 pilot objects and save the depth maps (`depth.py`)
- [x] Compute surface normals and a smoothness signal from depth (`geometry.py`)
- [ ] Add the geometry rule and the no-depth control to `intrinsic.py`
- [ ] 2-fold threshold split in `evaluate_mit.py`
- [ ] Pilot on 5 objects, then all 20
