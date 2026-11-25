# Geethanjali Ravichandra
# November 25, 2026
# CS 7170 Advanced Perception - Final Project
#
# depth.py
# Runs Depth Anything 3 (DA3-BASE, inference only) on MIT objects and saves
# the predicted depth and camera intrinsics to depth_cache/<object>.npz, so
# the depth model only has to run once per image.
#
# The MIT images are linear 16-bit, but DA3 was trained on normal photos,
# so we convert to 8-bit sRGB before giving them to the model.
#
# Run: python depth.py --objects teabag1 box turtle potato sun

import argparse
import os
import sys
import types

import cv2
import numpy as np
import torch

from io_utils import linear_to_srgb, read_image

# DA3 lives in third_party/ (see README). Its 3D export code imports pycolmap,
# which we don't use and which clashes with torch's OpenMP on macOS, so we
# give it an empty stand-in module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party", "Depth-Anything-3", "src"))
sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
from depth_anything_3.api import DepthAnything3  # noqa: E402

PILOT_OBJECTS = ["teabag1", "box", "turtle", "potato", "sun"]


def load_model(name="depth-anything/da3-base"):
    """Download (first time only) and load DA3, on the Mac GPU if possible."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(name).to(device).eval()
    return model


def predict_depth(model, image_linear):
    """Depth map and 3x3 intrinsics for one linear RGB image, at the image's own size."""
    h, w = image_linear.shape[:2]
    srgb = (linear_to_srgb(image_linear / max(image_linear.max(), 1e-8)) * 255).astype(np.uint8)
    with torch.no_grad():
        pred = model.inference([srgb])

    # DA3 works at its own resolution, so scale depth and intrinsics back to ours
    depth = pred.depth[0]
    ph, pw = depth.shape
    depth = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    K = pred.intrinsics[0].astype(np.float64).copy()
    K[0] *= w / pw
    K[1] *= h / ph
    return depth, K


def main():
    """Run DA3 on each object and save the results."""
    parser = argparse.ArgumentParser(description="Depth Anything 3 on MIT objects")
    parser.add_argument("--data", default="mit_data/MIT-intrinsic/data")
    parser.add_argument("--objects", nargs="+", default=PILOT_OBJECTS)
    parser.add_argument("--model", default="depth-anything/da3-base")
    parser.add_argument("--out", default="depth_cache")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.model)
    for name in args.objects:
        image = read_image(os.path.join(args.data, name, "diffuse.png"))
        depth, K = predict_depth(model, image)
        np.savez(os.path.join(args.out, f"{name}.npz"), depth=depth, intrinsics=K)
        print(f"{name}: depth {depth.shape}, range {depth.min():.3f} to {depth.max():.3f}")
    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()
