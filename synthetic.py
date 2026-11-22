# Geethanjali Ravichandra
# November 1, 2026
# CS 7170 Advanced Perception - Project 2
#
# synthetic.py
# Makes fake "Mondrian" scenes (colored rectangles under smooth light) where
# I know the true reflectance and shading, then runs both methods on them.
#   smooth - color patches, smooth light (both should work)
#   shadow - adds a hard shadow (Retinex should fail)
#   gray   - gray patches only (color cue should fail)
#
# Run: python synthetic.py

import argparse
import os

import cv2
import numpy as np

from intrinsic import decompose
from io_utils import save_comparison, save_image, to_display
from metrics import lmse


def mondrian(h, w, n_rects, rng, gray=False):
    """Reflectance image made of random overlapping rectangles."""
    refl = np.full((h, w, 3), 0.5)
    for _ in range(n_rects):
        y0, x0 = rng.integers(0, h - 20), rng.integers(0, w - 20)
        y1, x1 = y0 + rng.integers(20, h // 2), x0 + rng.integers(20, w // 2)
        color = np.full(3, rng.uniform(0.1, 0.9)) if gray else rng.uniform(0.1, 0.9, 3)
        refl[y0:y1, x0:x1] = color
    return refl


def smooth_shading(h, w):
    """Smooth lighting: a wide spotlight that also fades to the right."""
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    spot = np.exp(-((xx - 0.35) ** 2 + (yy - 0.3) ** 2) / (2 * 0.35 ** 2))
    return 0.15 + 0.85 * spot * (1.0 - 0.4 * xx)


def add_shadow(shading, strength=0.35):
    """Darken a triangle in the top right to make a hard cast shadow."""
    h, w = shading.shape
    pts = np.array([[w * 0.55, 0], [w, 0], [w, h * 0.9]], dtype=np.int32)
    region = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(region, [pts], 1)
    factor = np.where(region > 0, strength, 1.0)
    factor = cv2.GaussianBlur(factor, (0, 0), 0.7)  # soften the edge a tiny bit
    return shading * factor


def make_scene(kind, size=256, seed=0):
    """Returns (image, true shading, true reflectance) for one scene type."""
    rng = np.random.default_rng(seed)
    refl = mondrian(size, size, 25, rng, gray=(kind == "gray"))
    shading = smooth_shading(size, size)
    if kind == "shadow":
        shading = add_shadow(shading)
    image = refl * shading[:, :, None]
    image = np.clip(image + rng.normal(0, 0.003, image.shape), 0, None)  # a little noise
    return image, shading, refl


def main():
    """Run both methods on the three scenes, save figures and scores."""
    parser = argparse.ArgumentParser(description="Test both methods on synthetic scenes")
    parser.add_argument("--out", default="results/synthetic")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    methods = {"Retinex": "retinex", "Color cue": "color"}
    lines = ["scene,method,lmse"]
    print(f"{'scene':<8} {'Retinex':>9} {'Color cue':>10}")
    for kind in ["smooth", "shadow", "gray"]:
        image, true_s, true_r = make_scene(kind)
        mask = np.ones(image.shape[:2], dtype=bool)
        save_image(os.path.join(args.out, f"{kind}_input.png"), to_display(image))

        results, scores = {}, {}
        for label, method in methods.items():
            s, r, edges = decompose(image, method)
            results[label] = (s, r, edges)
            scores[label] = lmse(true_s, true_r, s, r, mask)
            lines.append(f"{kind},{method},{scores[label]:.4f}")

        save_comparison(os.path.join(args.out, f"{kind}_comparison.png"), image,
                        results, mask, true_s, true_r,
                        title=f"Synthetic scene: {kind}")
        print(f"{kind:<8} {scores['Retinex']:>9.4f} {scores['Color cue']:>10.4f}")

    with open(os.path.join(args.out, "lmse.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved figures and lmse.csv to {args.out}/")


if __name__ == "__main__":
    main()
