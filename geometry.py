# Geethanjali Ravichandra
# November 25, 2026
# CS 7170 Advanced Perception - Final Project
#
# geometry.py
# Turns a DA3 depth map into surface normals and a "surface is smooth here"
# signal. The signal is the angle between the normals of neighboring pixels,
# laid out the same way as the image derivatives in intrinsic.py (x = pixel
# and its right neighbor, y = pixel and the one below), so the two can be
# compared edge by edge.
#
# Run: python geometry.py --objects teabag1 box turtle potato sun
#      (needs depth_cache/ from depth.py, saves figures to results/pilot/)

import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter

from io_utils import load_mit_object, save_grid, to_display

PILOT_OBJECTS = ["teabag1", "box", "turtle", "potato", "sun"]


def depth_to_normals(depth, K, blur_sigma=1.0):
    """Unit surface normals (H x W x 3) from a depth map and camera intrinsics."""
    # a little blur first, otherwise pixel noise in the depth shows up as fake bumps
    depth = gaussian_filter(depth, blur_sigma) if blur_sigma > 0 else depth

    # back-project every pixel to a 3D point using the pinhole camera model
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    points = np.stack([x, y, depth], axis=2)

    # normal = cross product of the surface tangents in the u and v directions
    du = np.gradient(points, axis=1)
    dv = np.gradient(points, axis=0)
    normals = np.cross(du, dv)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True) + 1e-12
    normals[normals[:, :, 2] > 0] *= -1  # make every normal face the camera
    return normals


def normal_change(normals):
    """Angle in degrees between the normals of neighboring pixels (x and y)."""
    ax = np.zeros(normals.shape[:2])
    ay = np.zeros(normals.shape[:2])
    dot_x = np.sum(normals[:, 1:] * normals[:, :-1], axis=2)
    dot_y = np.sum(normals[1:, :] * normals[:-1, :], axis=2)
    ax[:, :-1] = np.degrees(np.arccos(np.clip(dot_x, -1, 1)))
    ay[:-1, :] = np.degrees(np.arccos(np.clip(dot_y, -1, 1)))
    return ax, ay


def smooth_surface(normals, threshold_deg=2.0):
    """True where the surface is smooth, i.e. the normal barely turns between neighbors."""
    # threshold is just a starting value; it gets tuned in the main study
    ax, ay = normal_change(normals)
    return ax < threshold_deg, ay < threshold_deg


def load_geometry(cache_dir, name, blur_sigma=1.0):
    """Load cached depth for one object and return (depth, normals)."""
    data = np.load(os.path.join(cache_dir, f"{name}.npz"))
    depth, K = data["depth"], data["intrinsics"]
    return depth, depth_to_normals(depth, K, blur_sigma)


def main():
    """Compute normals and the smoothness signal for each object and save a figure."""
    parser = argparse.ArgumentParser(description="Normals and smoothness from DA3 depth")
    parser.add_argument("--data", default="mit_data/MIT-intrinsic/data")
    parser.add_argument("--objects", nargs="+", default=PILOT_OBJECTS)
    parser.add_argument("--cache", default="depth_cache")
    parser.add_argument("--threshold", type=float, default=2.0, help="smoothness threshold in degrees")
    parser.add_argument("--out", default="results/pilot")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for name in args.objects:
        obj = load_mit_object(args.data, name)
        mask = obj["mask"]
        depth, normals = load_geometry(args.cache, name)
        ax, ay = normal_change(normals)
        sx, sy = smooth_surface(normals, args.threshold)

        # pictures for the figure (masked to the object)
        d = depth[mask]
        depth_vis = np.clip((depth - d.min()) / max(d.max() - d.min(), 1e-8), 0, 1)
        normal_vis = (normals * [1, -1, -1] + 1) / 2  # usual normal-map colors
        change_vis = np.clip(np.maximum(ax, ay) / 10.0, 0, 1)  # 10 degrees = white
        smooth_vis = (sx & sy).astype(float)
        rows.append([to_display(obj["image"], mask), (1 - depth_vis) * mask,
                     normal_vis * mask[:, :, None], change_vis * mask, smooth_vis * mask])

        frac = (sx & sy)[mask].mean()
        print(f"{name}: median normal change {np.median(np.maximum(ax, ay)[mask]):.2f} deg, "
              f"{100 * frac:.0f}% of object pixels smooth at {args.threshold} deg")

    save_grid(os.path.join(args.out, "geometry.png"), rows, args.objects,
              ["Input", "Depth (near = bright)", "Normals",
               "Normal change (0-10 deg)", f"Smooth (< {args.threshold:g} deg)"])
    print(f"Saved {args.out}/geometry.png")


if __name__ == "__main__":
    main()
