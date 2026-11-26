# Geethanjali Ravichandra, Bhavya Teja Vejandla
# November 18, 2026
# CS 7170 Advanced Perception - Final Project (pilot study)
#
# geometry_rules.py  (sent as geometry.py; renamed so it does not clash with our geometry.py)
# Extends Project 2's intrinsic.py with a geometry cue: a monocular depth
# estimate is used to decide whether a strong intensity edge is explained
# by the surface's own shape (so it must be shading) or NOT explained by
# the shape (so it must be a real material/reflectance edge, e.g. printed
# ink on a flat surface).
#
# Final rule, replacing plain Retinex:
#   1. If the color/chromaticity changes across the edge -> reflectance
#      (same color cue as Project 2, unchanged).
#   2. Else, if the edge is strong AND the surface normal is flat there
#      (no geometric reason for a brightness change) -> reflectance.
#   3. Otherwise -> shading.
#
# This also includes the two controls from the proposal:
#   - no-depth control: just Project 2's existing "combined" method
#     (color cue OR plain Retinex), already implemented in intrinsic.py.
#   - grid-tile control: re-run the color cue PER TILE instead of
#     globally, with NO depth or normals involved, to separate any gain
#     from "using geometry" from any gain that just comes from making
#     decisions locally instead of over the whole image.
#
# NOTE ON DEPTH: load_depth_map()'s "real" mode calls Depth Anything 3 via
# the documented Hugging Face `transformers` pipeline API. This has NOT
# been run or verified in this environment (no GPU/internet here) - verify
# on your own machine before trusting it. The "synthetic" mode below lets
# us test the surrounding logic (normals, classifier, reconstruction) right
# now, without the real model.

import numpy as np
from scipy.ndimage import gaussian_filter
from intrinsic import classify_retinex

from intrinsic import (
    log_intensity, gradients, pair_masks, classify_color,
    poisson_reconstruct,
)

EPS = 1e-4


# ---------------------------------------------------------------------
# Depth -> surface normals
# ---------------------------------------------------------------------

def surface_normals_from_depth(depth, blur_sigma=2.0):
    depth_smooth = gaussian_filter(depth.astype(np.float64), blur_sigma)  # filters out texture-induced noise
    dzdy, dzdx = np.gradient(depth_smooth)
    normals = np.dstack([-dzdx, -dzdy, np.ones_like(depth_smooth)])
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm[norm == 0] = 1e-8
    return normals / norm


def normal_change(normals):
    """How much the surface normal changes across each horizontal/vertical
    edge (angle in radians between neighboring unit normals). Small value
    = flat/smooth surface there; large value = the surface is genuinely
    curving or has a depth discontinuity."""
    dot_x = np.sum(normals[:, 1:] * normals[:, :-1], axis=2)
    dot_y = np.sum(normals[1:, :] * normals[:-1, :], axis=2)
    ang_x = np.zeros(normals.shape[:2])
    ang_y = np.zeros(normals.shape[:2])
    ang_x[:, :-1] = np.arccos(np.clip(dot_x, -1, 1))
    ang_y[:-1, :] = np.arccos(np.clip(dot_y, -1, 1))
    return ang_x, ang_y


# ---------------------------------------------------------------------
# Depth loading: synthetic (testable now) vs. real (untested here)
# ---------------------------------------------------------------------

def load_depth_map(img, mode="synthetic", synthetic_depth=None):
    if mode == "synthetic":
        if synthetic_depth is not None:
            return synthetic_depth
        # flat fallback (no curvature anywhere)
        return np.ones(img.shape[:2], dtype=np.float64)

    elif mode == "real":
        from transformers import pipeline
        from PIL import Image as PILImage
        depth_pipe = pipeline(task="depth-estimation",
                            model="depth-anything/Depth-Anything-V2-Small-hf")
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_uint8)  # the pipeline needs a PIL Image, not a raw array
        result = depth_pipe(pil_img)
        return result["predicted_depth"].cpu().numpy()

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------
# The new geometry-informed classifier
# ---------------------------------------------------------------------

def classify_geometry(gx, gy, normals, edge_threshold=0.3, normal_threshold=0.05):
    """An edge is reflectance if it is strong (same magnitude test as
    Retinex) AND the surface normal barely changes there (no geometric
    reason for the brightness change). A strong edge where the normal
    ALSO changes a lot is left as shading, since the curvature already
    explains the brightness change."""
    ang_x, ang_y = normal_change(normals)
    strong_x = np.abs(gx) > edge_threshold
    strong_y = np.abs(gy) > edge_threshold
    flat_x = ang_x < normal_threshold
    flat_y = ang_y < normal_threshold
    return strong_x & flat_x, strong_y & flat_y


def decompose_with_geometry(img, depth, mask=None, color_threshold=1.0,
                             edge_threshold=0.3, normal_threshold=0.05,
                             retinex_threshold=0.3):
    """Full pipeline: color cue OR (strong-edge-but-flat-geometry) cue.
    Same return signature as intrinsic.decompose() for drop-in use with
    save_comparison()/lmse()."""
    if mask is None:
        mask = np.ones(img.shape[:2], dtype=bool)

    log_i = log_intensity(img)
    gx, gy = gradients(log_i)

    cx, cy = classify_color(img, color_threshold)
    normals = surface_normals_from_depth(depth)
    gex, gey = classify_geometry(gx, gy, normals, edge_threshold, normal_threshold)

    bx, by = classify_retinex(gx, gy, retinex_threshold)  # separate, higher threshold — the real safety net
    rx, ry = cx | bx | gex, cy | by | gey

    mx, my = pair_masks(mask)
    rx &= mx
    ry &= my

    sx = np.where(rx | ~mx, 0.0, gx)
    sy = np.where(ry | ~my, 0.0, gy)
    log_s = poisson_reconstruct(sx, sy)

    shading = np.exp(log_s)
    shading /= np.percentile(shading[mask], 99)
    reflectance = img / shading[:, :, None]

    edges = rx | ry
    return shading * mask, reflectance * mask[:, :, None], edges


# ---------------------------------------------------------------------
# Control: grid-tile, NO depth/normals at all
# ---------------------------------------------------------------------

def decompose_grid_tile(img, mask=None, color_threshold=1.0, edge_threshold=0.3,
                         tile_size=32):
    """No-geometry, local-only control: re-run the SAME color-cue-plus-
    Retinex decision, but separately per fixed grid tile instead of once
    globally. No depth or normals are used anywhere. This isolates
    whether a gain comes from geometry specifically, or just from
    making decisions locally rather than over the whole image."""
    if mask is None:
        mask = np.ones(img.shape[:2], dtype=bool)

    log_i = log_intensity(img)
    gx, gy = gradients(log_i)
    cx, cy = classify_color(img, color_threshold)

    h, w = img.shape[:2]
    rx = np.zeros_like(cx)
    ry = np.zeros_like(cy)
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
            tile_gx = gx[y0:y1, x0:x1]
            tile_gy = gy[y0:y1, x0:x1]
            # Retinex threshold applied PER TILE (local statistics), then
            # OR'd with the (already-local, pixelwise) color cue.
            tile_strong_x = np.abs(tile_gx) > edge_threshold
            tile_strong_y = np.abs(tile_gy) > edge_threshold
            rx[y0:y1, x0:x1] = cx[y0:y1, x0:x1] | tile_strong_x
            ry[y0:y1, x0:x1] = cy[y0:y1, x0:x1] | tile_strong_y

    mx, my = pair_masks(mask)
    rx &= mx
    ry &= my

    sx = np.where(rx | ~mx, 0.0, gx)
    sy = np.where(ry | ~my, 0.0, gy)
    log_s = poisson_reconstruct(sx, sy)

    shading = np.exp(log_s)
    shading /= np.percentile(shading[mask], 99)
    reflectance = img / shading[:, :, None]

    edges = rx | ry
    return shading * mask, reflectance * mask[:, :, None], edges
