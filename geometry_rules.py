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
# Nov 25 update: decompose_with_geometry() now takes surface normals instead
# of a depth map. For MIT objects the normals come from Depth Anything 3
# (depth.py + geometry.py, which use DA3's real depth and camera intrinsics);
# the old Depth Anything V2 loader was removed because V2 gives inverse depth,
# not depth. The synthetic test still uses surface_normals_from_depth() on its
# known depth. retinex_fallback picks between two rules:
#   True  - color OR Retinex OR (weaker edge AND flat)   [pilot rule]
#   False - color OR (strong edge AND flat)              [proposal rule]

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
    """Unit normals from a depth map on the pixel grid (no camera model)."""
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


def decompose_with_geometry(img, normals, mask=None, color_threshold=1.0,
                             edge_threshold=0.3, normal_threshold=0.05,
                             retinex_threshold=0.3, retinex_fallback=True):
    """Full pipeline: color cue OR (strong-edge-but-flat-geometry) cue,
    plus plain Retinex if retinex_fallback is True. normals is H x W x 3.
    Same return signature as intrinsic.decompose() for drop-in use with
    save_comparison()/lmse()."""
    if mask is None:
        mask = np.ones(img.shape[:2], dtype=bool)

    log_i = log_intensity(img)
    gx, gy = gradients(log_i)

    cx, cy = classify_color(img, color_threshold)
    gex, gey = classify_geometry(gx, gy, normals, edge_threshold, normal_threshold)
    rx, ry = cx | gex, cy | gey

    if retinex_fallback:
        bx, by = classify_retinex(gx, gy, retinex_threshold)  # separate, higher threshold — the real safety net
        rx, ry = rx | bx, ry | by

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
