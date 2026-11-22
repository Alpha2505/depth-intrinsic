# Geethanjali Ravichandra
# November 1, 2026
# CS 7170 Advanced Perception - Project 2
#
# intrinsic.py
# Main algorithm. Splits an image into reflectance and shading (I = R * S).
# Each derivative of the log image gets labeled as either a reflectance
# change or a shading change, and then the shading is rebuilt from the
# shading derivatives with a Poisson solve.
#
# Two ways of labeling the derivatives:
#   retinex - big derivatives are reflectance (Land & McCann)
#   color   - if the color changes across the edge it's reflectance
#             (the color cue from Tappen, Freeman & Adelson, NIPS 2002)

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter

EPS = 1e-4  # so we never take log(0)


def log_intensity(img):
    """Log of the gray image (average of R, G, B)."""
    gray = img.mean(axis=2) if img.ndim == 3 else img
    return np.log(gray + EPS)


def gradients(f):
    """Forward differences in x and y. Last column/row is left as 0."""
    gx = np.zeros_like(f)
    gy = np.zeros_like(f)
    gx[:, :-1] = f[:, 1:] - f[:, :-1]
    gy[:-1, :] = f[1:, :] - f[:-1, :]
    return gx, gy


def pair_masks(mask):
    """True where both pixels of a derivative are inside the object mask."""
    mx = np.zeros_like(mask)
    my = np.zeros_like(mask)
    mx[:, :-1] = mask[:, 1:] & mask[:, :-1]
    my[:-1, :] = mask[1:, :] & mask[:-1, :]
    return mx, my


def classify_retinex(gx, gy, threshold=0.3):
    """Retinex: anything above the threshold counts as a reflectance edge."""
    return np.abs(gx) > threshold, np.abs(gy) > threshold


def chromaticity_angle(img, blur_sigma=1.0):
    """Angle in degrees between the RGB vectors of neighboring pixels."""
    # shading scales R, G and B by the same amount, so the direction of the
    # color vector only changes when the material changes
    smooth = np.stack([gaussian_filter(img[:, :, c], blur_sigma) for c in range(3)], axis=2)
    smooth = smooth + 1e-3  # dark pixels are noisy, this pulls them toward gray
    unit = smooth / np.linalg.norm(smooth, axis=2, keepdims=True)

    ax = np.zeros(img.shape[:2])
    ay = np.zeros(img.shape[:2])
    dot_x = np.sum(unit[:, 1:] * unit[:, :-1], axis=2)
    dot_y = np.sum(unit[1:, :] * unit[:-1, :], axis=2)
    ax[:, :-1] = np.degrees(np.arccos(np.clip(dot_x, -1, 1)))
    ay[:-1, :] = np.degrees(np.arccos(np.clip(dot_y, -1, 1)))
    return ax, ay


def classify_color(img, threshold_deg=1.0, blur_sigma=1.0):
    """Color cue: edges where the color angle is above the threshold are reflectance."""
    ax, ay = chromaticity_angle(img, blur_sigma)
    return ax > threshold_deg, ay > threshold_deg


def poisson_reconstruct(gx, gy):
    """Rebuild an image from its x/y derivatives (least squares, solved with a DCT)."""
    h, w = gx.shape

    # divergence of the gradient field
    div = gx.copy()
    div[:, 1:] -= gx[:, :-1]
    div += gy
    div[1:, :] -= gy[:-1, :]

    # the Laplacian is diagonal in the DCT basis, so we can just divide
    yy = 2 * np.cos(np.pi * np.arange(h) / h) - 2
    xx = 2 * np.cos(np.pi * np.arange(w) / w) - 2
    denom = yy[:, None] + xx[None, :]
    denom[0, 0] = 1.0  # the constant offset can't be recovered anyway

    coeffs = dctn(div, norm="ortho") / denom
    coeffs[0, 0] = 0.0
    return idctn(coeffs, norm="ortho")


def decompose(img, method="color", mask=None, retinex_threshold=0.3,
              color_threshold=1.0):
    """Split a linear RGB image into (shading, reflectance, edge map)."""
    if mask is None:
        mask = np.ones(img.shape[:2], dtype=bool)

    # log image derivatives: log I = log R + log S
    log_i = log_intensity(img)
    gx, gy = gradients(log_i)

    # decide which derivatives are reflectance edges
    if method == "retinex":
        rx, ry = classify_retinex(gx, gy, retinex_threshold)
    elif method == "color":
        rx, ry = classify_color(img, color_threshold)
    else:
        raise ValueError(f"Unknown method: {method}")

    # skip derivatives that cross the edge of the object
    mx, my = pair_masks(mask)
    rx &= mx
    ry &= my

    # shading = whatever is left after removing the reflectance edges
    sx = np.where(rx | ~mx, 0.0, gx)
    sy = np.where(ry | ~my, 0.0, gy)
    log_s = poisson_reconstruct(sx, sy)

    # back out of log space; scale is unknown so brightest shading = 1
    shading = np.exp(log_s)
    shading /= np.percentile(shading[mask], 99)
    reflectance = img / shading[:, :, None]

    edges = rx | ry
    return shading * mask, reflectance * mask[:, :, None], edges
