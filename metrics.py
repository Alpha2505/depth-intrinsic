# Geethanjali Ravichandra
# November 1, 2026
# CS 7170 Advanced Perception - Project 2
#
# metrics.py
# LMSE (local mean squared error) from Grosse et al. ICCV 2009, the usual
# score for the MIT Intrinsic Images dataset. Lower is better, 0 = perfect.
# The functions follow the structure of the reference Python code released
# with that dataset (ssq error, local error, average of shading/reflectance).

import numpy as np


def scale_invariant_ssq(true, est, mask):
    """Squared error after scaling est by the best single factor."""
    denom = np.sum(est ** 2 * mask)
    alpha = np.sum(true * est * mask) / denom if denom > 1e-5 else 0.0
    return np.sum(mask * (true - alpha * est) ** 2)


def local_error(true, est, mask, window=20):
    """Scale-invariant error summed over overlapping windows, normalized."""
    h, w = true.shape
    step = window // 2
    ssq = total = 0.0
    for i in range(0, h - window + 1, step):
        for j in range(0, w - window + 1, step):
            t = true[i:i + window, j:j + window]
            e = est[i:i + window, j:j + window]
            m = mask[i:i + window, j:j + window]
            ssq += scale_invariant_ssq(t, e, m)
            total += np.sum(m * t ** 2)  # error of guessing all zeros
    return ssq / total if total > 0 else 0.0


def lmse(true_shading, true_refl, est_shading, est_refl, mask, window=20):
    """Average of the shading and reflectance errors (compared in grayscale)."""
    def gray(x):
        # average the channels if the image is color
        return x.mean(axis=2) if x.ndim == 3 else x

    mask = mask.astype(np.float64)
    s_err = local_error(gray(true_shading), gray(est_shading), mask, window)
    r_err = local_error(gray(true_refl), gray(est_refl), mask, window)
    return 0.5 * s_err + 0.5 * r_err
