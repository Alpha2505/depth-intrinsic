# Geethanjali Ravichandra
# November 1, 2026
# CS 7170 Advanced Perception - Project 2
#
# io_utils.py
# Loading images (my photos and the MIT dataset), sRGB <-> linear
# conversion, and saving images/figures.

import os

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # just save figures, don't open windows
import matplotlib.pyplot as plt


def srgb_to_linear(img):
    """Undo the sRGB gamma so pixel values are proportional to light."""
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img):
    """Apply the sRGB gamma so a linear image looks right on screen."""
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * img ** (1 / 2.4) - 0.055)


def read_image(path):
    """Read an 8 or 16 bit image as floats in [0, 1] (RGB order)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    max_val = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0
    img = img.astype(np.float64) / max_val
    if img.ndim == 3:
        img = img[:, :, :3][:, :, ::-1]  # drop alpha, BGR -> RGB
    return img


def load_photo(path, max_size=640):
    """Load a normal photo, shrink it if it's big, and convert to linear RGB."""
    # IMREAD_COLOR uses the EXIF rotation so phone photos aren't sideways
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path} (convert HEIC photos to JPG)")
    img = img[:, :, ::-1].astype(np.float64) / 255.0
    scale = max_size / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return srgb_to_linear(img)


def load_mit_object(data_dir, name):
    """Load image, true shading, true reflectance and mask for one MIT object."""
    folder = os.path.join(data_dir, name)

    def gray(img):
        # average the channels if the image is color
        return img.mean(axis=2) if img.ndim == 3 else img

    image = read_image(os.path.join(folder, "diffuse.png"))  # already linear
    shading = gray(read_image(os.path.join(folder, "shading.png")))
    reflectance = read_image(os.path.join(folder, "reflectance.png"))
    mask = gray(read_image(os.path.join(folder, "mask.png"))) > 0.5
    return {"name": name, "image": image, "shading": shading,
            "reflectance": reflectance, "mask": mask}


def list_mit_objects(data_dir):
    """Names of the object folders in the MIT data directory."""
    return sorted(d for d in os.listdir(data_dir)
                  if os.path.isfile(os.path.join(data_dir, d, "diffuse.png")))


def to_display(img, mask=None, top=None):
    """Make a linear image viewable: scale so the bright end is 1, then gamma."""
    # passing top lets two images be shown at the same brightness
    if top is None:
        valid = img[mask] if mask is not None else img
        top = np.percentile(valid, 99) if valid.size else 1.0
    out = linear_to_srgb(img / max(top, 1e-8))
    if mask is not None:
        out = out * (mask[:, :, None] if out.ndim == 3 else mask)
    return out


def save_image(path, img):
    """Save a [0, 1] image as an 8-bit PNG."""
    out = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    if out.ndim == 3:
        out = out[:, :, ::-1]  # OpenCV wants BGR
    cv2.imwrite(path, out)


def save_grid(path, rows, row_labels, col_labels, title=None):
    """Save a grid of images (list of rows) as one figure."""
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows),
                             squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r][c]
            ax.imshow(np.clip(img, 0, 1), cmap="gray" if img.ndim == 2 else None,
                      vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_labels[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison(path, image, results, mask=None, gt_shading=None,
                    gt_reflectance=None, title=None):
    """Figure with ground truth (if we have it) on top and one row per method."""
    # results is {method name: (shading, reflectance, edges)}
    blank = np.zeros(image.shape[:2])
    rows, labels = [], []
    if gt_shading is not None:
        rows.append([to_display(image, mask), to_display(gt_reflectance, mask),
                     to_display(gt_shading, mask)])
        labels.append("Ground truth")
    else:
        rows.append([to_display(image, mask), blank, blank])
        labels.append("Input")
    for name, (shading, reflectance, edges) in results.items():
        rows.append([edges.astype(float), to_display(reflectance, mask),
                     to_display(shading, mask)])
        labels.append(name)
    col_labels = ["Input / reflectance edges", "Reflectance", "Shading"]
    save_grid(path, rows, labels, col_labels, title)
