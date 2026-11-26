# Geethanjali Ravichandra, Bhavya Teja Vejandla
# November 18, 2026
# CS 7170 Advanced Perception - Final Project (pilot study)
#
# geometry_test.py
# A synthetic scene built specifically to probe the hypothesis: a curved
# "pillow" bump (strong but smooth shading gradient a plain magnitude
# threshold would over-flag as reflectance) PLUS a same-hue "ink mark" on
# the FLAT background (brightness-only edge, same failure mode as the real
# teabag1 case from Project 2's report - color cue alone misses it).
#
# Five methods are compared against ground truth:
#   Retinex        - intrinsic.decompose(method="retinex")
#   Color cue      - intrinsic.decompose(method="color")
#   Combined       - intrinsic.decompose(method="combined")  [no-depth control]
#   Grid-tile      - geometry.decompose_grid_tile()           [no-geometry control]
#   Color+Geometry - geometry.decompose_with_geometry()        [our new method]
#
# Run: python geometry_test.py

import os

import numpy as np

from intrinsic import decompose
from io_utils import save_comparison, save_image, to_display
from metrics import lmse
from geometry_rules import decompose_with_geometry, decompose_grid_tile, surface_normals_from_depth


def make_depth(h, w):
    """Flat background (depth=1) with a raised, fairly steep bump in the
    middle. Tuned so the steepest part of the bump's shading gradient
    (its rim, where it meets the flat background) crosses Retinex's
    magnitude threshold -- a real risk of over-segmenting a smoothly
    curved, single-material surface into fake "reflectance" patches,
    the same false-positive failure mode a plain magnitude threshold has
    on a curved object like Tappen's pillow example."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx, radius = h * 0.5, w * 0.35, min(h, w) * 0.18
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    bump_height = np.clip(1.0 - (r / radius) ** 2, 0, None)
    depth = 1.0 + 2.5 * np.sqrt(bump_height)
    return depth


def lambertian_shading(depth, light_dir=(0.7, 0.2, 0.5)):
    """Shading = dot(surface normal, light direction), Lambertian model."""
    dzdy, dzdx = np.gradient(depth)
    normals = np.dstack([-dzdx, -dzdy, np.ones_like(depth)])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    light = np.array(light_dir) / np.linalg.norm(light_dir)
    shading = np.clip(normals @ light, 0.05, None)
    return shading / shading.max()


def make_reflectance(h, w, base_color=(0.55, 0.45, 0.30)):
    """Uniform base color, plus a same-hue, darker "ink mark" rectangle
    stamped on the FLAT background (away from the bump) - a brightness-only
    edge with no hue change and no underlying geometry change. The contrast
    (0.82x) is deliberately kept WEAK -- its log-intensity edge sits below
    Retinex's 0.3 magnitude threshold (so plain Retinex, and Combined,
    both miss it, same as the real teabag1 failure case from Project 2's
    report), but above the geometry classifier's lower threshold, which is
    only usable BECAUSE the surface there is confirmed flat."""
    refl = np.ones((h, w, 3)) * np.array(base_color)
    y0, y1 = int(h * 0.65), int(h * 0.80)
    x0, x1 = int(w * 0.55), int(w * 0.85)
    refl[y0:y1, x0:x1] = np.array(base_color) * 0.82  # weak contrast, same hue
    return refl, (y0, y1, x0, x1)


def main():
    """Build the synthetic scene, run every method, and save scores and a figure."""
    h = w = 256
    out = "results/geometry_pilot"
    os.makedirs(out, exist_ok=True)

    depth = make_depth(h, w)
    true_shading = lambertian_shading(depth)
    true_refl, ink_box = make_reflectance(h, w)
    image = true_refl * true_shading[:, :, None]
    mask = np.ones((h, w), dtype=bool)

    save_image(os.path.join(out, "input.png"), to_display(image))

    results, scores = {}, {}

    for label, method in [("Retinex", "retinex"), ("Color cue", "color"), ("Combined", "combined")]:
        s, r, e = decompose(image, method)
        results[label] = (s, r, e)
        scores[label] = lmse(true_shading, true_refl, s, r, mask)

    s, r, e = decompose_grid_tile(image, mask, tile_size=32)
    results["Grid-tile"] = (s, r, e)
    scores["Grid-tile"] = lmse(true_shading, true_refl, s, r, mask)

    # geometry's own edge threshold is lower than Retinex's 0.3: a flat,
    # confirmed-geometry region needs much less evidence to call an edge
    # "material," since there's no competing shading explanation there.
    normals = surface_normals_from_depth(depth)  # true depth, so this is a sanity check
    s, r, e = decompose_with_geometry(image, normals, mask, edge_threshold=0.12)
    results["Color+Geometry"] = (s, r, e)
    scores["Color+Geometry"] = lmse(true_shading, true_refl, s, r, mask)

    save_comparison(os.path.join(out, "comparison.png"), image, results, mask,
                     true_shading, true_refl,
                     title="Pilot test: curved bump + same-hue ink mark")

    print(f"{'Method':<16} {'LMSE (whole image)':>20}")
    lines = ["method,lmse_whole,lmse_ink_patch"]
    y0, y1, x0, x1 = ink_box
    ink_mask = np.zeros((h, w), dtype=bool)
    ink_mask[y0 - 10:y1 + 10, x0 - 10:x1 + 10] = True  # ink mark + a margin
    for label in ["Retinex", "Color cue", "Combined", "Grid-tile", "Color+Geometry"]:
        s, r, e = results[label]
        ink_score = lmse(true_shading, true_refl, s, r, ink_mask)
        print(f"{label:<16} {scores[label]:>20.4f}   (ink patch region: {ink_score:.4f})")
        lines.append(f"{label},{scores[label]:.4f},{ink_score:.4f}")

    with open(os.path.join(out, "lmse.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved figures and lmse.csv to {out}/")


if __name__ == "__main__":
    main()
