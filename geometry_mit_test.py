# Geethanjali Ravichandra, Bhavya Teja Vejandla
# November 25, 2026
# CS 7170 Advanced Perception - Final Project (pilot study)
#
# geometry_mit_test.py
# Runs all the methods (Retinex, Color cue, Combined, Grid-tile, and two
# versions of Color+Geometry) on real MIT Intrinsic Images objects, using
# surface normals from Depth Anything 3 (cached by depth.py) instead of the
# synthetic bump used in geometry_test.py.
#
# Objects chosen deliberately: apple, pear and cup1 have real curved
# surfaces (the geometry cue's intended use case); teabag1 and teabag2
# are the two objects Project 2's own report flagged as the color cue's
# known failure case (brightness-only printed text, same hue as the
# bag) - if geometry helps there too, that's strong real-world evidence,
# not just synthetic.
#
# Run: python depth.py --objects apple pear cup1 teabag1 teabag2   (once)
#      python geometry_mit_test.py

import argparse
import os

import numpy as np

from io_utils import load_mit_object, save_comparison
from intrinsic import decompose
from metrics import lmse
from geometry import load_geometry
from geometry_rules import decompose_with_geometry, decompose_grid_tile

OBJECTS = ["apple", "pear", "cup1", "teabag1", "teabag2"]


METHODS = ["Retinex", "Color cue", "Combined", "Grid-tile", "Geo (pilot rule)", "Geo (proposal rule)"]


def main():
    """Run every method on each object, print and save LMSE, and save figures."""
    parser = argparse.ArgumentParser(description="Geometry cue test on real MIT objects")
    parser.add_argument("--data", default="mit_data/MIT-intrinsic/data", help="folder with one subfolder per object")
    parser.add_argument("--cache", default="depth_cache", help="DA3 depth saved by depth.py")
    parser.add_argument("--blur", type=float, default=2.0, help="depth blur (sigma) before computing normals")
    parser.add_argument("--out", default="results/geometry_mit")
    parser.add_argument("--edge-threshold", type=float, default=0.12,
                        help="geometry classifier's edge threshold (lower than Retinex's 0.3, "
                             "since a confirmed-flat surface needs less evidence)")
    parser.add_argument("--normal-threshold", type=float, default=0.05,
                        help="how flat a normal has to be to count as 'no geometric explanation' (radians)")
    parser.add_argument("--strong-threshold", type=float, default=0.3,
                        help="edge threshold for the proposal rule (no Retinex fallback)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = ["object,retinex,color_cue,combined,grid_tile,geo_pilot_rule,geo_proposal_rule"]
    print(f"{'object':<10} {'Retinex':>9} {'Color':>9} {'Combined':>9} {'Grid-tile':>9} {'Geo-pilot':>9} {'Geo-prop':>9}")
    all_scores = []

    for name in OBJECTS:
        obj = load_mit_object(args.data, name)
        image, mask = obj["image"], obj["mask"]

        _, normals = load_geometry(args.cache, name, args.blur)

        results, scores = {}, {}
        for label, method in [("Retinex", "retinex"), ("Color cue", "color"), ("Combined", "combined")]:
            s, r, e = decompose(image, method, mask)
            results[label] = (s, r, e)
            scores[label] = lmse(obj["shading"], obj["reflectance"], s, r, mask)

        s, r, e = decompose_grid_tile(image, mask, tile_size=32)
        results["Grid-tile"] = (s, r, e)
        scores["Grid-tile"] = lmse(obj["shading"], obj["reflectance"], s, r, mask)

        # pilot rule: color OR Retinex OR (weaker edge AND flat)
        s, r, e = decompose_with_geometry(image, normals, mask,
                                   edge_threshold=args.edge_threshold,      # 0.12, geometry-specific
                                   normal_threshold=args.normal_threshold,
                                   retinex_threshold=0.3)                   # matches Combined's safety net
        results["Geo (pilot rule)"] = (s, r, e)
        scores["Geo (pilot rule)"] = lmse(obj["shading"], obj["reflectance"], s, r, mask)

        # proposal rule: color OR (strong edge AND flat), no Retinex fallback
        s, r, e = decompose_with_geometry(image, normals, mask,
                                   edge_threshold=args.strong_threshold,
                                   normal_threshold=args.normal_threshold,
                                   retinex_fallback=False)
        results["Geo (proposal rule)"] = (s, r, e)
        scores["Geo (proposal rule)"] = lmse(obj["shading"], obj["reflectance"], s, r, mask)

        all_scores.append([scores[k] for k in METHODS])
        print(f"{name:<10} " + " ".join(f"{scores[k]:>9.4f}" for k in METHODS))
        rows.append(f"{name}," + ",".join(f"{scores[k]:.4f}" for k in METHODS))

        save_comparison(os.path.join(args.out, f"{name}.png"), image, results, mask,
                         obj["shading"], obj["reflectance"], title=name)

    means = np.mean(all_scores, axis=0)
    print(f"{'MEAN':<10} " + " ".join(f"{v:>9.4f}" for v in means))
    rows.append("MEAN," + ",".join(f"{v:.4f}" for v in means))

    with open(os.path.join(args.out, "lmse.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nSaved figures and lmse.csv to {args.out}/")


if __name__ == "__main__":
    main()