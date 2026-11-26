# Geethanjali Ravichandra, Bhavya Teja Vejandla
# November 25, 2026
# CS 7170 Advanced Perception - Final Project (pilot study)
#
# geometry_mit_test.py
# Runs all five methods (Retinex, Color cue, Combined, Grid-tile,
# Color+Geometry) on real MIT Intrinsic Images objects, using a REAL
# monocular depth estimate (Depth Anything V2, via load_depth_map's
# "real" mode) instead of the synthetic bump used in geometry_test.py.
#
# Objects chosen deliberately: apple, pear and cup1 have real curved
# surfaces (the geometry cue's intended use case); teabag1 and teabag2
# are the two objects Project 2's own report flagged as the color cue's
# known failure case (brightness-only printed text, same hue as the
# bag) - if geometry helps there too, that's strong real-world evidence,
# not just synthetic.
#
# Run: python geometry_mit_test.py --data MIT-intrinsic/data

import argparse
import os

import numpy as np

from io_utils import load_mit_object, save_comparison
from intrinsic import decompose
from metrics import lmse
from geometry_rules import decompose_with_geometry, decompose_grid_tile, load_depth_map

OBJECTS = ["apple", "pear", "cup1", "teabag1", "teabag2"]


def main():
    parser = argparse.ArgumentParser(description="Geometry cue test on real MIT objects")
    parser.add_argument("--data", required=True, help="folder with one subfolder per object")
    parser.add_argument("--out", default="results/geometry_mit")
    parser.add_argument("--edge-threshold", type=float, default=0.12,
                        help="geometry classifier's edge threshold (lower than Retinex's 0.3, "
                             "since a confirmed-flat surface needs less evidence)")
    parser.add_argument("--normal-threshold", type=float, default=0.05,
                        help="how flat a normal has to be to count as 'no geometric explanation'")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = ["object,retinex,color_cue,combined,grid_tile,color_geometry"]
    print(f"{'object':<10} {'Retinex':>9} {'Color':>9} {'Combined':>9} {'Grid-tile':>9} {'Geometry':>9}")

    for name in OBJECTS:
        obj = load_mit_object(args.data, name)
        image, mask = obj["image"], obj["mask"]

        print(f"  [{name}] estimating depth with Depth Anything...")
        depth = load_depth_map(image, mode="real")

        results, scores = {}, {}
        for label, method in [("Retinex", "retinex"), ("Color cue", "color"), ("Combined", "combined")]:
            s, r, e = decompose(image, method, mask)
            results[label] = (s, r, e)
            scores[label] = lmse(obj["shading"], obj["reflectance"], s, r, mask)

        s, r, e = decompose_grid_tile(image, mask, tile_size=32)
        results["Grid-tile"] = (s, r, e)
        scores["Grid-tile"] = lmse(obj["shading"], obj["reflectance"], s, r, mask)

        s, r, e = decompose_with_geometry(image, depth, mask,
                                   edge_threshold=args.edge_threshold,      # 0.12, geometry-specific
                                   normal_threshold=args.normal_threshold,
                                   retinex_threshold=0.3)                   # matches Combined's safety net
        results["Color+Geometry"] = (s, r, e)
        scores["Color+Geometry"] = lmse(obj["shading"], obj["reflectance"], s, r, mask)

        print(f"{name:<10} " + " ".join(f"{scores[k]:>9.4f}" for k in
              ["Retinex", "Color cue", "Combined", "Grid-tile", "Color+Geometry"]))
        rows.append(f"{name}," + ",".join(f"{scores[k]:.4f}" for k in
                     ["Retinex", "Color cue", "Combined", "Grid-tile", "Color+Geometry"]))

        save_comparison(os.path.join(args.out, f"{name}.png"), image, results, mask,
                         obj["shading"], obj["reflectance"], title=name)

    with open(os.path.join(args.out, "lmse.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nSaved figures and lmse.csv to {args.out}/")


if __name__ == "__main__":
    main()