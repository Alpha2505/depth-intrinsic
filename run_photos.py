# Geethanjali Ravichandra
# November 1, 2026
# CS 7170 Advanced Perception - Project 2
#
# run_photos.py
# Runs both methods on my own photos (no ground truth here). Saves a
# comparison figure and the separate reflectance/shading images.
#
# Run: python run_photos.py photos/
#  or: python run_photos.py photos/a.jpg photos/b.jpg

import argparse
import os

from intrinsic import decompose
from io_utils import load_photo, save_comparison, save_image, to_display

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_paths(inputs):
    """Turn a mix of files and folders into a list of image files."""
    paths = []
    for p in inputs:
        if os.path.isdir(p):
            paths += sorted(os.path.join(p, f) for f in os.listdir(p)
                            if f.lower().endswith(EXTENSIONS))
        else:
            paths.append(p)
    return paths


def main():
    """Decompose each photo with both methods and save everything."""
    parser = argparse.ArgumentParser(description="Decompose your own photos")
    parser.add_argument("inputs", nargs="+", help="image files or folders")
    parser.add_argument("--out", default="results/photos")
    parser.add_argument("--retinex-threshold", type=float, default=0.3)
    parser.add_argument("--color-threshold", type=float, default=1.0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for path in collect_paths(args.inputs):
        name = os.path.splitext(os.path.basename(path))[0]
        image = load_photo(path)
        results = {
            "Retinex": decompose(image, "retinex", retinex_threshold=args.retinex_threshold),
            "Color cue": decompose(image, "color", color_threshold=args.color_threshold),
        }
        save_comparison(os.path.join(args.out, f"{name}_comparison.png"), image,
                        results, title=name)
        for label, (s, r, _) in results.items():
            tag = label.lower().replace(" ", "_")
            save_image(os.path.join(args.out, f"{name}_{tag}_reflectance.png"), to_display(r))
            save_image(os.path.join(args.out, f"{name}_{tag}_shading.png"), to_display(s))
        print(f"Done: {path}")
    print(f"Results saved to {args.out}/")


if __name__ == "__main__":
    main()
