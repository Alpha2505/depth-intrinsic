# Geethanjali Ravichandra
# November 1, 2026
# CS 7170 Advanced Perception - Project 2
#
# evaluate_mit.py
# Runs Retinex and the color cue on the MIT Intrinsic Images dataset
# (Grosse et al. 2009), saves a figure per object, and writes a table of
# LMSE scores (with two trivial baselines). --sweep also tries different
# thresholds for each method and plots the mean error.
#
# Run: python evaluate_mit.py --data mit_data/MIT-intrinsic/data --sweep

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from intrinsic import decompose
from io_utils import list_mit_objects, load_mit_object, save_comparison
from metrics import lmse

RETINEX_THRESHOLDS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
COLOR_THRESHOLDS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8]


def baseline_scores(obj):
    """Scores for 'everything is shading' and 'everything is reflectance'."""
    image, mask = obj["image"], obj["mask"]
    gray = image.mean(axis=2)
    const = np.ones_like(gray)
    all_shading = lmse(obj["shading"], obj["reflectance"], gray, const, mask)
    all_refl = lmse(obj["shading"], obj["reflectance"], const, image, mask)
    return all_shading, all_refl


def evaluate(objects, args):
    """Run both methods on every object, save figures and the score table."""
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rows = []
    print(f"{'object':<10} {'All shad.':>10} {'All refl.':>10} {'Retinex':>10} {'Color cue':>10}")
    for obj in objects:
        results = {
            "Retinex": decompose(obj["image"], "retinex", obj["mask"],
                                 retinex_threshold=args.retinex_threshold),
            "Color cue": decompose(obj["image"], "color", obj["mask"],
                                   color_threshold=args.color_threshold),
        }
        scores = list(baseline_scores(obj))
        for s, r, _ in results.values():
            scores.append(lmse(obj["shading"], obj["reflectance"], s, r, obj["mask"]))
        rows.append((obj["name"], *scores))
        print(f"{obj['name']:<10} " + " ".join(f"{v:>10.4f}" for v in scores))

        save_comparison(os.path.join(fig_dir, f"{obj['name']}.png"), obj["image"],
                        results, obj["mask"], obj["shading"], obj["reflectance"],
                        title=obj["name"])

    means = np.mean([r[1:] for r in rows], axis=0)
    print(f"{'MEAN':<10} " + " ".join(f"{v:>10.4f}" for v in means))

    with open(os.path.join(args.out, "lmse.csv"), "w") as f:
        f.write("object,all_shading,all_reflectance,retinex,color_cue\n")
        for r in rows + [("MEAN", *means)]:
            f.write(r[0] + "," + ",".join(f"{v:.4f}" for v in r[1:]) + "\n")


def sweep(objects, args):
    """Mean LMSE for each threshold value, for both methods. Saves a plot and csv."""
    # run every threshold for each method and average over the objects
    curves = {}
    for method, thresholds, key in [("retinex", RETINEX_THRESHOLDS, "retinex_threshold"),
                                    ("color", COLOR_THRESHOLDS, "color_threshold")]:
        means = []
        for t in thresholds:
            errs = []
            for obj in objects:
                s, r, _ = decompose(obj["image"], method, obj["mask"], **{key: t})
                errs.append(lmse(obj["shading"], obj["reflectance"], s, r, obj["mask"]))
            means.append(np.mean(errs))
            print(f"{method:<8} threshold={t:<5} mean LMSE={means[-1]:.4f}")
        curves[method] = (thresholds, means)

    # save the numbers
    with open(os.path.join(args.out, "sweep.csv"), "w") as f:
        f.write("method,threshold,mean_lmse\n")
        for method, (ts, ms) in curves.items():
            for t, m in zip(ts, ms):
                f.write(f"{method},{t},{m:.4f}\n")

    # separate plots since the two thresholds are in different units
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    panels = [("retinex", "Retinex", "Log-gradient threshold", "#2a78d6"),
              ("color", "Color cue", "Chromaticity angle threshold (degrees)", "#eb6834")]
    for ax, (method, title, xlabel, color) in zip(axes, panels):
        ts, ms = curves[method]
        best = int(np.argmin(ms))
        ax.plot(ts, ms, color=color, linewidth=2, marker="o", markersize=5)
        ax.set_title(f"{title} (best {ms[best]:.4f} at {ts[best]})", fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Mean LMSE (lower is better)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "threshold_sweep.png"), dpi=150)
    plt.close(fig)


def main():
    """Load the dataset and run the evaluation (plus the sweep if asked)."""
    parser = argparse.ArgumentParser(description="Evaluate on the MIT Intrinsic Images dataset")
    parser.add_argument("--data", required=True, help="folder with one subfolder per object")
    parser.add_argument("--out", default="results/mit")
    parser.add_argument("--retinex-threshold", type=float, default=0.3)
    parser.add_argument("--color-threshold", type=float, default=1.0)
    parser.add_argument("--sweep", action="store_true", help="also run the threshold sweep")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = list_mit_objects(args.data)
    if not names:
        raise SystemExit(f"No objects with diffuse.png found in {args.data}")
    print(f"Loading {len(names)} objects from {args.data}")
    objects = [load_mit_object(args.data, n) for n in names]

    evaluate(objects, args)
    if args.sweep:
        sweep(objects, args)
    print(f"Results saved to {args.out}/")


if __name__ == "__main__":
    main()
