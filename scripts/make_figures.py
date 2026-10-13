"""Midterm figures for the MRI coverage-failure result (T2.7).

Three figures, all drawn from the table ``coverage_by_patient.py`` writes, so a figure and
the record can never disagree:

1. **coverage by operator** -- naive transfer against group-conditional calibration, every
   deployment operator, both directions (D-056), patient-clustered 95% intervals;
2. **gap against signed distance** -- the coverage gap on D-053's directional axis, where
   ascending and descending share one x-axis and the asymmetry is visible;
3. **coverage against flag area** -- why a coverage of 1.000 is not a success when it is
   bought by flagging almost every region.

Images go to ``figures/``, which is gitignored: the figures are regenerated from this
script, never committed.

    python scripts/coverage_by_patient.py runs/<run_id>_by_patient \
        --low mri.random1d.R2.c0.08.s0 --high mri.random1d.R8.c0.08.s0 \
        --out runs/w05_analysis/coverage_by_patient.csv
    python scripts/make_figures.py runs/w05_analysis/coverage_by_patient.csv \
        --bank runs/<run_id>_by_patient
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from caliber.operators.distance import compute_distance  # noqa: E402
from caliber.operators.registry import build_operator  # noqa: E402

# Validated with the dataviz skill's validator (light, --pairs all): CVD dE 24.7,
# normal-vision dE 33.6, both >= 3:1 on the surface. Colour follows the method in every
# figure, so orange always means naive transfer and blue always group-conditional.
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
METHOD = {
    "naive": ("naive transfer", "#eb6834"),
    "group": ("group-conditional", "#2a78d6"),
}
FAMILIES = ["random1d", "equispaced1d", "variable_density2d"]
FAMILY_MARKER = {"random1d": "o", "equispaced1d": "s", "variable_density2d": "^"}
ACCELERATIONS = [2, 4, 8]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK_2,
})


def parse_operator(oid: str) -> tuple[str, int]:
    """``mri.equispaced1d.R4.c0.08.s0`` -> ("equispaced1d", 4)."""
    parts = oid.split(".")
    return parts[1], int(float(parts[2][1:]))


def signed_distances(bank: Path, table: pd.DataFrame) -> dict[tuple[str, str], float]:
    """D-053's ``null_space_growth`` from each calibration operator to each deployment one."""
    manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    ops = {
        row["operator_id"]: build_operator({"modality": row["modality"], **row["params"]},
                                           tuple(row["image_shape"]))
        for row in manifest["operators"]
    }
    out = {}
    for calib, deploy in table[["calibrated_at", "deployed_at"]].drop_duplicates().itertuples(
            index=False):
        out[(calib, deploy)] = float(compute_distance(ops[calib], ops[deploy],
                                                      "null_space_growth", normalise="ambient"))
    return out


def y_floor(t: pd.DataFrame) -> float:
    """The lowest y on a coverage axis: 0.3, or lower if an interval reaches below it.

    A fixed floor once clipped the one interval the result rests on (the candidacy-gate
    label puts its lower end at 0.15), and a clipped interval reads as a tighter one.
    """
    return min(0.3, math.floor(float(t["lo"].min()) * 10) / 10)


def nominal_line(ax, nominal: float, label: bool = True) -> None:
    ax.axhline(nominal, color=MUTED, linewidth=1.0, zorder=1)
    if label:
        ax.text(1.0, nominal, f" nominal {nominal:.2f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", color=INK_2, fontsize=8, clip_on=False)


def point(ax, x, y, lo, hi, color, marker="o"):
    ax.plot([x, x], [lo, hi], color=color, linewidth=2.0, solid_capstyle="round", zorder=2)
    ax.plot([x], [y], marker=marker, markersize=7, color=color, markeredgecolor=SURFACE,
            markeredgewidth=1.6, linestyle="none", zorder=3)


def fig_coverage_by_operator(t: pd.DataFrame, nominal: float, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, (direction, part) in zip(axes, t.groupby("direction", sort=False), strict=True):
        calib = part["calibrated_at"].iloc[0]
        cfam, cacc = parse_operator(calib)
        ticks, labels = [], []
        for fi, fam in enumerate(FAMILIES):
            for ai, acc in enumerate(ACCELERATIONS):
                x = fi * 3.6 + ai
                ticks.append(x)
                labels.append(f"R{acc}" + ("\n(calib.)" if (fam, acc) == (cfam, cacc) else ""))
                for method, dx in (("naive", -0.16), ("group", 0.16)):
                    r = part[(part["method"] == method)
                             & (part["deployed_at"].map(parse_operator) == (fam, acc))].iloc[0]
                    point(ax, x + dx, r["coverage"], r["lo"], r["hi"], METHOD[method][1])
            ax.text(fi * 3.6 + 1, -0.2, fam, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", color=INK_2, fontsize=8.5)
        ax.set_xticks(ticks, labels)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(y_floor(t), 1.04)
        nominal_line(ax, nominal, label=ax is axes[-1])
        title = ("calibrated at random1d R2, deployed upward" if direction == "ascending"
                 else "calibrated at random1d R8, deployed downward")
        ax.set_title(title, loc="left", fontsize=9.5, pad=8)
    axes[0].set_ylabel("coverage of hallucinated regions")
    worst = t[t["method"] == "naive"].nsmallest(1, "coverage").iloc[0]
    wax = axes[[d for d, _ in t.groupby("direction", sort=False)].index(worst["direction"])]
    wf, wa = parse_operator(worst["deployed_at"])
    wx = FAMILIES.index(wf) * 3.6 + ACCELERATIONS.index(wa) - 0.16
    wax.annotate(f"{worst['coverage']:.3f}", (wx, worst["coverage"]), xytext=(-30, -4),
                 textcoords="offset points", ha="right", va="center", color=INK, fontsize=8.5,
                 arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    handles = [Line2D([], [], color=c, marker="o", markersize=7, markeredgecolor=SURFACE,
                      linewidth=2.0, label=lab) for lab, c in METHOD.values()]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Coverage under operator shift, MRI (fastMRI knee, 44 test patients, "
                 "95% intervals by patient)", x=0.01, ha="left", y=1.07, fontsize=10.5,
                 color=INK)
    fig.subplots_adjust(bottom=0.22, wspace=0.08, right=0.92)
    return save(fig, out / "fig1_coverage_by_operator")


def fig_gap_vs_signed_distance(t: pd.DataFrame, dist: dict, nominal: float, out: Path) -> Path:
    # Naive transfer only: this is D-053's point, and Figure 1 already carries the
    # group-conditional contrast. Both directions meet at x = 0 (matched acceleration has
    # zero signed growth across families), so family gets a marker and an x-offset, and
    # direction gets filled (calibrated at R2) against open (calibrated at R8).
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    color = METHOD["naive"][1]
    jitter = {"random1d": -0.02, "equispaced1d": 0.0, "variable_density2d": 0.02}
    lows = t["calibrated_at"].drop_duplicates().tolist()
    naive = t[t["method"] == "naive"]
    for _, r in naive.iterrows():
        fam, _ = parse_operator(r["deployed_at"])
        ascending = r["calibrated_at"] == lows[0]
        # direction first (ascending left of its true x, descending right), then family
        x = dist[(r["calibrated_at"], r["deployed_at"])] + jitter[fam] + (
            -0.045 if ascending else 0.045)
        gap, glo, ghi = nominal - r["coverage"], nominal - r["hi"], nominal - r["lo"]
        ax.plot([x, x], [glo, ghi], color=color, linewidth=2.0, solid_capstyle="round",
                zorder=2)
        ax.plot([x], [gap], marker=FAMILY_MARKER[fam], markersize=7, linestyle="none",
                markerfacecolor=color if ascending else SURFACE, markeredgecolor=color,
                markeredgewidth=1.6, zorder=3)
    ax.axhline(0.0, color=MUTED, linewidth=1.0, zorder=1)
    ax.set_xlabel("signed operator distance, null_space_growth (D-053)\n"
                  r"deployed less aggressive  $\leftarrow$   0   $\rightarrow$  "
                  "deployed more aggressive")
    # mathtext: ASCII in the source (tests/test_console_encoding.py), glyphs in the figure
    ax.set_ylabel(r"coverage gap  (nominal $-$ coverage)")
    ax.text(-0.27, 0.46, r"$\uparrow$ misses hallucinations", color=INK_2, fontsize=8,
            ha="left")
    ax.text(0.16, -0.06, r"$\downarrow$ over-flags", color=INK_2, fontsize=8, ha="left")
    family_handles = [Line2D([], [], color=INK_2, marker=m, linestyle="none", markersize=6,
                             label=f) for f, m in FAMILY_MARKER.items()]
    dir_handles = [
        Line2D([], [], color=INK_2, marker="o", linestyle="none", markersize=6,
               label="calibrated at R2 (ascending)"),
        Line2D([], [], color=INK_2, marker="o", linestyle="none", markersize=6,
               markerfacecolor=SURFACE, label="calibrated at R8 (descending)"),
    ]
    ax.legend(handles=family_handles + dir_handles, loc="upper right", fontsize=8)
    ax.set_title("Naive transfer: the coverage gap depends on direction", loc="left",
                 fontsize=10.5, pad=10)
    fig.tight_layout()
    return save(fig, out / "fig2_gap_vs_signed_distance")


def fig_coverage_vs_flag_area(t: pd.DataFrame, nominal: float, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    floor = y_floor(t)
    ax.axhspan(floor, nominal, color=MUTED, alpha=0.08, zorder=0, linewidth=0)
    direction_marker = {"ascending": "o", "descending": "^"}
    for _, r in t.iterrows():
        point(ax, r["flag_area"], r["coverage"], r["lo"], r["hi"], METHOD[r["method"]][1],
              direction_marker[r["direction"]])
    nominal_line(ax, nominal)
    ax.set_xlim(0.0, 1.03)
    # Headroom above 1.0 holds the corner label clear of every interval; ticks stop at
    # 1.0 because coverage cannot exceed it.
    ax.set_ylim(floor, 1.12)
    ax.set_yticks([round(floor + 0.1 * i, 1) for i in range(round((1.0 - floor) * 10) + 1)])
    ax.set_xlabel("flag area (fraction of all regions flagged)")
    ax.set_ylabel("coverage of hallucinated regions")
    ax.text(0.015, floor + 0.03, "below nominal: the promise is broken", color=INK_2, fontsize=8,
            va="bottom")
    # Selected by what makes them notable -- coverage near 1 bought with most of the image --
    # not by a fixed area cutoff, which under one label caught one of three such cells.
    corner = t[(t["method"] == "naive") & (t["coverage"] >= 0.95) & (t["flag_area"] > 0.8)]
    if len(corner):
        # Placed in the empty upper-left and led to the corner, never inside the cluster.
        ax.annotate(f"naive transfer, valid but useless:\ncoverage ~1.0 by flagging "
                    f"{corner['flag_area'].min():.0%}-{corner['flag_area'].max():.0%} "
                    "of all regions",
                    (corner["flag_area"].min(), corner["coverage"].mean()),
                    xytext=(0.03, 1.105), textcoords="data", ha="left", va="top", color=INK,
                    fontsize=8, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    method_handles = [Line2D([], [], color=c, marker="o", linestyle="none", markersize=7,
                             markeredgecolor=SURFACE, label=lab) for lab, c in METHOD.values()]
    dir_handles = [Line2D([], [], color=INK_2, marker=m, linestyle="none", markersize=6,
                          label=f"{d}") for d, m in direction_marker.items()]
    ax.legend(handles=method_handles + dir_handles, loc="lower right", fontsize=8)
    ax.set_title("Coverage only means something next to flag area", loc="left",
                 fontsize=10.5, pad=10)
    fig.tight_layout()
    return save(fig, out / "fig3_coverage_vs_flag_area")


def save(fig, stem: Path) -> Path:
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("coverage_csv", help="the table coverage_by_patient.py --out writes")
    ap.add_argument("--bank", required=True, help="bank whose manifest lists the operators")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args(argv)

    t = pd.read_csv(args.coverage_csv)
    nominal = 1.0 - args.alpha
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dist = signed_distances(Path(args.bank), t)
    for path in (fig_coverage_by_operator(t, nominal, out),
                 fig_gap_vs_signed_distance(t, dist, nominal, out),
                 fig_coverage_vs_flag_area(t, nominal, out)):
        print(f"[figures] wrote {path} (+ .pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
