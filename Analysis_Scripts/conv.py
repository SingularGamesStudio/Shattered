#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


REQUIRED_COLUMNS = {"step", "in_step_iteration", "avg_abs_c"}
DISTINCT_COLORS = [
    "#E15759",
    "#59A14F",
    "#4E79A7",
    "#F28E2B",
    "#B07AA1",
    "#EDC948",
    "#76B7B2",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#D37295",
    "#8CD17D",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Scan a run folder for CSV files, keep only those with required convergence fields, "
            "and draw overlapping AbsC convergence trends for all valid runs."
        )
    )
    p.add_argument("--run_folder", type=Path, default=Path("."), help="Folder that contains run CSV files")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("absc_overlap.png"),
        help="Output image path",
    )
    p.add_argument(
        "--marker",
        type=int,
        default=None,
        help="Optional marker filter. Example: --marker 0",
    )
    p.add_argument(
        "--stop-tol",
        type=float,
        default=0.03,
        help="Stop when remaining improvement is <= this fraction of current AbsC (default: 3%%)",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Median smoothing window for each run trend",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Optional minimum count filter when the CSV has a count column",
    )
    p.add_argument(
        "--band-alpha",
        type=float,
        default=0.10,
        help="Transparency for the 25-75 percentile bands (default: 0.10)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under run_folder",
    )
    p.add_argument(
        "--title",
        default="Сходимость алгоритмов",
        help="Plot title",
    )
    p.add_argument(
        "--max-iteration",
        type=int,
        default=20,
        help="Maximum in-step iteration to include (default: 20)",
    )
    return p.parse_args()


def rolling_median(values, window):
    s = pd.Series(values)
    return s.rolling(window=max(1, int(window)), center=True, min_periods=1).median().to_numpy()


def remaining_gain_stop(curve, iterations, tol=0.03):
    curve = np.asarray(curve, dtype=float)
    iterations = np.asarray(iterations, dtype=int)

    valid = np.isfinite(curve)
    curve = curve[valid]
    iterations = iterations[valid]
    if curve.size == 0:
        return None

    eps = np.finfo(float).eps
    future_best = np.minimum.accumulate(curve[::-1])[::-1]
    remaining_gain = np.maximum(curve - future_best, 0.0) / np.maximum(curve, eps)
    idxs = np.where(remaining_gain <= tol)[0]
    if idxs.size:
        return int(iterations[idxs[0]])
    return int(iterations[int(np.argmin(curve))])


def prepare_curve(df, max_iteration=20):
    df = df[df["in_step_iteration"] <= max_iteration].copy()
    if df.empty:
        return None

    pivot = (
        df.pivot_table(
            index="step",
            columns="in_step_iteration",
            values="avg_abs_c",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    pivot = pivot.dropna(axis=1, how="all")
    if pivot.empty:
        return None

    values = pivot.to_numpy(dtype=float)
    iterations = pivot.columns.to_numpy(dtype=int)
    median_curve = np.nanmedian(values, axis=0)
    p25_curve = np.nanpercentile(values, 25, axis=0)
    p75_curve = np.nanpercentile(values, 75, axis=0)
    return iterations, median_curve, p25_curve, p75_curve, pivot.shape[0]


def positive_masked(*arrays):
    out = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float).copy()
        a[~np.isfinite(a)] = np.nan
        a[a <= 0.0] = np.nan
        out.append(a)
    return out


def load_run_curve(
    csv_path,
    marker=None,
    min_count=1,
    smooth_window=3,
    stop_tol=0.03,
    max_iteration=20,
):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if not REQUIRED_COLUMNS.issubset(df.columns):
        return None

    df = df.copy()
    if marker is not None and "marker" in df.columns:
        df = df[df["marker"] == marker]
    if "count" in df.columns:
        df = df[df["count"] >= min_count]
    df = df[np.isfinite(df["avg_abs_c"])].copy()
    if df.empty:
        return None

    prepared = prepare_curve(df, max_iteration=max_iteration)
    if prepared is None:
        return None

    iterations, median_curve, p25_curve, p75_curve, step_count = prepared
    smooth_curve = rolling_median(median_curve, smooth_window)
    stop_iter = remaining_gain_stop(smooth_curve, iterations, tol=stop_tol)
    best_iter = int(iterations[int(np.nanargmin(smooth_curve))])
    best_val = float(np.nanmin(smooth_curve))

    stop_idx = int(np.where(iterations == stop_iter)[0][0])
    stop_val = float(smooth_curve[stop_idx])
    extra_gain = 0.0 if stop_val <= 0 else max((stop_val - best_val) / stop_val, 0.0)

    return {
        "name": csv_path.stem,
        "path": str(csv_path.resolve()),
        "iterations": iterations,
        "median_curve": median_curve,
        "p25_curve": p25_curve,
        "p75_curve": p75_curve,
        "smooth_curve": smooth_curve,
        "step_count": int(step_count),
        "recommended_iter": int(stop_iter),
        "best_iter": int(best_iter),
        "extra_gain_to_best": float(extra_gain),
    }

def write_inline_pgfplots(output_base: Path, runs):
    """
    Simple, robust pgfplots snippet:
    - monochrome
    - translucent percentile bands
    - legend shows line styles correctly
    """
    if not runs:
        return

    tex_path = output_base.with_suffix(".pgfplots.tex")

    all_iters = sorted({int(i) for r in runs for i in r["iterations"]})
    max_iter = max(all_iters) if all_iters else 0

    line_styles = [
        "solid",
        "densely dashed",
        "densely dotted",
        "dashdotted",
        "densely dashdotted",
        "loosely dashed",
        "loosely dotted",
    ]

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by absc_overlap.py\n")
        f.write("% Requires: pgfplots + fillbetween\n\n")

        f.write("\\begin{tikzpicture}\n")
        f.write("  \\begin{axis}[\n")
        f.write("    width=0.78\\linewidth,\n")
        f.write("    height=0.52\\linewidth,\n")
        f.write("    xlabel={Внутришаговая итерация},\n")
        f.write("    ylabel={$\\overline{C}_{\\mathrm{XPBI}},\\ \\sqrt{\\text{Па}}$},\n")
        f.write("    xlabel near ticks,\n")
        f.write("    ylabel near ticks,\n")
        f.write("    ymode=log,\n")
        f.write("    xmin=1,\n")
        f.write(f"    xmax={max_iter},\n")
        f.write("    xtick={1,...,%d},\n" % max_iter)
        f.write("    scaled ticks=false,\n")
        f.write("    tick align=outside,\n")
        f.write("    tick style={black, thin},\n")
        f.write("    axis line style={black, semithick},\n")
        f.write("    xlabel style={font=\\small},\n")
        f.write("    ylabel style={font=\\small},\n")
        f.write("    tick label style={font=\\small},\n")
        f.write("    grid=both,\n")
        f.write("    major grid style={draw=gray!35, line width=0.3pt},\n")
        f.write("    minor grid style={draw=gray!20, line width=0.2pt},\n")
        f.write("    legend style={\n")
        f.write("      font=\\small,\n")
        f.write("      draw=black,\n")
        f.write("      fill=white,\n")
        f.write("      at={(0.98,0.98)},\n")
        f.write("      anchor=north east\n")
        f.write("    },\n")
        f.write("    legend cell align=left,\n")
        f.write("  ]\n\n")

        for idx, run in enumerate(runs):
            name = run["name"]
            line_style = line_styles[idx % len(line_styles)]

            iters = np.asarray(run["iterations"], dtype=int)
            median = np.asarray(run["median_curve"], dtype=float)
            p25 = np.asarray(run["p25_curve"], dtype=float)
            p75 = np.asarray(run["p75_curve"], dtype=float)

            band_mask = (
                np.isfinite(p25)
                & np.isfinite(p75)
                & (p75 >= p25)
                & (p25 > 0.0)
                & (p75 > 0.0)
            )
            median_mask = np.isfinite(median) & (median > 0.0)

            if not np.any(median_mask):
                continue

            safe_name = "".join(c if c.isalnum() else "_" for c in name)
            lower_name = f"p25_{safe_name}"
            upper_name = f"p75_{safe_name}"

            if np.any(band_mask):
                f.write(f"    % lower percentile path for {name}\n")
                f.write(
                    f"    \\addplot[name path={lower_name}, draw=none, forget plot] coordinates {{\n"
                )
                for x, y in zip(iters[band_mask], p25[band_mask]):
                    f.write(f"      ({int(x)}, {y:.6g})\n")
                f.write("    };\n\n")

                f.write(f"    % upper percentile path for {name}\n")
                f.write(
                    f"    \\addplot[name path={upper_name}, draw=none, forget plot] coordinates {{\n"
                )
                for x, y in zip(iters[band_mask], p75[band_mask]):
                    f.write(f"      ({int(x)}, {y:.6g})\n")
                f.write("    };\n\n")

                f.write(f"    % 25--75 percentile band for {name}\n")
                f.write(
                    "    \\addplot[draw=none, fill=gray, fill opacity=0.18, forget plot] "
                    f"fill between[of={lower_name} and {upper_name}];\n\n"
                )

            f.write(f"    % median curve for {name}\n")
            f.write(
                f"    \\addplot[black, semithick, {line_style}] coordinates {{\n"
            )
            for x, y in zip(iters[median_mask], median[median_mask]):
                f.write(f"      ({int(x)}, {y:.6g})\n")
            f.write("    };\n")
            f.write(f"    \\addlegendentry{{{name}}}\n\n")

        f.write("  \\end{axis}\n")
        f.write("\\end{tikzpicture}\n")

def main():
    args = parse_args()
    run_folder = args.run_folder
    if not run_folder.exists() or not run_folder.is_dir():
        raise ValueError(f"Run folder does not exist or is not a directory: {run_folder}")

    pattern_iter = run_folder.rglob("*.csv") if args.recursive else run_folder.glob("*.csv")
    csv_paths = sorted(pattern_iter)
    if not csv_paths:
        raise ValueError(f"No CSV files found under: {run_folder}")

    runs = []
    for csv_path in csv_paths:
        run = load_run_curve(
            csv_path,
            marker=args.marker,
            min_count=args.min_count,
            smooth_window=args.smooth_window,
            stop_tol=args.stop_tol,
            max_iteration=args.max_iteration,
        )
        if run is not None:
            runs.append(run)

    if not runs:
        raise ValueError("No CSV files with required fields produced usable convergence curves.")

    runs.sort(key=lambda r: r["name"].lower())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)

    colors = [DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i in range(len(runs))]

    summary_rows = []

    for run, color in zip(runs, colors):
        iterations = run["iterations"]
        median_curve = run["median_curve"]
        p25_curve = run["p25_curve"]
        p75_curve = run["p75_curve"]
        smooth_curve = run["median_curve"]
        rec_iter = run["recommended_iter"]

        median_curve, p25_curve, p75_curve, smooth_curve = positive_masked(
            median_curve, p25_curve, p75_curve, smooth_curve
        )

        band_mask = np.isfinite(p25_curve) & np.isfinite(p75_curve) & (p75_curve >= p25_curve)
        line_mask = np.isfinite(smooth_curve)
        median_mask = np.isfinite(median_curve)

        if not np.any(line_mask):
            continue

        rec_matches = np.where(iterations == rec_iter)[0]
        if rec_matches.size == 0:
            continue
        rec_idx = int(rec_matches[0])
        rec_val = float(smooth_curve[rec_idx]) if np.isfinite(smooth_curve[rec_idx]) else np.nan

        if np.any(band_mask):
            ax.fill_between(
                iterations,
                p25_curve,
                p75_curve,
                where=band_mask,
                interpolate=False,
                color=color,
                alpha=args.band_alpha,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                iterations,
                np.where(band_mask, p25_curve, np.nan),
                color=color,
                alpha=min(args.band_alpha * 1.8, 0.22),
                linewidth=0.8,
                zorder=1,
            )
            ax.plot(
                iterations,
                np.where(band_mask, p75_curve, np.nan),
                color=color,
                alpha=min(args.band_alpha * 1.8, 0.22),
                linewidth=0.8,
                zorder=1,
            )

        if np.any(median_mask):
            ax.plot(
                iterations,
                median_curve,
                color=color,
                alpha=0.16,
                linewidth=1.0,
                zorder=2,
            )

        ax.plot(
            iterations,
            smooth_curve,
            color=color,
            linewidth=3.0,
            label=run["name"],
            zorder=3,
        )

        if np.isfinite(rec_val):
            ax.scatter(
                [rec_iter],
                [rec_val],
                color=color,
                s=36,
                zorder=4,
            )

        summary_rows.append(
            {
                "run": run["name"],
                "csv_path": run["path"],
                "steps": run["step_count"],
                "recommended_iter": run["recommended_iter"],
                "best_iter": run["best_iter"],
                "extra_gain_to_best": run["extra_gain_to_best"],
            }
        )

    ax.set_title(args.title)
    ax.set_xlabel("Число итераций")
    ax.set_ylabel("Среднее значение функции ограничения")
    ax.set_yscale("log")
    ax.set_xlim(1, args.max_iteration)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="both", alpha=0.25)

    if len(runs) <= 12:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    else:
        ax.legend(loc="upper right", ncol=2, fontsize=9, frameon=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = args.output.with_suffix(".summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    notes_path = args.output.with_suffix(".summary.txt")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(f"valid_runs={len(runs)}\n")
        f.write(f"source_folder={run_folder.resolve()}\n")
        f.write(f"recursive={args.recursive}\n")
        f.write(f"stop_tol={args.stop_tol}\n")
        f.write(f"marker={args.marker}\n")
        f.write(f"band_alpha={args.band_alpha}\n")
        f.write(f"max_iteration={args.max_iteration}\n")

    write_inline_pgfplots(args.output, runs)


if __name__ == "__main__":
    main()