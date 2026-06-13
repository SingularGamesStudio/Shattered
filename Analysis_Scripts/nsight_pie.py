#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# =============================================================================
# Marker categories by XPBI solve step.
# Order matters: first matching prefix wins.
# =============================================================================
CATEGORY_PREFIX_MAP = {
    "Collision detection": [
        "XPBI.Query",
        "XPBI.PropagateVertexContacts",
        "XPBI.PropagateEdgeContacts",
        "XPBI.BuildOwnerFeatureField",
        "XPBI.BinBoundaryEdges",
        "XPBI.BuildBoundaryFeatures",
        "XPBI.BinBoundaryVertices",
        "XPBI.BuildFineContactStencils",
        "XPBI.ClearNodeCollisionRefAuxFine",
        "XPBI.ClearNodeCollisionRefWriteFine",
        "XPBI.ScatterFineContactRefs",
        "XPBI.ExclusiveScanNodeCollisionRefCountFine",
        "XPBI.BuildCoarseContactStencils",
        "XPBI.ClearNodeCollisionRefAuxCoarse",
        "XPBI.ClearNodeCollisionRefWriteCoarse",
        "XPBI.ScatterCoarseContactRefs",
        "XPBI.ExclusiveScanNodeCollisionRefCountCoarse",
        "XPBI.ClearCoarseContacts",
    ],
    "Neighbor graph coloring": [
        "XPBI.NeighborColoring.",
    ],
    "Pre-solve caching": [
        "XPBI.CacheHierarchicalStats",
        "XPBI.FinalizeHierarchicalStats",
        "XPBI.ClearHierarchicalStats",
        "XPBI.RebuildParentsAtLayer",
        "XPBI.Restrict",
        "XPBI.Prolongate",
        "XPBI.ClearRestrictedDeltaV",
        "XPBI.ApplyRestrictedDeltaVToActiveAndPrefix",
        "XPBI.RemoveRestrictedDeltaVFromActive",
        "XPBI.SmoothProlongatedFineVel",
        "XPBI.ClearState",
        "XPBI.ClearConvergenceDebugStats",
        "XPBI.ExternalForces",
    ],
    "Main solve - GS": [
        "XPBI.RelaxColored",
        "XPBI.ComputeCorrectionL",
        "XPBI.ResetCollisionLambda",
        "XPBI.RelaxColoredPersistentCoarse",
    ],
    "Main solve - JR": [
        "XPBI.JR.",
        "XPBI.CollisionJacobi.SavePrevAndClear.JR",
        "XPBI.CollisionJacobi.Apply.JR",
    ],
    "Collisions solve": [
        "XPBI.CollisionJacobi.AtomicSolve.JR",
        "XPBI.CollisionJacobi.AtomicSolve.GS",
        "XPBI.CollisionJacobi.SavePrevAndClear.GS",
        "XPBI.CollisionJacobi.Apply.GS",
    ],
    "Post-solve work + regularization": [
        "XPBI.ApplyXsph",
        "XPBI.CopyVelToPrev",
        "XPBI.SaveVelPrefix",
        "XPBI.PositionCorrection",
        "XPBI.CacheF0AndResetLambda",
        "XPBI.ClampVelocities",
        "XPBI.CommitDeformation",
        "XPBI.IntegratePositions",
        "XPBI.UpdateDtPositions",
    ],
    "DT maintenance": [
        "XPBI.UpdateDtPositionsMapped",
        "XPBI.DT.",
    ],
}


COLLISION_STEAL_FROM_GS = 0.0
COLLISION_STEAL_FROM_JR = 0.0

GS_CATEGORY_NAME = "Main solve - GS"
JR_CATEGORY_NAME = "Main solve - JR"
COLLISION_CONSTRAINTS_CATEGORY_NAME = "Collision constraints (estimated)"
UNCATEGORIZED_NAME = "Other"
MERGE_SMALL_SLICES_BELOW_PCT = 0.0

MARKER_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)%\s+(\d+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+(\S.*?)\s*$"
)
GRAND_TOTAL_RE = re.compile(r"Grand total\s*:\s*([0-9]*\.?[0-9]+)\s*ms")

LATEX_GRAYS = [
    "black!10",
    "black!22",
    "black!34",
    "black!46",
    "black!58",
    "black!70",
    "black!82",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build PNG/CSV/LaTeX pie charts from an Nsight marker summary text file."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(r"C:\Projects\Shattered\Analysis\final.txt"),
        help="Path to the summary text file",
    )
    p.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Pie chart PNG path",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Category CSV path",
    )
    p.add_argument(
        "--output-tex",
        type=Path,
        default=None,
        help="LaTeX figure snippet path (.tex)",
    )
    p.add_argument(
        "--title",
        default="XPBI GPU time by category",
        help="Chart title / LaTeX caption",
    )
    return p.parse_args()


def parse_summary_file(path: Path):
    text = path.read_text(encoding="utf-8")
    markers = []
    grand_total = None

    for line in text.splitlines():
        m_total = GRAND_TOTAL_RE.search(line)
        if m_total:
            grand_total = float(m_total.group(1))

        m = MARKER_ROW_RE.match(line)
        if not m:
            continue

        markers.append(
            {
                "rank": int(m.group(1)),
                "total_ms": float(m.group(2)),
                "share_pct": float(m.group(3)),
                "calls": int(m.group(4)),
                "avg_ms": float(m.group(5)),
                "max_ms": float(m.group(6)),
                "marker": m.group(7).strip(),
            }
        )

    if not markers:
        raise ValueError(f"No marker rows found in {path}")

    if grand_total is None:
        grand_total = sum(row["total_ms"] for row in markers)

    return markers, grand_total


def categorize_marker(marker: str) -> str:
    for category, prefixes in CATEGORY_PREFIX_MAP.items():
        for prefix in prefixes:
            if marker.startswith(prefix):
                return category
    print(marker)
    return UNCATEGORIZED_NAME


def aggregate_categories(markers, grand_total: float):
    totals = defaultdict(float)
    marker_lists = defaultdict(list)

    for row in markers:
        category = categorize_marker(row["marker"])
        totals[category] += row["total_ms"]
        marker_lists[category].append(row["marker"])

    gs_total = totals.get(GS_CATEGORY_NAME, 0.0)
    if gs_total > 0 and COLLISION_STEAL_FROM_GS > 0:
        totals[GS_CATEGORY_NAME] -= COLLISION_STEAL_FROM_GS
        totals[COLLISION_CONSTRAINTS_CATEGORY_NAME] += COLLISION_STEAL_FROM_GS

    jr_total = totals.get(JR_CATEGORY_NAME, 0.0)
    if jr_total > 0 and COLLISION_STEAL_FROM_JR > 0:
        totals[JR_CATEGORY_NAME] -= COLLISION_STEAL_FROM_JR
        totals[COLLISION_CONSTRAINTS_CATEGORY_NAME] += COLLISION_STEAL_FROM_JR

    rows = []
    for category, total_ms in totals.items():
        if total_ms <= 0:
            continue
        unique_markers = sorted(set(marker_lists.get(category, [])))
        rows.append(
            {
                "category": category,
                "total_ms": total_ms,
                "share_pct": (100.0 * total_ms / grand_total) if grand_total > 0 else 0.0,
                "marker_count": len(unique_markers),
                "markers": unique_markers,
            }
        )

    rows.sort(key=lambda x: x["total_ms"], reverse=True)

    if MERGE_SMALL_SLICES_BELOW_PCT > 0:
        kept = []
        other_total = 0.0
        other_markers = []
        for row in rows:
            if row["share_pct"] < MERGE_SMALL_SLICES_BELOW_PCT:
                other_total += row["total_ms"]
                other_markers.extend(row["markers"])
            else:
                kept.append(row)

        if other_total > 0:
            kept.append(
                {
                    "category": UNCATEGORIZED_NAME,
                    "total_ms": other_total,
                    "share_pct": 100.0 * other_total / grand_total if grand_total > 0 else 0.0,
                    "marker_count": len(sorted(set(other_markers))),
                    "markers": sorted(set(other_markers)),
                }
            )
        rows = sorted(kept, key=lambda x: x["total_ms"], reverse=True)

    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "total_ms", "share_pct", "marker_count", "markers"])
        for row in rows:
            writer.writerow(
                [
                    row["category"],
                    f"{row['total_ms']:.4f}",
                    f"{row['share_pct']:.2f}",
                    row["marker_count"],
                    " | ".join(row["markers"]),
                ]
            )


def autopct_with_value(values):
    total = sum(values)

    def inner(pct):
        value = total * pct / 100.0
        return f"{pct:.1f}%\n{value:.1f} ms"

    return inner


def write_pie(rows, title: str, output_png: Path):
    labels = [r["category"] for r in rows]
    values = [r["total_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(13, 8), facecolor="white")
    wedges, texts, autotexts = ax.pie(
        values,
        autopct=autopct_with_value(values),
        startangle=120,
        pctdistance=0.72,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    ax.axis("equal")
    ax.set_title(title)

    legend_labels = [
        f"{label} — {value:.2f} ms ({100.0 * value / sum(values):.2f}%)"
        for label, value in zip(labels, values)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_label_from_title(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
    if not slug:
        slug = "pie-chart"
    return f"fig:{slug}"


def write_latex_pie(rows, title: str, output_tex: Path):
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    def latex_escape(text: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(ch, ch) for ch in text)

    def latex_label_from_title(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
        return f"fig:{slug or 'pie-chart'}"

    patterns = [
        "horizontal lines",
        "vertical lines",
        "north east lines",
        "north west lines",
        "grid",
        "crosshatch",
        "dots",
        "crosshatch dots",
    ]

    radius = 3.2
    legend_x = 4.8
    legend_y_start = 2.8
    legend_step = 0.62

    total = sum(max(r["share_pct"], 0.0) for r in rows)
    if total <= 0:
        raise ValueError("No positive category shares to plot")

    lines = []
    lines.append("% Auto-generated LaTeX pie chart for black-and-white thesis printing")
    lines.append("% Add this once in the preamble:")
    lines.append("% \\usepackage{tikz}")
    lines.append("% \\usetikzlibrary{patterns}")
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \begin{tikzpicture}[x=1cm,y=1cm]")
    lines.append(
        rf"        \draw[line width=0.8pt] (0,0) circle ({radius});"
    )

    start_angle = 90.0
    for i, row in enumerate(rows):
        pct = 100.0 * row["total_ms"] / sum(r["total_ms"] for r in rows)
        angle = 360.0 * pct / 100.0
        end_angle = start_angle - angle
        pattern = patterns[i % len(patterns)]

        lines.append(
            rf"        \filldraw[fill=white, pattern={{{pattern}}}, pattern color=black, draw=black, line width=0.6pt] "
            rf"(0,0) -- ({start_angle:.4f}:{radius}) arc[start angle={start_angle:.4f}, end angle={end_angle:.4f}, radius={radius}] -- cycle;"
        )

        mid_angle = 0.5 * (start_angle + end_angle)
        if pct >= 6.0:
            label_r = radius * 0.62
            lines.append(
                rf"        \node[font=\scriptsize, fill=white, inner sep=1pt] at ({mid_angle:.4f}:{label_r:.4f}) {{{pct:.1f}\%}};"
            )

        start_angle = end_angle

    lines.append("")

    for i, row in enumerate(rows):
        pattern = patterns[i % len(patterns)]
        y = legend_y_start - i * legend_step
        cat = latex_escape(row["category"])
        pct = 100.0 * row["total_ms"] / sum(r["total_ms"] for r in rows)
        label = latex_escape(f"{cat} — {row['total_ms']:.1f} ms ({pct:.1f}\\%)")

        lines.append(
            rf"        \filldraw[fill=white, pattern={{{pattern}}}, pattern color=black, draw=black] "
            rf"({legend_x:.2f},{y:.2f}) rectangle ({legend_x + 0.45:.2f},{y + 0.32:.2f});"
        )
        lines.append(
            rf"        \node[anchor=west, font=\small] at ({legend_x + 0.60:.2f},{y + 0.16:.2f}) {{{label}}};"
        )

    lines.append(r"    \end{tikzpicture}")
    lines.append(rf"    \caption{{{latex_escape(title)}}}")
    lines.append(rf"    \label{{{latex_label_from_title(title)}}}")
    lines.append(r"\end{figure}")
    lines.append("")

    output_tex.write_text("\n".join(lines), encoding="utf-8")


def default_output_paths(input_path: Path):
    stem = input_path.stem
    return (
        input_path.with_name(f"{stem}_categories.csv"),
        input_path.with_name(f"{stem}_pie.png"),
        input_path.with_name(f"{stem}_pie_print.tex"),
    )


def main():
    args = parse_args()
    default_csv, default_png, default_tex = default_output_paths(args.input)
    output_csv = args.output_csv or default_csv
    output_png = args.output_png or default_png
    output_tex = args.output_tex or default_tex

    markers, grand_total = parse_summary_file(args.input)
    rows = aggregate_categories(markers, grand_total)

    write_csv(rows, output_csv)
    write_pie(rows, args.title, output_png)
    write_latex_pie(rows, args.title, output_tex)

    print(f"Parsed markers: {len(markers)}")
    print(f"Grand total: {grand_total:.4f} ms")
    print(f"CSV: {output_csv}")
    print(f"PNG: {output_png}")
    print(f"TEX: {output_tex}")


if __name__ == "__main__":
    main()