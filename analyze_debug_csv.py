#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


COLOR_SUFFIX_RE = re.compile(r"\.C\d+$")
KNOWN_WRAPPERS = {"XPBI Async Batch"}


def parse_gpu_ms(raw: str) -> Optional[float]:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text == "-":
        return None
    if text.startswith("Δ") or text.startswith("Σ"):
        text = text[1:].strip()
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def find_columns(csv_path: Path) -> Tuple[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        fieldnames = [name.strip() for name in reader.fieldnames]

    lower_to_original = {name.lower(): name for name in fieldnames}

    desc_col = None
    for candidate in ["description", "event", "name"]:
        if candidate in lower_to_original:
            desc_col = lower_to_original[candidate]
            break

    gpu_col = None
    for candidate in ["gpu ms", "gpu time", "gpu duration", "duration (ms)", "duration"]:
        if candidate in lower_to_original:
            gpu_col = lower_to_original[candidate]
            break

    if desc_col is None:
        raise ValueError(f"Could not find description/event column. Headers: {fieldnames}")
    if gpu_col is None:
        raise ValueError(f"Could not find GPU timing column. Headers: {fieldnames}")

    return desc_col, gpu_col


def is_marker_event(description: str, marker_prefix: str) -> bool:
    if not description:
        return False
    desc = description.strip()
    if not desc:
        return False
    if desc.startswith("//"):
        return False
    if desc.startswith("void "):
        return False
    if marker_prefix and not desc.startswith(marker_prefix):
        return False
    return True


def normalize_marker_name(description: str) -> str:
    return COLOR_SUFFIX_RE.sub("", description)


def is_wrapper_marker(description: str) -> bool:
    if description in KNOWN_WRAPPERS:
        return True
    return description.endswith(" Async Batch")


def summarize(csv_path: Path, marker_prefix: str, top: int) -> int:
    desc_col, gpu_col = find_columns(csv_path)

    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    skipped_wrappers = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            desc = (row.get(desc_col) or "").strip()
            if not is_marker_event(desc, marker_prefix):
                continue

            if is_wrapper_marker(desc):
                skipped_wrappers += 1
                continue

            gpu_ms = parse_gpu_ms(row.get(gpu_col, ""))
            if gpu_ms is None:
                continue

            key = normalize_marker_name(desc)

            totals[key] += gpu_ms
            counts[key] += 1
            mins[key] = gpu_ms if key not in mins else min(mins[key], gpu_ms)
            maxs[key] = gpu_ms if key not in maxs else max(maxs[key], gpu_ms)

    if not totals:
        print(f"No marker events found for prefix '{marker_prefix}'")
        return 1

    rows: List[Tuple[str, int, float, float, float]] = []
    for desc, total_ms in totals.items():
        count = counts[desc]
        avg_ms = total_ms / count if count else 0.0
        rows.append((desc, count, total_ms, avg_ms, maxs[desc]))

    rows.sort(key=lambda r: r[2], reverse=True)
    grand_total = sum(r[2] for r in rows)
    total_markers = sum(r[1] for r in rows)

    print("=== Nsight Marker Summary (GPU ms) ===")
    print(f"Source file  : {csv_path}")
    print(f"Marker prefix: {marker_prefix}")
    print(f"Unique markers: {len(rows)}")
    print(f"Total calls   : {total_markers}")
    print(f"Grand total   : {grand_total:.4f} ms")
    print(f"Wrappers skipped: {skipped_wrappers}")
    print("Color grouping : trailing .C{N} merged into a single marker")
    print()

    print(f"Top {max(1, top)} markers by total GPU time")
    print("-" * 132)
    print(f"{'#':>3}  {'Total (ms)':>12}  {'Share':>7}  {'Calls':>7}  {'Avg (ms)':>10}  {'Max (ms)':>10}  Marker")
    print("-" * 132)
    for idx, (desc, count, total_ms, avg_ms, max_ms) in enumerate(rows[:max(1, top)], start=1):
        share = (100.0 * total_ms / grand_total) if grand_total > 0 else 0.0
        print(f"{idx:>3}  {total_ms:>12.4f}  {share:>6.2f}%  {count:>7}  {avg_ms:>10.4f}  {max_ms:>10.4f}  {desc}")
    print("-" * 132)

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize total GPU time by Nsight marker event from CSV")
    parser.add_argument("csv_path", nargs="?", default="debug.csv", help="Path to Nsight CSV (default: debug.csv)")
    parser.add_argument("--prefix", default="XPBI", help="Only include markers that start with this prefix")
    parser.add_argument("--top", type=int, default=50, help="Top N marker rows to print")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}")
        return 2
    return summarize(csv_path, marker_prefix=args.prefix, top=args.top)


if __name__ == "__main__":
    raise SystemExit(main())
