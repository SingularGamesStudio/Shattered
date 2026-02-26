#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RELAX_MARKER_RE = re.compile(r"XPBI\.RelaxColored\.L(?P<layer>\d+)\.C(?P<color>\d+)")


@dataclass
class RelaxEvent:
    layer: int
    color: int
    gpu_ms: float


def parse_gpu_ms(raw: str) -> Optional[float]:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text == "-":
        return None

    if text.startswith("Δ"):
        text = text[1:].strip()
    elif text.startswith("Σ"):
        text = text[1:].strip()

    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_relax_event(description: str, gpu_ms_raw: str) -> Optional[RelaxEvent]:
    if not description:
        return None

    if description.startswith("// End of"):
        return None

    marker_match = RELAX_MARKER_RE.search(description)
    if marker_match is None:
        return None

    gpu_ms = parse_gpu_ms(gpu_ms_raw)
    if gpu_ms is None:
        return None

    return RelaxEvent(
        layer=int(marker_match.group("layer")),
        color=int(marker_match.group("color")),
        gpu_ms=gpu_ms,
    )


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return float("nan")

    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]

    rank = (len(sorted_values) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def fmt_ms(value: float) -> str:
    return f"{value:.4f}"


def print_stats(values: List[float], label: str) -> None:
    if not values:
        print(f"{label}: no samples")
        return

    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)
    mean = total / n
    print(f"{label}: count={n}, total_ms={fmt_ms(total)}, mean_ms={fmt_ms(mean)}, min_ms={fmt_ms(sorted_values[0])}, max_ms={fmt_ms(sorted_values[-1])}")
    print(
        "  percentiles: "
        f"p50={fmt_ms(percentile(sorted_values, 50))}, "
        f"p90={fmt_ms(percentile(sorted_values, 90))}, "
        f"p95={fmt_ms(percentile(sorted_values, 95))}, "
        f"p99={fmt_ms(percentile(sorted_values, 99))}"
    )


def iter_relax_events(csv_path: Path, event_col: str, gpu_col: str) -> Iterable[RelaxEvent]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = parse_relax_event(row.get(event_col, ""), row.get(gpu_col, ""))
            if event is not None:
                yield event


def find_columns(csv_path: Path) -> Tuple[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        fieldnames = [name.strip() for name in reader.fieldnames]

    lower_to_original = {name.lower(): name for name in fieldnames}

    event_candidates = ["description", "event", "name"]
    gpu_candidates = ["gpu ms", "gpu time", "gpu duration", "duration (ms)", "duration"]

    event_col = None
    for candidate in event_candidates:
        if candidate in lower_to_original:
            event_col = lower_to_original[candidate]
            break

    gpu_col = None
    for candidate in gpu_candidates:
        if candidate in lower_to_original:
            gpu_col = lower_to_original[candidate]
            break

    if event_col is None:
        raise ValueError(f"Could not find event/description column. Headers: {fieldnames}")
    if gpu_col is None:
        raise ValueError(f"Could not find GPU duration column. Headers: {fieldnames}")

    return event_col, gpu_col


def summarize(events: List[RelaxEvent], fast_ms: float, slow_ms: float, top_n: int) -> None:
    all_ms = [e.gpu_ms for e in events]
    fast = [e for e in events if e.gpu_ms <= fast_ms]
    slow = [e for e in events if e.gpu_ms >= slow_ms]

    print("=== Relax Marker Timing Summary ===")
    print_stats(all_ms, "all_relax")
    print_stats([e.gpu_ms for e in fast], f"fast_relax (<= {fast_ms} ms)")
    print_stats([e.gpu_ms for e in slow], f"slow_relax (>= {slow_ms} ms)")
    print()

    layer_counts: Counter[int] = Counter(e.layer for e in events)
    fast_layer_counts: Counter[int] = Counter(e.layer for e in fast)
    slow_layer_counts: Counter[int] = Counter(e.layer for e in slow)

    print("=== Counts by Layer ===")
    all_layers = sorted(layer_counts)
    print("layer,total,fast,slow,fast_pct,slow_pct")
    for layer in all_layers:
        total_count = layer_counts[layer]
        fast_count = fast_layer_counts[layer]
        slow_count = slow_layer_counts[layer]
        fast_pct = (100.0 * fast_count / total_count) if total_count else 0.0
        slow_pct = (100.0 * slow_count / total_count) if total_count else 0.0
        print(f"{layer},{total_count},{fast_count},{slow_count},{fast_pct:.2f},{slow_pct:.2f}")
    print()

    layer_values: Dict[int, List[float]] = defaultdict(list)
    for e in events:
        layer_values[e.layer].append(e.gpu_ms)

    print("=== Timing Distribution by Layer ===")
    for layer in sorted(layer_values):
        print_stats(layer_values[layer], f"layer_{layer}")
    print()

    slow_by_layer_color: Counter[Tuple[int, int]] = Counter((e.layer, e.color) for e in slow)
    print(f"=== Top {top_n} Slow Layer/Color Buckets (>= {slow_ms} ms) ===")
    if not slow_by_layer_color:
        print("(none)")
    else:
        print("layer,color,count")
        for (layer, color), count in slow_by_layer_color.most_common(top_n):
            print(f"{layer},{color},{count}")
    print()

    all_sorted = sorted(events, key=lambda e: e.gpu_ms, reverse=True)
    print(f"=== Top {top_n} Slowest Relax Events (overall) ===")
    print("rank,layer,color,gpu_ms")
    for idx, event in enumerate(all_sorted[:top_n], start=1):
        print(f"{idx},{event.layer},{event.color},{event.gpu_ms:.4f}")

    slow_sorted = sorted(slow, key=lambda e: e.gpu_ms, reverse=True)
    print()
    print(f"=== Top {top_n} Slowest Relax Events (>= {slow_ms} ms) ===")
    if not slow_sorted:
        print("(none)")
    else:
        print("rank,layer,color,gpu_ms")
        for idx, event in enumerate(slow_sorted[:top_n], start=1):
            print(f"{idx},{event.layer},{event.color},{event.gpu_ms:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Nsight Graphics frame-event CSV for XPBI relax marker timing behavior."
    )
    parser.add_argument("csv_path", nargs="?", default="debug.csv", help="Path to Nsight CSV (default: debug.csv)")
    parser.add_argument("--fast-ms", type=float, default=0.07, help="Fast threshold in ms (inclusive)")
    parser.add_argument("--slow-ms", type=float, default=30.0, help="Slow threshold in ms (inclusive)")
    parser.add_argument("--top", type=int, default=20, help="Top-N lists for slow groups/events")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}")
        return 2

    if args.fast_ms < 0 or args.slow_ms < 0:
        print("ERROR: thresholds must be >= 0")
        return 2

    event_col, gpu_col = find_columns(csv_path)
    events = list(iter_relax_events(csv_path, event_col=event_col, gpu_col=gpu_col))

    if not events:
        print("No XPBI relax marker rows found. Expected Description/Event values like XPBI.RelaxColored.Lx.Cy")
        return 1

    summarize(events, fast_ms=args.fast_ms, slow_ms=args.slow_ms, top_n=max(1, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
