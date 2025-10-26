#!/usr/bin/env python3
"""
Copy benchmark candidates (as-is) from all data/*benchmark_candidates_*.json files
into a single file, filtering by a utility score threshold and optional max count.

No schema changes. No prompt reconstruction. Entries are copied verbatim.

Usage examples:
  python scripts/filter_benchmark_candidates.py --min_utility 0.7
  python scripts/filter_benchmark_candidates.py --min_utility 0.6 --max_examples 100
  python scripts/filter_benchmark_candidates.py --min_utility 0.65 --output data/my_candidates.json
"""

import argparse
import glob
import json
import os
import time
from typing import Any, Dict, List


def load_candidates_file(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        candidates = data.get("candidates", [])
        if isinstance(candidates, list):
            return candidates
        return []
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy benchmark candidates as-is with utility filtering")
    parser.add_argument("--min_utility", type=float, default=0.7, help="Minimum utility score to keep")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of candidates to include")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: data/copied_candidates_<ts>.json)")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pattern = os.path.join(data_dir, "benchmarks", "*benchmark_candidates_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No candidate files found via pattern: {pattern}")
        return

    print(f"Found {len(files)} candidate files")

    all_candidates: List[Dict[str, Any]] = []
    for path in files:
        cs = load_candidates_file(path)
        all_candidates.extend(cs)
        print(f"  + {os.path.basename(path)}: {len(cs)} candidates")

    print(f"Total candidates loaded: {len(all_candidates)}")

    # Filter by utility score; keep entries that have the field and meet threshold
    filtered = [c for c in all_candidates if c.get("utility_score", 0.0) >= args.min_utility]
    print(f"Candidates with utility >= {args.min_utility}: {len(filtered)}")

    # Sort by utility_score desc (stable, does not change entry contents)
    filtered.sort(key=lambda c: c.get("utility_score", 0.0), reverse=True)

    # Truncate if requested
    if args.max_examples is not None:
        filtered = filtered[: args.max_examples]
        print(f"Limited to top {args.max_examples} candidates")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output or os.path.join(data_dir, "benchmarks", f"benchmark_{args.min_utility}_{ts}.json")

    # Write out verbatim candidates
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "min_utility": args.min_utility,
        "num_candidates": len(filtered),
        "source_files": files,
        "utility_stats": {
            "min": min((c.get("utility_score", 0.0) for c in filtered), default=0.0),
            "max": max((c.get("utility_score", 0.0) for c in filtered), default=0.0),
            "avg": (sum(c.get("utility_score", 0.0) for c in filtered) / len(filtered)) if filtered else 0.0,
        },
    }

    with open(out_path, "w") as f:
        json.dump({"candidates": filtered, "metadata": meta}, f, indent=2)

    print(f"Wrote {len(filtered)} candidates to {out_path}")


if __name__ == "__main__":
    main()


