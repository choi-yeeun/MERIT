#!/usr/bin/env python3
"""Print overall and per-type accuracy for an EgoLifeQA result file."""

import argparse
import json
from collections import defaultdict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_file", help="*_final.json produced by evaluate_egolife.py")
    args = p.parse_args()

    with open(args.result_file, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not rows:
        raise SystemExit("Result file is empty.")

    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        bucket = by_type[r.get("type", "")]
        bucket[1] += 1
        bucket[0] += 1 if r.get("evaluate") else 0

    total_correct = sum(c for c, _ in by_type.values())
    total = sum(n for _, n in by_type.values())

    width = max(len(str(k)) for k in by_type) if by_type else 8
    print(f"{'type':<{width}}  {'acc':>7}  {'correct':>8}  {'total':>6}")
    print("-" * (width + 26))
    for qtype in sorted(by_type):
        correct, n = by_type[qtype]
        print(f"{qtype:<{width}}  {correct / n * 100:6.2f}%  {correct:>8}  {n:>6}")
    print("-" * (width + 26))
    print(
        f"{'OVERALL':<{width}}  {total_correct / total * 100:6.2f}%  "
        f"{total_correct:>8}  {total:>6}"
    )

    errored = sum(1 for r in rows if r.get("response") == "Error")
    if errored:
        print(f"\nWARNING: {errored} question(s) failed with an error and count as wrong.")


if __name__ == "__main__":
    main()
