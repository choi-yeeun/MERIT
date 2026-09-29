#!/usr/bin/env python3
"""Retrieval hit rate on EgoLifeQA."""

import argparse
import glob
import json
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logger = logging.getLogger("hit_rate_egolife")

CLIP_SECONDS = 30
RETRIEVED_SPAN = re.compile(
    r"\[DAY\s*(\d+)\s+(\d{1,2}):(\d{2}):(\d{2})\s*-\s*DAY\s*(\d+)\s+(\d{1,2}):(\d{2}):(\d{2})\]",
    re.IGNORECASE,
)


def to_timestamp(day: str, time_str: str) -> int:
    """``("DAY1", "13163014")`` -> ``113163014``, the DHHMMSSFF encoding."""
    return int(day.upper().replace("DAY", "") + time_str.zfill(8))


def to_seconds(timestamp: int) -> int:
    """DHHMMSSFF -> seconds. The FF (sub-second) field is dropped."""
    return (
        (timestamp // 100000000) * 86400
        + ((timestamp // 1000000) % 100) * 3600
        + ((timestamp // 10000) % 100) * 60
        + ((timestamp // 100) % 100)
    )


def parse_target_timestamps(target_time: Dict[str, Any]) -> List[int]:
    """Every evidence timestamp of one question. Three shapes occur:

        {"date": "DAY1", "time": "13163014"}                     one timestamp
        {"date": "DAY1", "time_list": ["13394908", "13512718"]}  several, one day
        {"date": "DAY1", "time": "11360904DAY2_10453312DAY3_11235811"}
            several, where each DAYn gives the day of the timestamp after it
    """
    date = target_time.get("date") or "DAY1"
    time_str = target_time.get("time") or ""
    out: List[int] = []

    if time_str:
        if "DAY" in time_str.upper():
            day = date
            for chunk in time_str.split("_"):
                parts = re.split(r"(DAY\d+)", chunk, maxsplit=1, flags=re.IGNORECASE)
                if parts[0]:
                    out.append(to_timestamp(day, parts[0]))
                if len(parts) >= 2:
                    day = parts[1]
        else:
            out.append(to_timestamp(date, time_str))
    elif target_time.get("time_list"):
        for time_str in target_time["time_list"]:
            out.append(to_timestamp(date, time_str))

    return out


def clip_spans(segments: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Clip boundaries as (start, end) DHHMMSSFF pairs."""
    spans = []
    for segment in segments:
        day = str(segment.get("date", "")).upper().replace("DAY", "")
        start, end = segment.get("start_time"), segment.get("end_time")
        if not day or start is None or end is None:
            continue
        spans.append(
            (int(day + str(start).zfill(8)), int(day + str(end).zfill(8)))
        )
    return spans


def locate(timestamp: int, spans: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], bool]:
    """The clip holding ``timestamp``, or the nearest one. Second element is
    True when the timestamp fell in a gap between clips."""
    nearest, best = None, None
    for start, end in spans:
        if start <= timestamp <= end:
            return (start, end), False
        distance = start - timestamp if timestamp < start else timestamp - end
        if best is None or distance < best:
            nearest, best = (start, end), distance
    return nearest, True


def retrieved_span(content: str) -> Optional[Tuple[int, int]]:
    match = RETRIEVED_SPAN.search(content)
    if not match:
        return None
    g = match.groups()
    return (
        int(f"{g[0]}{g[1].zfill(2)}{g[2]}{g[3]}00"),
        int(f"{g[4]}{g[5].zfill(2)}{g[6]}{g[7]}00"),
    )


def analyze_question(
    round_history: List[Dict[str, Any]],
    targets: List[Tuple[int, int]],
    margin_clips: int,
) -> Dict[str, Any]:
    margin = margin_clips * CLIP_SECONDS
    found = set()
    hits: List[Tuple[int, int]] = []           # (round_num, rank)
    first: Optional[Tuple[int, int]] = None

    for entry in round_history:
        round_num = entry.get("round_num", 0)
        contents = entry.get("retrieved_content") or []
        if not isinstance(contents, list):
            continue
        for rank, content in enumerate(contents, 1):
            span = retrieved_span(content)
            if not span:
                continue
            low = to_seconds(span[0]) - margin
            high = to_seconds(span[1]) + margin
            for index, (target_start, target_end) in enumerate(targets):
                if to_seconds(target_start) <= high and to_seconds(target_end) >= low:
                    hits.append((round_num, rank))
                    found.add(index)
                    if first is None:
                        first = (round_num, rank)

    return {
        "num_targets": len(targets),
        "found_targets": len(found),
        "hits": hits,
        "first_hit_round": first[0] if first else None,
        "first_hit_rank": first[1] if first else None,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("result_file", help="*_final.json written by eval/evaluate_egolife.py")
    p.add_argument("--subject", default="A1_JAKE")
    p.add_argument("--data-dir", default="data/EgoLife")
    p.add_argument("--qa-file", default=None,
                   help="Defaults to <data-dir>/EgoLifeQA/EgoLifeQA_<subject>.json")
    p.add_argument("--caption-dir", default=None,
                   help="Defaults to <data-dir>/captions/<subject>")
    p.add_argument("--segment-file", default=None,
                   help="Clip boundaries. Defaults to the key file in --caption-dir")
    p.add_argument(
        "--margin-clips", type=int, default=0,
        help="Expand each retrieved clip by this many clips on both sides. "
             "0 is the retrieved clip alone; 2 is MERIT's neighbor window "
             f"(±{2 * CLIP_SECONDS}s)",
    )
    p.add_argument("--output", default=None, help="Also write the numbers as JSON")
    return p


def resolve_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.qa_file is None:
        args.qa_file = os.path.join(
            args.data_dir, "EgoLifeQA", f"EgoLifeQA_{args.subject}.json"
        )
    if args.caption_dir is None:
        args.caption_dir = os.path.join(args.data_dir, "captions", args.subject)
    if args.segment_file is None:
        found = sorted(glob.glob(os.path.join(args.caption_dir, "*_30sec_4_key.json")))
        if not found:
            parser.error(
                f"No '*_30sec_4_key.json' in {args.caption_dir}; pass --segment-file"
            )
        args.segment_file = found[0]
    for path in (args.result_file, args.qa_file, args.segment_file):
        if not os.path.exists(path):
            parser.error(f"Not found: {path}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()
    resolve_paths(args, parser)

    with open(args.result_file) as f:
        results = json.load(f)
    with open(args.qa_file) as f:
        qa_by_id = {str(row["ID"]): row for row in json.load(f)}
    with open(args.segment_file) as f:
        spans = clip_spans(json.load(f))

    logger.info(f"Result file  : {args.result_file}")
    logger.info(f"QA file      : {args.qa_file}")
    logger.info(f"Segment file : {args.segment_file}")
    logger.info(f"Questions    : {len(results)}    clips: {len(spans)}")
    logger.info(
        f"Margin       : ±{args.margin_clips} clips "
        f"(±{args.margin_clips * CLIP_SECONDS}s)"
    )

    total = with_targets = any_hit = all_hit = gap_cases = 0
    first_round = defaultdict(int)
    rank_by_round: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    coverage = defaultdict(int)
    correct_with_hit = correct_without_hit = 0
    n_with_hit = n_without_hit = 0
    per_question = []

    for result in results:
        qid = str(result.get("ID", ""))
        if qid not in qa_by_id:
            logger.warning(f"ID {qid} is not in the QA file; skipped")
            continue
        total += 1

        timestamps = parse_target_timestamps(qa_by_id[qid].get("target_time") or {})
        if not timestamps:
            continue
        with_targets += 1

        targets, in_gap = [], False
        for timestamp in timestamps:
            span, gap = locate(timestamp, spans)
            if span:
                targets.append(span)
            in_gap |= gap
        if in_gap:
            gap_cases += 1

        analysis = analyze_question(
            result.get("round_history") or [], targets, args.margin_clips
        )
        correct = bool(result.get("evaluate"))

        if analysis["found_targets"]:
            any_hit += 1
            n_with_hit += 1
            correct_with_hit += correct
            if analysis["found_targets"] == analysis["num_targets"]:
                all_hit += 1
            first_round[analysis["first_hit_round"]] += 1
            for round_num, rank in analysis["hits"]:
                rank_by_round[round_num][rank] += 1
        else:
            n_without_hit += 1
            correct_without_hit += correct

        if analysis["num_targets"]:
            ratio = analysis["found_targets"] / analysis["num_targets"]
            coverage[f"{int(ratio * 100)}%"] += 1

        per_question.append({
            "ID": qid,
            "num_targets": analysis["num_targets"],
            "found_targets": analysis["found_targets"],
            "first_hit_round": analysis["first_hit_round"],
            "first_hit_rank": analysis["first_hit_rank"],
            "in_gap": in_gap,
            "correct": correct,
        })

    def pct(n: int, d: int) -> float:
        return n / d * 100 if d else 0.0

    print("\n" + "=" * 58)
    print("RETRIEVAL HIT RATE")
    print("=" * 58)
    print(f"\nQuestions analysed     : {total}")
    print(f"With a target time     : {with_targets}")
    print(f"Target in a clip gap   : {gap_cases}  (nearest clip used)")

    print("\n[Hit rate]")
    print(f"  any target found     : {any_hit}/{with_targets} ({pct(any_hit, with_targets):.1f}%)")
    print(f"  all targets found    : {all_hit}/{with_targets} ({pct(all_hit, with_targets):.1f}%)")
    print(f"  no target found      : {with_targets - any_hit}/{with_targets} "
          f"({pct(with_targets - any_hit, with_targets):.1f}%)")

    print("\n[Round of the first hit]")
    for round_num in sorted(first_round):
        n = first_round[round_num]
        print(f"  round {round_num}              : {n} ({pct(n, any_hit):.1f}%)")

    print("\n[Rank of every hit, by round]")
    for round_num in sorted(rank_by_round):
        ranks = rank_by_round[round_num]
        subtotal = sum(ranks.values())
        print(f"  round {round_num}:")
        for rank in sorted(ranks):
            print(f"    top{rank:<3}             : {ranks[rank]} ({pct(ranks[rank], subtotal):.1f}%)")

    print("\n[Targets covered]")
    for bucket in sorted(coverage, key=lambda b: int(b.rstrip("%"))):
        n = coverage[bucket]
        print(f"  {bucket:<21}: {n} ({pct(n, with_targets):.1f}%)")

    print("\n[Accuracy split by hit]")
    print(f"  with a hit           : {correct_with_hit}/{n_with_hit} "
          f"({pct(correct_with_hit, n_with_hit):.1f}%)")
    print(f"  without a hit        : {correct_without_hit}/{n_without_hit} "
          f"({pct(correct_without_hit, n_without_hit):.1f}%)")
    print("=" * 58)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "margin_clips": args.margin_clips,
                "questions": total,
                "with_targets": with_targets,
                "gap_cases": gap_cases,
                "any_hit": any_hit,
                "all_hit": all_hit,
                "hit_rate": pct(any_hit, with_targets),
                "all_hit_rate": pct(all_hit, with_targets),
                "first_hit_round": dict(first_round),
                "rank_by_round": {str(k): dict(v) for k, v in rank_by_round.items()},
                "coverage": dict(coverage),
                "accuracy_with_hit": pct(correct_with_hit, n_with_hit),
                "accuracy_without_hit": pct(correct_without_hit, n_without_hit),
                "per_question": per_question,
            }, f, indent=2)
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
