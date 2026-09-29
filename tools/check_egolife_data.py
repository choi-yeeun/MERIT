#!/usr/bin/env python3
"""Check that the EgoLife data layout is complete."""

import argparse
import json
import os
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data", "EgoLife"))
    p.add_argument("--subject", default="A1_JAKE")
    p.add_argument("--video-root", default=None)
    p.add_argument("--sample", type=int, default=50, help="Clips to probe on disk")
    args = p.parse_args()

    video_root = args.video_root or os.path.join(args.data_dir, "videos")
    failures = []

    qa_file = os.path.join(
        args.data_dir, "EgoLifeQA", f"EgoLifeQA_{args.subject}.json"
    )
    if os.path.exists(qa_file):
        with open(qa_file, encoding="utf-8") as f:
            qa = json.load(f)
        print(f"[ok]   QA file        {qa_file}  ({len(qa)} questions)")
    else:
        failures.append(f"QA file missing: {qa_file}")
        print(f"[FAIL] QA file        {qa_file}")

    caption_dir = os.path.join(args.data_dir, "captions", args.subject)
    key_file = os.path.join(caption_dir, f"{args.subject}_30sec_4_key.json")
    entries = []
    if os.path.exists(key_file):
        with open(key_file, encoding="utf-8") as f:
            entries = json.load(f)
        missing_keys = sum(1 for e in entries if not e.get("event_key"))
        print(f"[ok]   Key file       {key_file}  ({len(entries)} clips)")
        if missing_keys:
            failures.append(f"{missing_keys} clips have no event_key")
            print(f"[FAIL] {missing_keys} clips are missing keys")
    else:
        failures.append(f"Key file missing: {key_file}")
        print(f"[FAIL] Key file       {key_file}")

    if entries:
        if not os.path.isdir(video_root):
            failures.append(f"Video root missing: {video_root}")
            print(f"[FAIL] Video root     {video_root}")
        else:
            rng = random.Random(0)
            sample = rng.sample(entries, min(args.sample, len(entries)))
            missing = []
            for e in sample:
                rel = e.get("video_path") or ""
                parts = rel.replace("\\", "/").strip("/").split("/")
                if not os.path.exists(os.path.join(video_root, *parts[-3:])):
                    missing.append(rel)
            if missing:
                failures.append(
                    f"{len(missing)}/{len(sample)} sampled clips not found under "
                    f"{video_root}"
                )
                print(
                    f"[FAIL] Videos         {len(missing)}/{len(sample)} sampled "
                    f"clips missing under {video_root}"
                )
                for rel in missing[:5]:
                    print(f"           e.g. {os.path.join(video_root, rel)}")
            else:
                print(
                    f"[ok]   Videos         {len(sample)}/{len(sample)} sampled "
                    f"clips found under {video_root}"
                )

    print()
    if failures:
        print("NOT READY:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("Data layout looks good.")


if __name__ == "__main__":
    main()
