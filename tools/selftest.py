#!/usr/bin/env python3
"""GPU-free sanity check of a MERIT checkout."""

import glob
import importlib.util
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

failures = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"[{'ok' if cond else 'FAIL'}]   {label}{('  ' + detail) if detail else ''}")
    if not cond:
        failures.append(label)


def skip(label: str, why: str) -> None:
    print(f"[skip] {label}  {why}")


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# --------------------------------------------------------------- 1. imports
section("imports")
try:
    import merit
    from merit.llm import LLMModel, PromptTemplateManager
    from merit.embedding import EmbeddingModel
    from merit import MeritMemory
    from merit import MultiKeyMemory
    from merit import transform_timestamp

    check(f"merit {merit.__version__} imports", True,
          f"({LLMModel.__name__}, {EmbeddingModel.__name__} available)")
except Exception as e:  # noqa: BLE001
    check("merit imports", False, repr(e))
    print("\nInstall the package first: pip install -e .  (or bash setup_env.sh)")
    sys.exit(1)

# ------------------------------------------------------------- 2. templates
section("prompt templates")
ptm = PromptTemplateManager()
names = set(ptm.list_template_names())
expected = {
    "memory_retrieval_round_reasoning",
    "final_qa_egolife",
    "final_qa_video_benchmark",
    "neighbor_filter",
    "multikey_extract",
}
check("all templates discovered", expected <= names, str(sorted(names)))
for name in sorted(expected & names):
    rendered = ptm.render(name)
    check(f"  {name} renders", bool(rendered) and rendered[0]["role"] == "system")

# ------------------------------------------------------------- 3. timestamps
section("timestamps")
check("DHHMMSS00 -> DAY1 11:12:09", transform_timestamp("1111209300") == "DAY1 11:12:09",
      transform_timestamp("1111209300"))

# ----------------------------------------------------------- 4. video paths
section("video path re-rooting")
check("relative stored path",
      MultiKeyMemory._remap_video_path("A1_JAKE/DAY1/x.mp4", "/videos")
      == "/videos/A1_JAKE/DAY1/x.mp4")
check("absolute stored path from another machine",
      MultiKeyMemory._remap_video_path("/old/data/EgoLife/A1_JAKE/DAY1/x.mp4", "/videos")
      == "/videos/A1_JAKE/DAY1/x.mp4")
check("no --video-root leaves the stored path alone",
      MultiKeyMemory._remap_video_path("A1_JAKE/DAY1/x.mp4", None)
      == "A1_JAKE/DAY1/x.mp4")

# ------------------------------------------------------- 5. driver defaults
section("evaluate_egolife.py")
spec = importlib.util.spec_from_file_location(
    "_evaluate_egolife", os.path.join(REPO_ROOT, "eval", "evaluate_egolife.py")
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

parser = driver.build_parser()
args = parser.parse_args([])
check("default respond-model is gpt-5", args.respond_model == "gpt-5")
check("default top-k is 10", args.top_k == 10)
check("default max-rounds is 5", args.max_rounds == 5)
check("default image-max-tokens is 512", args.image_max_tokens == 512)
check("exp-name derives from the config",
      driver.default_exp_name(args) == "merit_gpt-5_top10",
      driver.default_exp_name(args))

cwd = os.getcwd()
os.chdir(REPO_ROOT)
try:
    # The captions and keys ship with the repo; the QA file and the videos do
    # not, so a checkout without the dataset still exercises everything else.
    key_files = glob.glob(
        os.path.join("data", "EgoLife", "captions", "A1_JAKE", "*_30sec_4_key.json")
    )
    check("shipped key file present", bool(key_files),
          key_files[0] if key_files else "data/EgoLife/captions/A1_JAKE/")

    if os.path.exists(args.qa_file or ""):
        driver.resolve_paths(args, parser)
    elif os.path.exists(
        os.path.join("data", "EgoLife", "EgoLifeQA", "EgoLifeQA_A1_JAKE.json")
    ):
        driver.resolve_paths(args, parser)
    else:
        skip("full path resolution", "dataset not mounted; run tools/check_egolife_data.py once it is")
        args.cache_dir = args.cache_dir or os.path.join(".cache", "multi_keys", args.subject)
    check("cache dir is subject-scoped",
          args.cache_dir == os.path.join(".cache", "multi_keys", "A1_JAKE"),
          args.cache_dir)
except SystemExit:
    check("data layout resolved", False, "run tools/check_egolife_data.py")
finally:
    os.chdir(cwd)

# -------------------------------------------------- 6. one dry question pass
section("prompt assembly (offline, no API calls)")


class _RecordingLLM:
    """Stands in for the respond model; replays canned decisions."""

    def __init__(self):
        self.calls = []
        self.responses = [
            '{"decision": "search", "search_query": "screwdriver"}',
            json.dumps({f"clip_{i}": "" for i in range(1, 6)}),
            '{"decision": "answer"}',
            "ANSWER: C\nREASONING: offline self-test.",
        ]

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return self.responses.pop(0) if self.responses else "ANSWER: A"


class _Entry:
    def __init__(self, i):
        self.text = f"caption {i}"
        self.date = "DAY1"
        self.start_time = f"{11 + i:02d}000000"
        self.end_time = f"{11 + i:02d}003000"
        self.video_path = None

    @property
    def timestamp_int(self):
        return (int("1" + self.start_time.zfill(8)), int("1" + self.end_time.zfill(8)))


entries = [_Entry(i) for i in range(5)]
memory = MeritMemory.__new__(MeritMemory)
memory.prompt_template_manager = ptm
memory.max_rounds = 5
memory.max_errors = 5
memory.debug_log_count = 0
memory.top_k = 5
memory._qa_count = 0
memory._all_entries = entries
memory._entries_by_timestamp = {}
memory.indexed_time = 10 ** 12
memory._retrieve = lambda *a, **k: {
    "retrieved_content": [], "all_candidates": [], "entries": entries
}
llm = _RecordingLLM()
memory.respond_llm_model = llm

result = memory.answer(
    query="Who used the screwdriver first?",
    choices={"A": "Tasha", "B": "Alice", "C": "Shure", "D": "Jake"},
    until_time=111210217,
    question_id="selftest",
)

check("4 LLM calls (reason, filter, reason, QA)", len(llm.calls) == 4, str(len(llm.calls)))
check("2 rounds", result.num_rounds == 2, str(result.num_rounds))
check("answer captured", result.answer.startswith("ANSWER: C"), result.answer[:30])

reason_prompt = llm.calls[0][-1]["content"]
qa_prompt = llm.calls[3][-1]["content"]
qa_text = "".join(p["text"] for p in qa_prompt if p.get("type") == "text")
check("reasoning and QA prompts differ as expected",
      ("Query Time" in reason_prompt) != ("Query Time" in qa_text))
check("QA prompt carries the retrieved captions", "top1: [DAY1 11:00:00" in qa_text)

# --------------------------------------------------------------------- done
print()
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("Self-test passed.")
