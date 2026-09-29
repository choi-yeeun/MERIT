"""Batched neighbor-caption filtering template (text only, one call per round)."""

from typing import Dict, List, Any

NEIGHBOR_FILTER_SYSTEM = """You are a helpful assistant that extracts relevant information from video captions.

Given a question with multiple choice answers and captions from retrieved clips' neighborhoods (±1 minute window each), your task is to:
1. For each retrieved clip, analyze its 5 neighbor captions (before_2, before_1, center, after_1, after_2)
2. Extract ONLY the information relevant to answering the question
3. Return relevant info for each clip

Output format (JSON):
{
  "clip_1": "...concise relevant info...",
  "clip_2": "",
  ...
  "clip_N": "..."
}

Guidelines:
- Focus on information that directly helps answer the question
- If no relevant information is found for a clip, output empty string ""
- Keep each relevant_info concise (1-3 sentences)
- Output valid JSON only, no extra commentary
"""


def get_neighbor_filter_prompt():
    """Return the text-only batch neighbor filter prompt."""
    return [
        {"role": "system", "content": NEIGHBOR_FILTER_SYSTEM},
    ]


def format_neighbor_filter_content(
    query: str,
    choices: Dict[str, str],
    all_neighbor_captions: List[List[str]],
) -> str:
    """Format text-only content for batch neighbor filtering."""
    choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))

    lines = [f"Question: {query}", f"Choices: {choices_str}", ""]

    position_labels = ["before_2", "before_1", "center", "after_1", "after_2"]

    for clip_idx, neighbor_captions in enumerate(all_neighbor_captions, start=1):
        lines.append(f"=== Clip {clip_idx} ===")
        for pos_idx, caption in enumerate(neighbor_captions):
            pos_label = position_labels[pos_idx] if pos_idx < len(position_labels) else f"clip_{pos_idx}"
            lines.append(f"{pos_label}: {caption}")
        lines.append("")

    lines.append(
        "For each clip, extract relevant information from its neighbor captions "
        "that helps answer the question. Output JSON with keys clip_1 through "
        f"clip_{len(all_neighbor_captions)}."
    )

    return "\n".join(lines)


# Required by PromptTemplateManager
prompt_template = [
    {"role": "system", "content": NEIGHBOR_FILTER_SYSTEM},
]
