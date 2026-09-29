"""Key extraction template: event, dialogue, object, summary."""

from typing import Dict, List


MULTIKEY_EXTRACT_SYSTEM = (
    "You are extracting retrieval keys from an episodic video memory clip.\n\n"
    "Each input value corresponds to a ~30-second video clip and consists of:\n"
    "- physical actions and movements\n"
    "- spoken dialogue between people\n"
    "- interactions with objects\n"
    "- reflect the entire clip by summarizing the clip\n\n"
    "Your task is to extract EXACTLY FOUR retrieval keys from the value.\n"
    "Do NOT write extra explanations.\n"
    "Do NOT invent events.\n"
    "Use only information explicitly present in the value.\n\n"
    "The four keys MUST correspond to the following categories:\n\n"
    "1. Event / Action key\n"
    "- What physical actions or events actually happened between people?\n"
    "- Focus on observable actions and interactions.\n"
    "- Use one short sentence or phrase.\n"
    "- Include the agent if identifiable (use actual names if present).\n"
    "- If the speaker uses first-person expressions (I / me), use 'I' or 'me'.\n\n"
    "2. Dialogue / Mention key\n"
    "- What was said, asked, or mentioned in the dialogue?\n"
    "- Focus on questions, statements, commands, or repeated mentions.\n"
    "- Use one short sentence or phrase.\n\n"
    "3. Object-state / Item-centric key\n"
    "- What object was handled, requested, moved, or referenced?\n"
    "- Describe the object and its state or role in the scene.\n"
    "- Use one short sentence or phrase.\n\n"
    "4. Summary / Retrieval key\n"
    "- Generate ONE concise retrieval key that best represents the core event of the clip.\n"
    "- Abstract away redundant or repeated actions.\n"
    "- Capture the main entities, actions, and intent.\n"
    "- Be concise and retrieval-friendly.\n"
    "- Prefer compact keyword-style phrasing (not a full sentence).\n"
    "- Use spaces between words.\n"
    "- Stay grounded in the value; do not add details.\n\n"
    "Formatting rules:\n"
    "- Output exactly four lines\n"
    "- One key per line, in the order: event, dialogue, object, summary\n"
    "- Use spaces between words\n"
    "- DO NOT use underscores (_)\n"
    "- Do not include numbering, bullets, or extra explanations\n"
    "- Each of lines 1-3 must be a single sentence or a single clause\n"
    "- Line 4 should be a short keyword-style phrase (not necessarily a sentence)\n"
)

MULTIKEY_EXTRACT_FEWSHOT = (
    "# Few-shot Examples:\n\n"
    "Value:\n"
    "\"A group sits around a table with notebooks and laptops, discussing plans for an upcoming workshop. "
    "We talk about the expected number of participants and how long the session should be. "
    "Mina suggests keeping the workshop short to avoid fatigue, while Daniel proposes adding a short break. "
    "I agree and suggest starting at 2 PM with a brief introduction. "
    "The discussion shifts to preparation details, with Sarah mentioning handouts and name tags. "
    "We decide to use digital slides instead and bring extra chargers. "
    "Before ending the meeting, we joke about setting up the projector, and I confirm the final agenda.\"\n"
    "Output:\n"
    "People discuss workshop planning and schedule during a meeting\n"
    "Participants suggest session length, breaks, and preparation details\n"
    "Workshop materials and equipment are decided and assigned\n"
    "group meeting workshop planning schedule materials logistics\n"
)


def get_multikey_extract_prompt(caption: str) -> str:
    """Build the full prompt for extracting 4 keys from a caption."""
    return (
        MULTIKEY_EXTRACT_SYSTEM + "\n"
        + MULTIKEY_EXTRACT_FEWSHOT + "\n"
        "Now extract the four retrieval keys from the following value.\n\n"
        f"Value:\n\"{caption}\"\n\n"
        "Output:"
    )


def parse_multikey_response(response: str) -> Dict[str, str]:
    """Parse the 4-line response into key dict."""
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    def _get(idx: int) -> str:
        return lines[idx].strip("'\"") if idx < len(lines) else ""
    return {
        "event_key": _get(0),
        "dialogue_key": _get(1),
        "object_key": _get(2),
        "sum_key": _get(3),
    }


# Required by PromptTemplateManager
prompt_template = [
    {"role": "system", "content": MULTIKEY_EXTRACT_SYSTEM},
]
