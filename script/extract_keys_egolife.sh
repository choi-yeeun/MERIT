#!/usr/bin/env bash
# Regenerate the retrieval keys for an EgoLife subject. The A1_JAKE key file
# already ships in data/EgoLife/captions/A1_JAKE/.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUBJECT="A1_JAKE"
DATA_DIR="data/EgoLife"
MODEL="gpt-5-mini"
CONCURRENCY=20
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)     SUBJECT="$2";     shift 2 ;;
        --data-dir)    DATA_DIR="$2";    shift 2 ;;
        --model)       MODEL="$2";       shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        *)             EXTRA+=("$1");    shift   ;;
    esac
done

INPUT_FILE="${DATA_DIR}/captions/${SUBJECT}/${SUBJECT}_30sec.json"
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Caption file not found: $INPUT_FILE" >&2
    exit 1
fi

CMD=(python preprocess/extract_keys.py
     --input-file "$INPUT_FILE"
     --model "$MODEL"
     --concurrency "$CONCURRENCY")
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    CMD+=("${EXTRA[@]}")
fi

printf 'Running:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
