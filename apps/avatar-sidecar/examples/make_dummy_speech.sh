#!/usr/bin/env bash
# Makes two short test clips with macOS's built-in voices (no private audio needed).
#   dummy_he.wav  Hebrew (voice Carmit)
#   dummy_en.wav  English (voice Samantha)
# Output goes to the folder given as $1 (default: ./examples/out, git-ignored).
set -euo pipefail
OUT="${1:-$(dirname "$0")/out}"
mkdir -p "$OUT"
say -v Carmit -o "$OUT/dummy_he.wav" --data-format=LEI16@16000 "שלום, זו בדיקה קצרה של הסינתזה הקולית."
say -v Samantha -o "$OUT/dummy_en.wav" --data-format=LEI16@16000 "Hello, this is a short test of the talking head."
echo "wrote $OUT/dummy_he.wav and $OUT/dummy_en.wav"
