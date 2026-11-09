"""Replay fixed conversations through the real chat pipeline and check the replies.

Cases live outside Git ($MEMORIAL_DATA_HOME/eval/chat_cases.json) because the
forbidden phrases come from real bad replies. Each conversation is replayed
`samples` times in fresh sessions; every reply is checked against its turn's
rules and the whole run is saved to $MEMORIAL_DATA_HOME/eval/runs/.

Case file shape:
    {"samples": 3,
     "conversations": [
       {"name": "...",
        "turns": [{"user": "היוש", "max_chars": 80, "forbid": ["..."]}]}]}

Usage (from apps/api):
    .venv/bin/python -m scripts.eval_chat <person_id> [--cases PATH] [--samples N] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services.chat import chat_text  # noqa: E402

EVAL_DIR = settings.data_dir.parent / "eval"
_LATIN = re.compile(r"[A-Za-z]{2,}")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
MAX_EMOJI = 1


@dataclass
class TurnResult:
    conversation: str
    sample: int
    turn: int
    user: str
    reply: str
    seconds: float
    failures: list[str] = field(default_factory=list)


def check_reply(reply: str, rules: dict) -> list[str]:
    """Return the rule violations for one reply (empty list = pass)."""
    failures = []
    max_chars = rules.get("max_chars")
    if max_chars and len(reply) > max_chars:
        failures.append(f"too long ({len(reply)} > {max_chars})")
    min_chars = rules.get("min_chars")
    if min_chars and len(reply.strip()) < min_chars:
        failures.append(f"too short ({len(reply.strip())} < {min_chars})")
    lowered = reply.lower()
    failures += [f"forbidden: {phrase}" for phrase in rules.get("forbid", []) if phrase.lower() in lowered]
    must_any = rules.get("must_include_any", [])
    if must_any and not any(term.lower() in lowered for term in must_any):
        failures.append(f"missing one of: {must_any}")
    if not rules.get("allow_latin", False) and _LATIN.search(reply):
        failures.append("latin text")
    if len(_EMOJI.findall(reply)) > MAX_EMOJI:
        failures.append("too many emoji")
    if not reply.strip():
        failures.append("empty")
    return failures


def run(person_id: str, cases: dict, samples: int) -> list[TurnResult]:
    results = []
    for conversation in cases["conversations"]:
        for sample in range(1, samples + 1):
            session_id = None
            for turn_no, turn in enumerate(conversation["turns"], 1):
                start = time.monotonic()
                answer = chat_text(person_id, turn["user"], session_id)
                session_id = answer["session_id"]
                results.append(
                    TurnResult(
                        conversation=conversation["name"],
                        sample=sample,
                        turn=turn_no,
                        user=turn["user"],
                        reply=answer["reply"],
                        seconds=round(time.monotonic() - start, 1),
                        failures=check_reply(answer["reply"], turn),
                    )
                )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("person_id")
    parser.add_argument("--cases", type=Path, default=EVAL_DIR / "chat_cases.json")
    parser.add_argument("--samples", type=int, help="Override the case file's sample count")
    parser.add_argument("--quiet", action="store_true", help="Print pass/fail only, not reply text")
    parser.add_argument("--provider", help="Override LLM_PROVIDER for this run (cursor, ollama, anthropic)")
    parser.add_argument("--model", help="Override LLM_MODEL for this run")
    parser.add_argument("--temperature", type=float, help="Override LLM_TEMPERATURE for this run")
    args = parser.parse_args()
    if args.provider:
        settings.llm_provider = args.provider
    if args.model:
        settings.llm_model = args.model
    if args.temperature is not None:
        settings.llm_temperature = args.temperature

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    samples = args.samples or cases.get("samples", 3)
    results = run(args.person_id, cases, samples)

    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / f"chat-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    out.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    out.chmod(0o600)

    for r in results:
        status = "PASS" if not r.failures else "FAIL " + "; ".join(r.failures)
        line = f"[{r.conversation} #{r.sample}.{r.turn}] {r.seconds:>5}s {status}"
        print(line if args.quiet else f"{line}\n  > {r.user}\n  < {r.reply}")
    passed = sum(not r.failures for r in results)
    config = f"model={settings.llm_model} temp={settings.llm_temperature} min_score={settings.rerank_min_score}"
    print(f"\n{passed}/{len(results)} replies passed ({config}); saved {out.name}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
