"""Audit a JSONL of {complex, simple} pairs for data-quality issues.

Per record we compute:

  * `length_ratio` — output_words / source_words. Flagged `length_inflated`
    when the soft cap is exceeded (default 1.3×).
  * `pacing_variety` — diversity of sentence openings, 0..1. Flagged
    `monotonous` below `pacing_threshold` (default 0.4).
  * `judge_level` (optional) — CEFR level returned by the local LM judge.
    Flagged `too_easy` for A1 / <A1, `too_hard` for B1 / B2+.

Aggregate report shows mean scores and how many records tripped each flag.

Usage:
    uv run python dataset_audit.py data/sft.jsonl
    uv run python dataset_audit.py data/sft.jsonl --with-judge
    uv run python dataset_audit.py data/sft.jsonl --limit 20 --show-samples
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from langsimp.verifier import (
    BaseJudge,
    LocalJudge,
    PacingVarietyTest,
    length_ratio_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

LENGTH_SOFT_CAP = 1.3
PACING_THRESHOLD = 0.4

TOO_EASY_LEVELS = {"A1", "<A1"}
TOO_HARD_LEVELS = {"B1", "B2+"}

_pacing = PacingVarietyTest()


def _judge_level(simple_text: str, judge: BaseJudge) -> Optional[str]:
    """Ask the judge for a CEFR level on the simple text alone.

    Uses a minimal prompt rather than the heavy DifficultyRankingTest one —
    this is data triage, not RL scoring; we just want a label.
    """
    prompt = (
        "Classify the following English text on the CEFR scale "
        "(<A1, A1, A2, B1, B2+, NA). Reply with ONLY a JSON object: "
        '{"level": "<one of the labels>"}\n\n'
        f"TEXT:\n{simple_text}"
    )
    try:
        result = judge.evaluate(prompt)
        level = (result.get("level") or "").strip()
        return level or None
    except Exception:
        return None


def audit_record(rec: dict, judge: Optional[BaseJudge]) -> dict:
    """Run all checks on one (complex, simple) pair. Returns scores + flags."""
    src = rec.get("complex", "")
    out = rec.get("simple", "")

    length = length_ratio_score(src, out)
    pacing = _pacing.run(out, judge=None)

    flags: list[str] = []
    if length < 1.0:
        flags.append("length_inflated")
    if pacing < PACING_THRESHOLD:
        flags.append("monotonous")

    judge_level: Optional[str] = None
    if judge is not None:
        judge_level = _judge_level(out, judge)
        if judge_level in TOO_EASY_LEVELS:
            flags.append("too_easy")
        elif judge_level in TOO_HARD_LEVELS:
            flags.append("too_hard")

    return {
        "scores": {"length_ratio": length, "pacing_variety": pacing},
        "judge_level": judge_level,
        "flags": flags,
        "title": rec.get("title"),
    }


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def audit_file(path: Path, judge: Optional[BaseJudge], limit: int = 0) -> dict:
    rows = _read_jsonl(path)
    if limit:
        rows = rows[:limit]

    records = [audit_record(r, judge) for r in rows]

    flag_counts: Counter = Counter()
    for r in records:
        for f in r["flags"]:
            flag_counts[f] += 1

    level_counts: Counter = Counter()
    for r in records:
        if r["judge_level"]:
            level_counts[r["judge_level"]] += 1

    n = len(records) or 1
    totals = {
        "count": len(records),
        "mean_length_ratio": sum(r["scores"]["length_ratio"] for r in records) / n,
        "mean_pacing_variety": sum(r["scores"]["pacing_variety"] for r in records) / n,
        "flag_counts": dict(flag_counts),
        "level_counts": dict(level_counts),
    }
    return {"records": records, "totals": totals}


def _print_report(report: dict, source_rows: list[dict], show_samples: int = 0) -> None:
    t = report["totals"]
    print(f"\n=== AUDIT SUMMARY ({t['count']} records) ===")
    print(f"mean length_ratio score : {t['mean_length_ratio']:.3f}  (1.0 = no inflation, 0.0 = ≥2× source)")
    print(f"mean pacing_variety     : {t['mean_pacing_variety']:.3f}  (1.0 = all openings unique, 0.0 = all same)")
    print(f"flag counts             : {t['flag_counts'] or '(none)'}")
    if t["level_counts"]:
        print(f"judge level counts      : {t['level_counts']}")

    if show_samples:
        worst = sorted(
            zip(report["records"], source_rows),
            key=lambda pair: (
                len(pair[0]["flags"]),
                -pair[0]["scores"]["pacing_variety"],
            ),
            reverse=True,
        )[:show_samples]
        print(f"\n=== {show_samples} MOST-FLAGGED EXAMPLES ===")
        for rec, src in worst:
            print(f"\nflags={rec['flags']} scores={rec['scores']} title={rec.get('title')!r}")
            print(f"  COMPLEX: {src.get('complex', '')[:200]}…")
            print(f"  SIMPLE : {src.get('simple', '')[:200]}…")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="JSONL file with {complex, simple, ...} records")
    p.add_argument("--with-judge", action="store_true",
                   help="also run the LM Studio CEFR-level check (requires LM Studio)")
    p.add_argument("--lm-studio-url", default="http://127.0.0.1:1234/v1")
    p.add_argument("--judge-model", default="google/gemma-4-26b-a4b")
    p.add_argument("--limit", type=int, default=0, help="0 = audit all rows")
    p.add_argument("--show-samples", type=int, default=5,
                   help="print N most-flagged records at the end")
    args = p.parse_args()

    judge = None
    if args.with_judge:
        judge = LocalJudge(base_url=args.lm_studio_url, model_name=args.judge_model)

    path = Path(args.path)
    source_rows = _read_jsonl(path)
    if args.limit:
        source_rows = source_rows[: args.limit]

    report = audit_file(path, judge=judge, limit=args.limit)
    _print_report(report, source_rows, show_samples=args.show_samples)


if __name__ == "__main__":
    main()
