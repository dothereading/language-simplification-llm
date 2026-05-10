"""Held-out CEFR-judge evaluation for trained adapters.

Loads a base model + optional LoRA adapter, runs it on every prompt in
`data/eval.jsonl`, then asks the local LM Studio judge to label each output
on the CEFR scale. Reports % A2, % too-easy, % too-hard, length ratio, plus
a few sample outputs.

We deliberately reuse `verifier.DifficultyRankingTest` as the judge so the
eval signal is identical to the reward signal we'll use for GRPO later.

Usage:
    uv run python -m langsimp.inference.eval_harness --adapter base
    uv run python -m langsimp.inference.eval_harness --adapter adapters/sft-a2
    uv run python -m langsimp.inference.eval_harness --adapter adapters/dpo-a2 --output eval_results/dpo.json

Requires LM Studio running with the judge model loaded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Callable, Optional

from langsimp.inference.engine import build_prompt, clean_generation, load_model_with_adapter, make_generate_fn
from langsimp.verifier import DifficultyRankingTest, LocalJudge

# Re-export so existing imports (`from eval_harness import build_eval_prompt`,
# tests, etc.) keep working without churn.
build_eval_prompt = build_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_PATH = REPO_ROOT / "data" / "eval.jsonl"
DEFAULT_RESULTS_DIR = REPO_ROOT / "eval_results"


# ---------- pure orchestration (no I/O) ----------

def evaluate_eval_set(
    records: list[dict],
    generate_fn: Callable[[str], str],
    classify_fn: Callable[[str], str],
) -> list[dict]:
    """For each record, generate a simplification and classify its level.

    `generate_fn(complex_text) -> output_text`
    `classify_fn(output_text) -> CEFR label`

    If `classify_fn` raises, the level is recorded as "NA" rather than
    crashing the whole run — partial results are more useful than none.
    """
    results: list[dict] = []
    for rec in records:
        complex_text = rec["complex"]
        output = generate_fn(complex_text)
        try:
            level = classify_fn(output)
        except Exception as e:
            print(f"[eval] classify failed: {e}", flush=True)
            level = "NA"
        results.append({
            "complex": complex_text,
            "output": output,
            "level": level,
            "title": rec.get("title"),
            "url": rec.get("url"),
            "source_words": len(complex_text.split()),
            "output_words": len(output.split()),
        })
    return results


def summarize(results: list[dict]) -> dict:
    """Aggregate level distribution and length stats over per-record results."""
    n = len(results)
    if n == 0:
        return {
            "count": 0, "level_counts": {},
            "pct_a2": 0.0, "pct_too_easy": 0.0, "pct_too_hard": 0.0,
            "mean_length_ratio": 0.0,
        }

    level_counts: dict[str, int] = {}
    for r in results:
        level_counts[r["level"]] = level_counts.get(r["level"], 0) + 1

    a2 = level_counts.get("A2", 0)
    too_easy = level_counts.get("A1", 0) + level_counts.get("<A1", 0)
    too_hard = level_counts.get("B1", 0) + level_counts.get("B2+", 0)

    ratios = [
        r["output_words"] / r["source_words"]
        for r in results if r["source_words"] > 0
    ]

    return {
        "count": n,
        "level_counts": level_counts,
        "pct_a2": a2 / n,
        "pct_too_easy": too_easy / n,
        "pct_too_hard": too_hard / n,
        "mean_length_ratio": mean(ratios) if ratios else 0.0,
    }


# ---------- I/O wiring ----------

def _load_eval_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_verifier_samples() -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"A1": [], "A2": [], "B1": []}
    with open(REPO_ROOT / "samples.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["level"] in buckets:
                buckets[d["level"]].append(d["text"])
    return buckets


def _make_generate_fn(model_id: str, adapter_path: Optional[str], max_tokens: int) -> tuple[Callable, object]:
    """Return (generate_fn, tokenizer). adapter_path=None means base model."""
    model, tokenizer = load_model_with_adapter(model_id, adapter_path)
    return make_generate_fn(model, tokenizer, max_tokens=max_tokens), tokenizer


def _format_report(summary: dict, results: list[dict], n_samples: int) -> str:
    lines = [
        "",
        f"=== EVAL SUMMARY ({summary['count']} records) ===",
        f"  pct A2          : {summary['pct_a2']:.1%}",
        f"  pct too easy    : {summary['pct_too_easy']:.1%}  (A1, <A1)",
        f"  pct too hard    : {summary['pct_too_hard']:.1%}  (B1, B2+)",
        f"  mean length ratio: {summary['mean_length_ratio']:.2f}",
        f"  level counts    : {summary['level_counts']}",
    ]
    if n_samples > 0 and results:
        # Show one A2, one too-easy, one too-hard if available
        picks: list[dict] = []
        for level_set, label in [({"A2"}, "A2"), ({"A1", "<A1"}, "too easy"), ({"B1", "B2+"}, "too hard")]:
            match = next((r for r in results if r["level"] in level_set), None)
            if match:
                picks.append({"label": label, **match})
        lines.append("")
        lines.append(f"=== SAMPLE OUTPUTS (one per category) ===")
        for p in picks[:n_samples]:
            lines.append("")
            lines.append(f"--- {p['label']} ({p['level']}) — {p.get('title', '')} ---")
            lines.append(f"COMPLEX ({p['source_words']}w): {p['complex'][:200]}…")
            lines.append(f"OUTPUT  ({p['output_words']}w): {p['output'][:300]}")
    return "\n".join(lines)


def _save_results(out_path: Path, summary: dict, results: list[dict], meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"[eval] wrote results → {out_path}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True,
                   help="path to adapter dir, or 'base' for no adapter")
    p.add_argument("--model", default="mlx-community/gemma-3-1b-it-bf16")
    p.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH))
    p.add_argument("--output", default=None,
                   help="JSON results path; default eval_results/<adapter_name>_<timestamp>.json")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--lm-studio-url", default="http://127.0.0.1:1234/v1")
    p.add_argument("--judge-model", default="google/gemma-4-26b-a4b")
    p.add_argument("--n-samples", type=int, default=3,
                   help="how many sample outputs to show in the report")
    p.add_argument("--limit", type=int, default=0,
                   help="0 = run all eval records")
    args = p.parse_args()

    # Wire up the I/O.
    eval_records = _load_eval_records(Path(args.eval_path))
    if args.limit:
        eval_records = eval_records[: args.limit]
    print(f"[eval] {len(eval_records)} records from {args.eval_path}", flush=True)

    adapter_path = None if args.adapter == "base" else args.adapter
    generate_fn, _tokenizer = _make_generate_fn(args.model, adapter_path, args.max_tokens)

    samples = _load_verifier_samples()
    judge = LocalJudge(base_url=args.lm_studio_url, model_name=args.judge_model)
    judge_test = DifficultyRankingTest(
        a1_samples=samples["A1"], b1_samples=samples["B1"], a2_samples=samples["A2"],
        n_words=100,
    )

    def classify_fn(text: str) -> str:
        return judge_test.classify(text, judge)

    started = time.time()
    results = evaluate_eval_set(eval_records, generate_fn, classify_fn)
    elapsed = time.time() - started
    summary = summarize(results)

    print(_format_report(summary, results, args.n_samples))
    print(f"\n[eval] {elapsed:.1f}s total", flush=True)

    out_path = Path(args.output) if args.output else (
        DEFAULT_RESULTS_DIR / f"{Path(args.adapter).name if adapter_path else 'base'}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    _save_results(out_path, summary, results, meta={
        "model": args.model,
        "adapter": args.adapter,
        "eval_path": args.eval_path,
        "judge_model": args.judge_model,
        "max_tokens": args.max_tokens,
        "elapsed_seconds": elapsed,
    })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
