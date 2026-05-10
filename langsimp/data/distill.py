"""Distillation: ask a Teacher model (via OpenRouter) to simplify text.

The `Teacher` class wraps a single OpenAI-compatible async client and a model
name. It is shared by two CLI subcommands:

  * `python distill.py sft  --n 200 --output data/sft.jsonl`
        Source random Wikipedia paragraphs, ask the Teacher to simplify each,
        and write {complex, simple, ...} pairs.

  * `python distill.py dpo  --teacher google/gemma-3-4b-it`
        Re-simplify each prompt already in data/sft.jsonl with a *weaker*
        teacher to produce DPO `rejected` completions paired against the
        existing `chosen` ones.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from langsimp.prompts import (
    DISTILL_SYSTEM_PROMPT,
    REJECTED_CLARIFY_PROMPT,
    REJECTED_ELI5_PROMPT,
    REJECTED_SUMMARIZE_PROMPT,
)
from langsimp.data.sources import WikiParagraph, fetch_random_paragraphs

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------- DPO rejected-strategy mix ----------
#
# DPO learns "for prompt P, prefer chosen over rejected." If every rejected
# example fails the same way (e.g. always "Gemma-3-4B's mistakes with the
# A2 prompt"), the model learns to avoid THAT specific pattern but may not
# generalize to other failure modes.
#
# We diversify the rejected pool: majority are the original weak-model
# baseline (so DPO has a strong primary signal), with smaller buckets that
# fail in *different* ways — wrong task, wrong target level, no A2
# constraint at all.

@dataclass
class RejectedStrategy:
    name: str            # used in output records for downstream analysis
    weight: float        # share of the rejected pool, must sum to 1.0
    model: str           # OpenRouter model alias to call
    prompt: str          # system prompt (NOT the SFT prompt)


def build_dpo_strategies(weak_teacher: str, strong_teacher: str) -> list[RejectedStrategy]:
    """Default mix: 60% weak-model + same prompt, 40% strong-model + alt prompts.

    The 60% majority preserves the bulk DPO signal (a coherent "good vs
    mediocre A2" boundary). The 40% diversity exposes the model to other
    things to avoid.
    """
    return [
        RejectedStrategy("weak-distill", 0.60, weak_teacher, DISTILL_SYSTEM_PROMPT),
        RejectedStrategy("summarize",    0.15, strong_teacher, REJECTED_SUMMARIZE_PROMPT),
        RejectedStrategy("eli5",         0.15, strong_teacher, REJECTED_ELI5_PROMPT),
        RejectedStrategy("clarify",      0.10, strong_teacher, REJECTED_CLARIFY_PROMPT),
    ]


def pick_strategies(
    n: int, strategies: list[RejectedStrategy], seed: int = 0,
) -> list[RejectedStrategy]:
    """Deterministically assign one strategy per record.

    Uses round(weight * n) for each strategy and assigns any rounding
    remainder to the last bucket. For small n the smaller buckets may not
    appear at all — that's intentional, better than random sampling that
    would skew at the n=15 scale we use for testing.
    """
    counts = [round(s.weight * n) for s in strategies]
    counts[-1] += n - sum(counts)
    if any(c < 0 for c in counts):
        # Underweighted strategies stole from the last bucket; clamp.
        counts = [max(0, c) for c in counts]
        counts[-1] += n - sum(counts)
    assignments: list[RejectedStrategy] = []
    for s, c in zip(strategies, counts):
        assignments.extend([s] * c)
    random.Random(seed).shuffle(assignments)
    return assignments


class Teacher:
    """OpenRouter (or any OpenAI-compatible) chat client with retry logic.

    Usage:
        teacher = Teacher.from_env(model="anthropic/claude-opus-latest")
        text = await teacher.simplify(SYSTEM, "complex paragraph")
    """

    def __init__(
        self,
        client,
        model: str,
        max_retries: int = 3,
        temperature: float = 0.4,
        max_tokens: int = 1024,
    ):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

    @classmethod
    def from_env(cls, model: str, **kwargs) -> "Teacher":
        load_dotenv(REPO_ROOT / ".env")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set (env or .env)")
        client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        return cls(client=client, model=model, **kwargs)

    async def simplify(self, system_prompt: str, user_text: str) -> Optional[str]:
        """Return the simplified text, or None after exhausting retries."""
        for attempt in range(self.max_retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text
            except Exception as e:
                print(f"[teacher] attempt {attempt + 1} error: {e}", flush=True)
                await asyncio.sleep(2 * (attempt + 1))
        return None


# ---------- SFT subcommand ----------

async def _sft_run(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    if out_path != Path("/dev/stdout"):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    teacher = Teacher.from_env(
        model=args.model,
        max_retries=args.max_retries,
        temperature=args.temperature,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    pending: set[asyncio.Task] = set()
    kept = 0

    async def process(idx: int, p: WikiParagraph, fout):
        nonlocal kept
        async with semaphore:
            simplified = await teacher.simplify(DISTILL_SYSTEM_PROMPT, p.text)
        if simplified is None:
            return
        rec = {
            "complex": p.text,
            "simple": simplified,
            "title": p.title,
            "url": p.url,
            "model": args.model,
        }
        async with write_lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            kept += 1
            print(f"[{idx}] ok ({len(simplified.split())} words) — {p.title}", flush=True)

    mode = "a" if args.append else "w"
    print(f"streaming {args.n} examples → {out_path} (mode={'append' if args.append else 'overwrite'})", flush=True)

    with open(out_path, mode) as fout:
        idx = 0
        try:
            for p in fetch_random_paragraphs(
                n=args.n, min_words=args.min_words, max_words=args.max_words,
            ):
                task = asyncio.create_task(process(idx, p, fout))
                pending.add(task)
                task.add_done_callback(pending.discard)
                idx += 1
                if len(pending) >= args.concurrency * 4:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"sourcing stopped early: {e}", flush=True)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    print(f"wrote {kept}/{idx} examples to {out_path}", flush=True)


# ---------- DPO subcommand ----------

def _load_existing_prompts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)["prompt"])
            except Exception:
                continue
    return keys


async def _dpo_run(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_prompts(out_path) if args.resume else set()
    if existing:
        print(f"resuming: {len(existing)} examples already in {out_path}", flush=True)

    records: list[dict] = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["complex"] in existing:
                continue
            records.append(r)
            if args.limit and len(records) >= args.limit:
                break

    # Build strategy mix and assign one to each record.
    strategies = build_dpo_strategies(args.weak_teacher, args.strong_teacher)
    assignments = pick_strategies(len(records), strategies, seed=args.seed)
    counts = {s.name: 0 for s in strategies}
    for a in assignments:
        counts[a.name] += 1
    print(f"strategy mix for {len(records)} records: {counts}", flush=True)

    # Cache one Teacher per unique model (avoid re-creating the OpenAI client).
    teachers: dict[str, Teacher] = {}
    def get_teacher(model: str) -> Teacher:
        if model not in teachers:
            teachers[model] = Teacher.from_env(
                model=model, max_retries=args.max_retries, temperature=args.temperature,
            )
        return teachers[model]

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    kept = 0
    mode = "a" if (args.resume and out_path.exists()) else "w"

    async def process(idx: int, r: dict, strategy: RejectedStrategy, fout):
        nonlocal kept
        teacher = get_teacher(strategy.model)
        async with semaphore:
            rejected = await teacher.simplify(strategy.prompt, r["complex"])
        if rejected is None:
            print(f"[{idx}] FAILED ({strategy.name}) — {r.get('title', '')}", flush=True)
            return
        out = {
            "prompt": r["complex"],
            "chosen": r["simple"],
            "rejected": rejected,
            "title": r.get("title"),
            "url": r.get("url"),
            "chosen_model": r.get("model"),
            "rejected_model": strategy.model,
            "rejected_strategy": strategy.name,
        }
        async with write_lock:
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            kept += 1
            print(f"[{idx}] ok ({strategy.name}) — {r.get('title', '')}", flush=True)

    print(f"streaming {len(records)} rejected completions → {out_path}", flush=True)
    with open(out_path, mode) as fout:
        tasks = [
            asyncio.create_task(process(i, r, s, fout))
            for i, (r, s) in enumerate(zip(records, assignments))
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    print(f"wrote {kept}/{len(records)} examples to {out_path}", flush=True)


# ---------- CLI entrypoint ----------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sft = sub.add_parser("sft", help="generate SFT pairs from random Wikipedia paragraphs")
    sft.add_argument("--n", type=int, default=200)
    sft.add_argument("--output", default=str(REPO_ROOT / "data" / "sft.jsonl"))
    sft.add_argument("--append", action="store_true")
    sft.add_argument("--model", default="~anthropic/claude-opus-latest")
    sft.add_argument("--concurrency", type=int, default=8)
    sft.add_argument("--max-retries", type=int, default=3)
    sft.add_argument("--temperature", type=float, default=0.4)
    sft.add_argument("--min-words", type=int, default=60)
    sft.add_argument("--max-words", type=int, default=220)

    dpo = sub.add_parser(
        "dpo",
        help="generate diversified 'rejected' completions for DPO training",
    )
    dpo.add_argument("--input", default=str(REPO_ROOT / "data" / "sft.jsonl"))
    dpo.add_argument("--output", default=str(REPO_ROOT / "data" / "dpo.jsonl"))
    dpo.add_argument("--weak-teacher", default="google/gemma-3-4b-it",
                     help="model for the 'mediocre A2 attempt' strategy (60%% of records)")
    dpo.add_argument("--strong-teacher", default="~anthropic/claude-haiku-latest",
                     help="model for the alternative-prompt strategies (40%% of records)")
    dpo.add_argument("--concurrency", type=int, default=8)
    dpo.add_argument("--max-retries", type=int, default=3)
    dpo.add_argument("--temperature", type=float, default=0.4)
    dpo.add_argument("--limit", type=int, default=0, help="0 = all")
    dpo.add_argument("--seed", type=int, default=0,
                     help="determines per-record strategy assignment")
    dpo.add_argument("--resume", action="store_true",
                     help="skip prompts already present in output")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "sft":
        asyncio.run(_sft_run(args))
    elif args.cmd == "dpo":
        asyncio.run(_dpo_run(args))


if __name__ == "__main__":
    main()
