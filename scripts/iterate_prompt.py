"""Re-simplify a fixed sample of source paragraphs with the current prompt.

Used to A/B-test prompt iterations: pick the same N source paragraphs from
data/sft.jsonl, regenerate simplifications with the in-tree
DISTILL_SYSTEM_PROMPT, and write them to a fresh JSONL we can run through
dataset_audit.py.

Not part of the production pipeline — kept under scripts/ so it's easy to
see this is iteration scaffolding.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from langsimp.data.distill import Teacher  # noqa: E402
from langsimp.prompts import DISTILL_SYSTEM_PROMPT  # noqa: E402


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(REPO_ROOT / "data" / "sft.jsonl"))
    p.add_argument("--output", default=str(REPO_ROOT / "data" / "sft_new_prompt.jsonl"))
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model", default="~anthropic/claude-opus-latest")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    rows: list[dict] = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    rng = random.Random(args.seed)
    sample = rng.sample(rows, args.n)
    print(f"sampled {len(sample)} paragraphs (seed={args.seed})", flush=True)

    teacher = Teacher.from_env(model=args.model, max_retries=3, temperature=0.4)
    sem = asyncio.Semaphore(args.concurrency)

    async def regen(idx: int, rec: dict) -> dict | None:
        async with sem:
            simple = await teacher.simplify(DISTILL_SYSTEM_PROMPT, rec["complex"])
        if simple is None:
            print(f"[{idx}] FAILED — {rec.get('title', '')}", flush=True)
            return None
        print(f"[{idx}] ok — {rec.get('title', '')}", flush=True)
        return {
            "complex": rec["complex"],
            "simple": simple,
            "title": rec.get("title"),
            "url": rec.get("url"),
            "model": args.model,
            "old_simple": rec["simple"],  # keep the original for side-by-side
        }

    results = await asyncio.gather(*[regen(i, r) for i, r in enumerate(sample)])
    results = [r for r in results if r]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} → {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
