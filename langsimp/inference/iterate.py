"""Iterate on the A2 simplification prompt against ONE paragraph.

Pulls a single Wikipedia paragraph (or uses --paragraph-file), sends it to
the OpenRouter teacher N times with `DISTILL_SYSTEM_PROMPT`, prints each
output, and scores each one with the local CEFR verifier.

Usage:
    uv run python -m langsimp.inference.iterate --runs 4
    uv run python -m langsimp.inference.iterate --runs 4 --seed 42
    uv run python -m langsimp.inference.iterate --runs 4 --paragraph-file my_paragraph.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from langsimp.data.distill import Teacher
from langsimp.prompts import DISTILL_SYSTEM_PROMPT
from langsimp.data.sources import fetch_random_paragraphs
from langsimp.verifier import DifficultyRankingTest, LocalJudge, RewardVerifier

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_verifier_samples():
    buckets = {"A1": [], "A2": [], "B1": []}
    with open(REPO_ROOT / "samples.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["level"] in buckets:
                buckets[d["level"]].append(d["text"])
    return buckets


async def run(args: argparse.Namespace) -> None:
    if args.paragraph_file:
        paragraph = Path(args.paragraph_file).read_text().strip()
        title, url = "(local file)", args.paragraph_file
    else:
        wp = next(fetch_random_paragraphs(
            n=1, min_words=args.min_words, max_words=args.max_words, seed=args.seed,
        ))
        paragraph, title, url = wp.text, wp.title, wp.url

    print("=" * 80)
    print(f"SOURCE PARAGRAPH — {title}")
    print(f"URL: {url}")
    print(f"({len(paragraph.split())} words)")
    print("-" * 80)
    print(paragraph)
    print("=" * 80)

    teacher = Teacher.from_env(model=args.model, temperature=args.temperature, max_retries=1)
    outputs = await asyncio.gather(*[
        teacher.simplify(DISTILL_SYSTEM_PROMPT, paragraph) for _ in range(args.runs)
    ])

    samples = load_verifier_samples()
    judge = LocalJudge(base_url=args.lm_studio_url, model_name=args.judge_model)
    verifier = RewardVerifier(judge)
    verifier.add_test(DifficultyRankingTest(
        a1_samples=samples["A1"], b1_samples=samples["B1"], a2_samples=samples["A2"],
        n_words=100,
    ))

    for i, out in enumerate(outputs, 1):
        print(f"\n--- RUN {i}/{args.runs} ({len(out.split()) if out else 0} words) ---")
        print(out or "(no output)")
        if args.skip_verifier or not out:
            continue
        try:
            score = verifier.verify(out)
            verdict = "A2" if score == 1.0 else "NOT A2"
            print(f"[verifier] {verdict} (score={score})")
        except Exception as e:
            print(f"[verifier] error: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--model", default="anthropic/claude-haiku-latest")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=None, help="seed for paragraph selection")
    p.add_argument("--min-words", type=int, default=80)
    p.add_argument("--max-words", type=int, default=220)
    p.add_argument("--paragraph-file", default=None, help="use a fixed paragraph from a file")
    p.add_argument("--lm-studio-url", default="http://127.0.0.1:1234/v1")
    p.add_argument("--judge-model", default="google/gemma-4-26b-a4b")
    p.add_argument("--skip-verifier", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
