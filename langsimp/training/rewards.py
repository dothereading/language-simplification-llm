"""GRPO reward components for language simplification.

Three active components in v1:
  * LengthVsSourceReward      — output/source word ratio in target band
  * VocabSimplicityReward     — penalty for too many uncommon words/sentence
  * SemanticPreservationReward — judge call comparing source vs output

CombinedReward aggregates them as a weighted sum, with a *meaning gate*:
if SemanticPreservation < gate threshold, the whole reward is zeroed.
This prevents the model from learning to game length/vocab while
silently shedding source content.

Two stubs (RepetitionReward, SmoothDifficultyReward) exist so the wiring
is in place; their numeric behavior is TODO.

The bottom of the file contains thin `@register_reward_function`-style
adapters that mlx_lm_lora.train can discover via --reward-functions-file.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from wordfreq import top_n_list

from langsimp.verifier import BaseJudge, split_sentences

# Default common-word list size. Top-3000 strikes the right balance: the
# bare top-2000 misses common A2 concrete vocabulary (e.g. "tower",
# "destroyed", "stands"), while top-5000+ stops penalizing real B1+
# academic prose. Calibrated against data/{sft,dpo}.jsonl — see
# TestVocabSimplicityCalibration. Cached per-N at module level so the
# wordfreq import cost is paid once.
_DEFAULT_TOP_N = 3000
_COMMON_WORDS_CACHE: dict[int, frozenset[str]] = {}


def _common_words(top_n: int) -> frozenset[str]:
    if top_n not in _COMMON_WORDS_CACHE:
        _COMMON_WORDS_CACHE[top_n] = frozenset(top_n_list("en", top_n))
    return _COMMON_WORDS_CACHE[top_n]


@dataclass
class RewardContext:
    """Per-rollout context. Source = the complex paragraph the model is
    rewriting. Answer = optional reference simplification (Opus chosen)
    that some rewards may compare against; not used in v1."""
    source: str
    answer: Optional[str] = None


class RewardComponent(ABC):
    """Single reward component. Returns a float in [0, 1]."""
    name: str = "reward"

    @abstractmethod
    def compute(
        self, output: str, ctx: RewardContext, judge: Optional[BaseJudge] = None,
    ) -> float: ...


# ---------- LengthVsSourceReward ----------

class LengthVsSourceReward(RewardComponent):
    """Reward 1.0 when output_words/source_words is within [floor, ceiling],
    decaying linearly outside. Penalizes both excessive condensation
    (model dropped content) and excessive padding (model expanded source)."""
    name = "length"

    def __init__(
        self,
        soft_floor: float = 0.8,
        soft_ceiling: float = 1.3,
        hard_floor: float = 0.3,
        hard_ceiling: float = 2.5,
    ):
        self.soft_floor = soft_floor
        self.soft_ceiling = soft_ceiling
        self.hard_floor = hard_floor
        self.hard_ceiling = hard_ceiling

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        src_n = len(ctx.source.split())
        if src_n == 0:
            return 0.0
        ratio = len(output.split()) / src_n
        if self.soft_floor <= ratio <= self.soft_ceiling:
            return 1.0
        if ratio < self.soft_floor:
            if ratio <= self.hard_floor:
                return 0.0
            return (ratio - self.hard_floor) / (self.soft_floor - self.hard_floor)
        # ratio > soft_ceiling
        if ratio >= self.hard_ceiling:
            return 0.0
        return 1.0 - (ratio - self.soft_ceiling) / (self.hard_ceiling - self.soft_ceiling)


# ---------- VocabSimplicityReward ----------

_PROPER_NOUN_RE = re.compile(r"^[A-Z][a-z]+$")


def _tokenize_words(sentence: str) -> list[str]:
    """Split into word-only tokens (drops punctuation)."""
    return re.findall(r"[A-Za-z'']+", sentence)


def _is_likely_proper_noun(word: str, position_in_sentence: int) -> bool:
    """Heuristic: capitalized AND not the first word of the sentence."""
    return position_in_sentence > 0 and bool(_PROPER_NOUN_RE.match(word))


class VocabSimplicityReward(RewardComponent):
    """Per-sentence: count words not in the top-N most common English words
    (skipping proper nouns). Penalty kicks in once that count exceeds
    `allowed_uncommon` per sentence. Reward = 1 - (mean penalty across
    sentences), clipped to [0, 1].

    Defaults (top_n=3000, allowed=1, severity=0.5) calibrated against
    chosen/rejected/bad distributions in data/{sft,dpo}.jsonl. See
    TestVocabSimplicityCalibration for the regression targets."""
    name = "vocab"

    def __init__(self, top_n: int = _DEFAULT_TOP_N, allowed_uncommon: int = 1, severity: float = 0.5):
        self.common_words = _common_words(top_n)
        self.allowed_uncommon = allowed_uncommon
        self.severity = severity  # how much each excess uncommon word costs

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        sentences = split_sentences(output)
        if not sentences:
            return 1.0
        penalties: list[float] = []
        for s in sentences:
            tokens = _tokenize_words(s)
            uncommon = 0
            for i, tok in enumerate(tokens):
                if _is_likely_proper_noun(tok, i):
                    continue
                if tok.lower() not in self.common_words:
                    uncommon += 1
            excess = max(0, uncommon - self.allowed_uncommon)
            penalties.append(min(1.0, excess * self.severity))
        return max(0.0, 1.0 - sum(penalties) / len(penalties))


# ---------- SemanticPreservationReward ----------

_MEANING_PROMPT_TEMPLATE = """You will compare a SOURCE text and a SIMPLIFICATION of it. Rate the simplification on two axes, each 1-5.

1. **facts_preserved** — Did the simplification keep the important facts of the source? 5 = all important facts present; 1 = most important facts dropped.

2. **no_hallucinations** — Did the simplification avoid adding facts NOT in the source? 5 = nothing invented; 1 = many facts invented.

Note: dropping minor / decorative detail is fine and should NOT lower facts_preserved. Only score down when *important* facts are missing.

SOURCE:
{source}

SIMPLIFICATION:
{output}

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"facts_preserved": int, "no_hallucinations": int}}"""


class SemanticPreservationReward(RewardComponent):
    """Asks a judge whether the simplification preserves source meaning
    without hallucinating. Average of the two scores, normalized to [0, 1]."""
    name = "meaning"

    def compute(self, output: str, ctx: RewardContext, judge: Optional[BaseJudge] = None) -> float:
        if judge is None:
            return 0.5  # no judge → unknown; mid score so we don't crash GRPO
        prompt = _MEANING_PROMPT_TEMPLATE.format(source=ctx.source, output=output)
        try:
            result = judge.evaluate(prompt)
        except Exception:
            return 0.5
        try:
            facts = float(result.get("facts_preserved", 0))
            halluc = float(result.get("no_hallucinations", 0))
        except (TypeError, ValueError):
            return 0.5
        # Both axes contribute equally; convert from 1-5 → 0-1
        # (a score of 1 is the worst, 5 the best, so subtract 1 and /4)
        score = ((facts - 1) / 4 + (halluc - 1) / 4) / 2
        return max(0.0, min(1.0, score))


# ---------- Stubs (TODO) ----------

class RepetitionReward(RewardComponent):
    """TODO: penalize repetitive outputs (low unique-words/total ratio).
    Stubbed at 1.0 for v1 — included so the wiring works when we activate it."""
    name = "repetition"

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        return 1.0  # TODO: implement actual repetition detection


class SmoothDifficultyReward(RewardComponent):
    """TODO: judge-based CEFR level → smooth score (A2=1.0, A1=0.6, B1=0.4, ...).
    Stubbed at 1.0 for v1. Will reuse few-shot samples like
    verifier.DifficultyRankingTest does."""
    name = "difficulty"

    def __init__(self, a1_samples=None, a2_samples=None, b1_samples=None):
        self.a1_samples = a1_samples or []
        self.a2_samples = a2_samples or []
        self.b1_samples = b1_samples or []

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        return 1.0  # TODO: judge call + smooth A2/A1/B1/B2+/<A1 → score


# ---------- CombinedReward ----------

class CombinedReward(RewardComponent):
    """Weighted sum of components. If a 'meaning' component is present and
    scores below `meaning_gate`, the whole reward is zeroed.

    The meaning gate is the safety belt against the model gaming the
    cheaper rewards (length, vocab) by silently dropping content."""
    name = "combined"

    def __init__(
        self,
        components: list[tuple[float, RewardComponent]],
        meaning_gate: float = 0.5,
    ):
        self.components = components
        self.meaning_gate = meaning_gate

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        scores: dict[str, float] = {}
        total = 0.0
        for w, comp in self.components:
            s = comp.compute(output, ctx, judge=judge)
            scores[comp.name] = s
            total += w * s
        meaning = scores.get("meaning")
        if meaning is not None and meaning < self.meaning_gate:
            return 0.0
        return max(0.0, min(1.0, total))


# ---------- mlx_lm_lora @register_reward_function adapters ----------
#
# mlx_lm_lora calls reward functions with the signature
#   (prompts: list[str], completions: list[str], answer: list[str],
#    types: list[str] | None) -> list[float]
# Note: the framework passes `answer` as a *singular* kwarg even though it
# is a list — the param name is part of the contract. Returning one float
# in [0, 1] per (prompt, completion). The framework picks the functions up
# by name via --reward-functions and --reward-functions-file.

_LENGTH = LengthVsSourceReward()
_VOCAB = VocabSimplicityReward()
_MEANING = SemanticPreservationReward()


_OPENROUTER_DEFAULT_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_DEFAULT_MODEL = "anthropic/claude-haiku-latest"


def _get_judge():
    """Lazy-load a judge from env. Backend selection:

      * MEANING_JUDGE_BACKEND=openrouter → OpenRouter (needs OPENROUTER_API_KEY).
        Defaults: model=anthropic/claude-haiku-latest, url=https://openrouter.ai/api/v1.
        Override via MEANING_JUDGE_MODEL / MEANING_JUDGE_URL.
      * MEANING_JUDGE_URL set            → local LM Studio (no auth).
      * neither                          → None; meaning_reward returns 0.5
        (constant contribution → no signal but no crash).
    """
    import os
    if hasattr(_get_judge, "_cached"):
        return _get_judge._cached

    backend = os.environ.get("MEANING_JUDGE_BACKEND", "").lower()
    from langsimp.verifier import LocalJudge

    if backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MEANING_JUDGE_BACKEND=openrouter but OPENROUTER_API_KEY is not set"
            )
        url = os.environ.get("MEANING_JUDGE_URL", _OPENROUTER_DEFAULT_URL)
        model = os.environ.get("MEANING_JUDGE_MODEL", _OPENROUTER_DEFAULT_MODEL)
        _get_judge._cached = LocalJudge(base_url=url, model_name=model, api_key=api_key)
        return _get_judge._cached

    url = os.environ.get("MEANING_JUDGE_URL")
    if not url:
        return None
    model = os.environ.get("MEANING_JUDGE_MODEL", "google/gemma-4-26b-a4b")
    _get_judge._cached = LocalJudge(base_url=url, model_name=model)
    return _get_judge._cached


try:
    from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function
except ImportError:
    # Tests don't need mlx_lm_lora — fall back to a no-op decorator so
    # this module is still importable.
    def register_reward_function(name=None):
        def deco(fn): return fn
        return deco


@register_reward_function()
def length_reward(prompts, completions, answer, types=None) -> list[float]:
    return [
        _LENGTH.compute(c, RewardContext(source=p, answer=a))
        for p, c, a in zip(prompts, completions, answer)
    ]


@register_reward_function()
def vocab_reward(prompts, completions, answer, types=None) -> list[float]:
    return [
        _VOCAB.compute(c, RewardContext(source=p, answer=a))
        for p, c, a in zip(prompts, completions, answer)
    ]


@register_reward_function()
def meaning_reward(prompts, completions, answer, types=None) -> list[float]:
    judge = _get_judge()
    return [
        _MEANING.compute(c, RewardContext(source=p, answer=a), judge=judge)
        for p, c, a in zip(prompts, completions, answer)
    ]


# ---------- audit + variety (offline diagnostics) ----------
#
# Used to verify rewards make sense before training, and to monitor reward
# variance per group during/after training. Reward variance ≈ 0 inside a
# GRPO group means the advantage signal is dead.

def _default_combined() -> CombinedReward:
    return CombinedReward(
        components=[
            (0.50, _MEANING),
            (0.25, _LENGTH),
            (0.25, _VOCAB),
        ],
        meaning_gate=0.5,
    )


def audit_record(source: str, output: str, judge: Optional[BaseJudge] = None) -> dict[str, float]:
    """Per-component scores for one (source, output) pair, plus combined."""
    ctx = RewardContext(source=source)
    out = {
        "length": _LENGTH.compute(output, ctx),
        "vocab": _VOCAB.compute(output, ctx),
        "meaning": _MEANING.compute(output, ctx, judge=judge),
    }
    out["combined"] = _default_combined().compute(output, ctx, judge=judge)
    return out


def compute_variety(
    prompts: list[str],
    rollouts_per_prompt: list[list[str]],
    judge: Optional[BaseJudge] = None,
) -> dict:
    """For each prompt, score its rollouts and report mean/std.

    GRPO advantage = (reward - mean) / std within each group; if std ≈ 0
    the gradient is zero and no learning happens. This function tells us
    whether our rewards are *discriminating* between rollouts.
    """
    import statistics
    combined = _default_combined()
    per_prompt: list[dict] = []
    stds: list[float] = []
    for p, rollouts in zip(prompts, rollouts_per_prompt):
        scores = [
            combined.compute(r, RewardContext(source=p), judge=judge)
            for r in rollouts
        ]
        mean_s = statistics.mean(scores) if scores else 0.0
        std_s = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        per_prompt.append({"mean": mean_s, "std": std_s, "rewards": scores})
        stds.append(std_s)
    return {
        "per_prompt": per_prompt,
        "mean_std": statistics.mean(stds) if stds else 0.0,
        "min_std": min(stds) if stds else 0.0,
        "max_std": max(stds) if stds else 0.0,
    }


# ---------- CLI ----------

def _variety_cli(args) -> None:
    """Sample G rollouts per prompt from a real adapter; report reward
    std per group. GRPO advantage = (reward - mean) / std within a group;
    if std ≈ 0 across most groups, GRPO can't learn — this catches that
    BEFORE we burn training compute."""
    from langsimp.inference.engine import load_model_with_adapter, make_generate_fn

    judge = None
    if args.with_judge:
        from verifier import LocalJudge
        judge = LocalJudge(base_url=args.lm_studio_url, model_name=args.judge_model)

    # Load prompts. Accept either GRPO-shape (`prompt`) or SFT-shape (`complex`).
    prompts: list[str] = []
    with open(args.prompts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompts.append(r.get("prompt") or r.get("complex"))
            if len(prompts) >= args.n_prompts:
                break
    print(f"[variety] {len(prompts)} prompts × {args.group_size} rollouts at temp={args.temperature}", flush=True)

    adapter_path = None if args.adapter == "base" else args.adapter
    model, tokenizer = load_model_with_adapter(args.model, adapter_path)
    gen = make_generate_fn(model, tokenizer, max_tokens=args.max_tokens, temp=args.temperature)

    rollouts_per_prompt: list[list[str]] = []
    for i, p in enumerate(prompts):
        rollouts: list[str] = []
        for j in range(args.group_size):
            out = gen(p)
            rollouts.append(out)
            print(f"  [{i+1}/{len(prompts)}, rollout {j+1}/{args.group_size}] {len(out.split())}w", flush=True)
        rollouts_per_prompt.append(rollouts)

    stats = compute_variety(prompts, rollouts_per_prompt, judge=judge)
    print(f"\n=== REWARD VARIETY ({len(prompts)} prompts, G={args.group_size}) ===")
    print(f"  mean across-group std : {stats['mean_std']:.4f}")
    print(f"  min  across-group std : {stats['min_std']:.4f}")
    print(f"  max  across-group std : {stats['max_std']:.4f}")
    if stats["mean_std"] < 0.05:
        print("  ⚠️  mean std < 0.05 — GRPO advantage signal will be weak!")
    print(f"\n=== PER-PROMPT BREAKDOWN ===")
    for i, (p, prompt_text, rollouts) in enumerate(zip(stats["per_prompt"], prompts, rollouts_per_prompt)):
        print(f"\n[{i+1}] mean={p['mean']:.3f} std={p['std']:.4f}  rewards={[round(r, 3) for r in p['rewards']]}")
        if args.show_rollouts:
            print(f"    SOURCE ({len(prompt_text.split())}w): {prompt_text[:140]}…")
            for j, (r, score) in enumerate(zip(rollouts, p['rewards'])):
                # Per-component scores for this rollout
                comp = audit_record(prompt_text, r, judge=judge)
                print(f"    [rollout {j+1} | combined={score:.3f} L={comp['length']:.2f} V={comp['vocab']:.2f} M={comp['meaning']:.2f}] {len(r.split())}w")
                print(f"      {r[:200]}…" if len(r) > 200 else f"      {r}")


def _audit_cli(args) -> None:
    """Score a JSONL of {complex, simple} (or {complex, output}) records."""
    judge = None
    if args.with_judge:
        from verifier import LocalJudge
        judge = LocalJudge(base_url=args.lm_studio_url, model_name=args.judge_model)

    records: list[dict] = []
    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]

    output_field = args.output_field
    rows: list[dict] = []
    for rec in records:
        out = rec.get(output_field) or rec.get("simple") or rec.get("output", "")
        scores = audit_record(rec["complex"], out, judge=judge)
        rows.append({"title": rec.get("title", ""), **scores})

    if not rows:
        print("no records")
        return

    keys = ["length", "vocab", "meaning", "combined"]
    means = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    print(f"\n=== REWARD AUDIT ({len(rows)} records) ===")
    for k in keys:
        print(f"  mean {k:>9}: {means[k]:.3f}")

    if args.show_worst:
        worst = sorted(rows, key=lambda r: r["combined"])[: args.show_worst]
        print(f"\n=== {args.show_worst} WORST records by combined score ===")
        for r in worst:
            print(f"  {r['combined']:.3f}  L={r['length']:.2f} V={r['vocab']:.2f} M={r['meaning']:.2f}  {r['title']}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    audit = sub.add_parser("audit", help="score per-component rewards on a JSONL")
    audit.add_argument("path")
    audit.add_argument("--with-judge", action="store_true")
    audit.add_argument("--lm-studio-url", default="http://127.0.0.1:1234/v1")
    audit.add_argument("--judge-model", default="google/gemma-4-26b-a4b")
    audit.add_argument("--output-field", default="simple",
                       help="JSON field that holds the model output (simple|output|...)")
    audit.add_argument("--limit", type=int, default=0)
    audit.add_argument("--show-worst", type=int, default=5)

    variety = sub.add_parser("variety", help="sample rollouts from an adapter and report reward std per group")
    variety.add_argument("--adapter", required=True, help="adapter dir or 'base' for no adapter")
    variety.add_argument("--prompts-path", default="data/grpo/train.jsonl")
    variety.add_argument("--n-prompts", type=int, default=5)
    variety.add_argument("--group-size", type=int, default=4)
    variety.add_argument("--temperature", type=float, default=0.8)
    variety.add_argument("--max-tokens", type=int, default=512)
    variety.add_argument("--model", default="mlx-community/gemma-3-1b-it-bf16")
    variety.add_argument("--with-judge", action="store_true")
    variety.add_argument("--show-rollouts", action="store_true",
                         help="print each rollout's text and per-component scores")
    variety.add_argument("--lm-studio-url", default="http://127.0.0.1:1234/v1")
    variety.add_argument("--judge-model", default="google/gemma-4-26b-a4b")

    args = p.parse_args()
    if args.cmd == "audit":
        _audit_cli(args)
    elif args.cmd == "variety":
        _variety_cli(args)


if __name__ == "__main__":
    main()
