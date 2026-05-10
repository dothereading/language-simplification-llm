"""Convert generated JSONL data into the on-disk format mlx-lm expects.

  * `python mlx_data.py sft` reads data/sft.jsonl (records with `complex` and
    `simple` fields) and writes data/mlx/{train,valid}.jsonl as chat records.

  * `python mlx_data.py dpo` reads data/dpo.jsonl (records with `prompt`,
    `chosen`, `rejected`) and writes data/dpo_mlx/{train,valid}.jsonl in the
    DPO format mlx_lm_lora expects.

The shared helpers (`to_mlx_sft_record`, `to_mlx_dpo_record`,
`split_train_valid`) are exported so tests can exercise them directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from langsimp.prompts import SFT_SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_PATH = REPO_ROOT / "data" / "eval.jsonl"


# ---------- hash-based split assignment ----------

def assign_split(key: str, valid_frac: float, seed: int = 0) -> str:
    """Deterministic train/valid bucket for a key.

    Same `key` + `seed` always returns the same bucket. Used so that:
      * SFT and DPO derive identical valid sets (they share source prompts);
      * Adding new prompts to the dataset later doesn't shuffle existing
        assignments (key membership is stable).
    """
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    bucket = int(h[:8], 16) / (1 << 32)
    return "valid" if bucket < valid_frac else "train"


def select_eval_keys(keys: list[str], n: int, seed: int = 0) -> list[str]:
    """Pick `n` keys to hold out as eval prompts, stably under dataset growth.

    Sorts by hash and takes the top n. Adding more keys later only changes
    the eval set if a new key happens to hash *lower* than one of the
    existing top-n — which is rare for reasonable n.
    """
    return sorted(keys, key=lambda k: hashlib.sha256(f"eval:{seed}:{k}".encode()).hexdigest())[:n]


def carve_eval(source_path: Path, output_path: Path, n: int, seed: int = 0) -> int:
    """Write n eval records (deterministic) to output_path. Returns count.

    Source file is left untouched — the eval records still appear in the
    source dataset. Splitting code (`prepare_sft_splits`, `prepare_dpo_splits`)
    filters them out at training-data prep time.
    """
    rows = _read_jsonl(source_path)
    keys = [r["complex"] for r in rows]
    eval_keys = set(select_eval_keys(keys, n=n, seed=seed))
    eval_rows = [r for r in rows if r["complex"] in eval_keys]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(eval_rows)


def load_eval_prompts(eval_path: Path) -> set[str]:
    """Set of prompt strings to exclude from training. Empty if file missing.

    Reads either schema: SFT-shaped (`complex` field) or DPO-shaped (`prompt`).
    """
    if not eval_path.exists():
        return set()
    out: set[str] = set()
    for r in _read_jsonl(eval_path):
        if "complex" in r:
            out.add(r["complex"])
        elif "prompt" in r:
            out.add(r["prompt"])
    return out


def prepare_sft_splits(
    rows: list[dict],
    eval_prompts: set[str],
    valid_frac: float,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Filter eval prompts out of SFT rows and assign train/valid by hash."""
    train: list[dict] = []
    valid: list[dict] = []
    for r in rows:
        if r["complex"] in eval_prompts:
            continue
        if assign_split(r["complex"], valid_frac, seed) == "valid":
            valid.append(r)
        else:
            train.append(r)
    return train, valid


def prepare_dpo_splits(
    rows: list[dict],
    eval_prompts: set[str],
    valid_frac: float,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """DPO version of prepare_sft_splits — uses `prompt` as the join key."""
    train: list[dict] = []
    valid: list[dict] = []
    for r in rows:
        if r["prompt"] in eval_prompts:
            continue
        if assign_split(r["prompt"], valid_frac, seed) == "valid":
            valid.append(r)
        else:
            train.append(r)
    return train, valid


def to_mlx_sft_record(complex_text: str, simple_text: str) -> dict:
    """Format one (complex, simple) pair as an mlx-lm chat record."""
    return {
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": complex_text.strip()},
            {"role": "assistant", "content": simple_text.strip()},
        ]
    }


def to_mlx_dpo_record(prompt: str, chosen: str, rejected: str) -> dict:
    """Format one preference triple as an mlx_lm_lora DPO record."""
    return {
        "system": SFT_SYSTEM_PROMPT,
        "prompt": prompt.strip(),
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
    }


def to_mlx_grpo_record(complex_text: str, simple_text: str) -> dict:
    """Format one record for mlx_lm_lora GRPO. The 'answer' field carries
    the Opus reference simplification — not used by v1 rewards but
    available for future ones (e.g. embedding-similarity vs reference)."""
    return {
        "prompt": complex_text.strip(),
        "answer": simple_text.strip(),
        "system": SFT_SYSTEM_PROMPT,
    }


def prepare_grpo_splits(
    rows: list[dict],
    eval_prompts: set[str],
    valid_n: int,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Filter eval prompts, hold out `valid_n` records, sort train by source
    length ascending (curriculum: easier prompts first)."""
    available = [r for r in rows if r["complex"] not in eval_prompts]
    valid_keys = set(select_eval_keys([r["complex"] for r in available], n=valid_n, seed=seed))
    train: list[dict] = []
    valid: list[dict] = []
    for r in available:
        if r["complex"] in valid_keys:
            valid.append(r)
        else:
            train.append(r)
    train.sort(key=lambda r: len(r["complex"].split()))
    return train, valid


def split_train_valid(
    rows: list[dict], valid_frac: float, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Shuffle `rows` and split into (train, valid). At least one valid row."""
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_valid = max(1, int(len(shuffled) * valid_frac))
    return shuffled[n_valid:], shuffled[:n_valid]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_split(out_dir: Path, split_name: str, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split_name}.jsonl"
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} → {path}")


def _sft_main(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.input))
    if args.n and len(rows) < args.n:
        raise SystemExit(f"requested --n={args.n} but only {len(rows)} rows in {args.input}")
    if args.n:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.n]

    eval_prompts = load_eval_prompts(Path(args.eval_path))
    if eval_prompts:
        print(f"excluding {len(eval_prompts)} eval prompts from {args.eval_path}")

    train, valid = prepare_sft_splits(rows, eval_prompts, args.valid_frac, args.seed)
    out_dir = Path(args.output_dir)
    _write_split(out_dir, "train", [to_mlx_sft_record(r["complex"], r["simple"]) for r in train])
    _write_split(out_dir, "valid", [to_mlx_sft_record(r["complex"], r["simple"]) for r in valid])


def _dpo_main(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.input))
    eval_prompts = load_eval_prompts(Path(args.eval_path))
    if eval_prompts:
        print(f"excluding {len(eval_prompts)} eval prompts from {args.eval_path}")

    train, valid = prepare_dpo_splits(rows, eval_prompts, args.valid_frac, args.seed)
    out_dir = Path(args.output_dir)
    _write_split(
        out_dir, "train",
        [to_mlx_dpo_record(r["prompt"], r["chosen"], r["rejected"]) for r in train],
    )
    _write_split(
        out_dir, "valid",
        [to_mlx_dpo_record(r["prompt"], r["chosen"], r["rejected"]) for r in valid],
    )


def _grpo_main(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.input))
    eval_prompts = load_eval_prompts(Path(args.eval_path))
    if eval_prompts:
        print(f"excluding {len(eval_prompts)} eval prompts from {args.eval_path}")

    train, valid = prepare_grpo_splits(rows, eval_prompts, args.valid_n, args.seed)
    out_dir = Path(args.output_dir)
    _write_split(
        out_dir, "train",
        [to_mlx_grpo_record(r["complex"], r["simple"]) for r in train],
    )
    _write_split(
        out_dir, "valid",
        [to_mlx_grpo_record(r["complex"], r["simple"]) for r in valid],
    )


def _carve_main(args: argparse.Namespace) -> None:
    out = Path(args.output)
    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} already exists. The eval set is meant to be FROZEN — "
            f"re-rolling it would silently invalidate every prior comparison. "
            f"Pass --force only if you really mean to."
        )
    n = carve_eval(Path(args.input), out, n=args.n, seed=args.seed)
    print(f"wrote {n} eval records → {out}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sft = sub.add_parser("sft", help="convert data/sft.jsonl → data/mlx/{train,valid}.jsonl")
    sft.add_argument("--input", default=str(REPO_ROOT / "data" / "sft.jsonl"))
    sft.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "mlx"))
    sft.add_argument("--n", type=int, default=0, help="0 = use all rows")
    sft.add_argument("--valid-frac", type=float, default=0.1)
    sft.add_argument("--seed", type=int, default=0)
    sft.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH),
                     help="path to held-out eval JSONL whose prompts will be excluded from train/valid")

    dpo = sub.add_parser("dpo", help="convert data/dpo.jsonl → data/dpo_mlx/{train,valid}.jsonl")
    dpo.add_argument("--input", default=str(REPO_ROOT / "data" / "dpo.jsonl"))
    dpo.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "dpo_mlx"))
    dpo.add_argument("--valid-frac", type=float, default=0.1)
    dpo.add_argument("--seed", type=int, default=0)
    dpo.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH),
                     help="path to held-out eval JSONL whose prompts will be excluded from train/valid")

    grpo = sub.add_parser("grpo", help="convert data/sft.jsonl → data/grpo/{train,valid}.jsonl")
    grpo.add_argument("--input", default=str(REPO_ROOT / "data" / "sft.jsonl"))
    grpo.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "grpo"))
    grpo.add_argument("--valid-n", type=int, default=30,
                      help="number of records to hold out as the GRPO valid set (separate from data/eval.jsonl)")
    grpo.add_argument("--seed", type=int, default=0)
    grpo.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH),
                      help="path to held-out eval JSONL whose prompts will be excluded entirely")

    carve = sub.add_parser("carve-eval",
                           help="freeze a held-out evaluation set (refuses to overwrite without --force)")
    carve.add_argument("--input", default=str(REPO_ROOT / "data" / "sft.jsonl"))
    carve.add_argument("--output", default=str(DEFAULT_EVAL_PATH))
    carve.add_argument("--n", type=int, default=30)
    carve.add_argument("--seed", type=int, default=0)
    carve.add_argument("--force", action="store_true",
                       help="overwrite existing eval set (strongly discouraged)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "sft":
        _sft_main(args)
    elif args.cmd == "dpo":
        _dpo_main(args)
    elif args.cmd == "grpo":
        _grpo_main(args)
    elif args.cmd == "carve-eval":
        _carve_main(args)


if __name__ == "__main__":
    main()
