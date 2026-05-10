"""Run a single (or batch of) simplification(s) through a trained adapter.

Useful for ad-hoc qualitative review or debugging individual outputs without
involving the eval harness or judge.

Usage:
    uv run python -m langsimp.inference.generate --adapter base "Complex text here..."
    uv run python -m langsimp.inference.generate --adapter adapters/sft/latest --file input.txt
    cat input.txt | uv run python -m langsimp.inference.generate --adapter adapters/sft/latest

Multi-paragraph input (file or stdin) is split on blank lines and each
paragraph is generated separately.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, drop empty chunks, collapse internal whitespace.

    Same shape as sources._split_paragraphs but kept here to avoid pulling
    in the Wikipedia-fetching module just for one helper.
    """
    out: list[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(re.sub(r"\s+", " ", chunk))
    return out


def read_input(text: Optional[str], file_path: Optional[Path], stdin: Optional[TextIO]) -> list[str]:
    """Decide where the input comes from and return a list of paragraphs.

    Precedence: positional `text` > `file_path` > `stdin`. Raises ValueError
    if all three are absent.
    """
    if text:
        return [text.strip()]
    if file_path is not None:
        return _split_paragraphs(Path(file_path).read_text())
    if stdin is not None:
        data = stdin.read()
        if data.strip():
            return _split_paragraphs(data)
    raise ValueError("no input provided (need text arg, --file, or stdin)")


def format_output(sources: list[str], outputs: list[str], show_source: bool) -> str:
    """Render the (source, output) pairs as a single string.

    For one paragraph we just print the simplification. For many, we put
    light dividers between them so it's clear where one ends.
    """
    lines: list[str] = []
    multi = len(outputs) > 1
    for i, (src, out) in enumerate(zip(sources, outputs), 1):
        if multi:
            lines.append(f"\n--- paragraph {i} ---")
        if show_source:
            lines.append(f"COMPLEX ({len(src.split())}w):")
            lines.append(src)
            lines.append("")
            lines.append(f"SIMPLE ({len(out.split())}w):")
        lines.append(out)
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text", nargs="?", default=None, help="text to simplify (positional)")
    p.add_argument("--file", default=None, help="read paragraphs from this file")
    p.add_argument("--adapter", required=True,
                   help="path to adapter dir, or 'base' for no adapter")
    p.add_argument("--model", default="mlx-community/gemma-3-1b-it-bf16")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--show-source", action="store_true",
                   help="print the source paragraph alongside each output")
    args = p.parse_args()

    # Avoid hanging on a TTY: only pull from stdin when something's piped in.
    stdin = None if sys.stdin.isatty() else sys.stdin
    paragraphs = read_input(args.text, Path(args.file) if args.file else None, stdin)

    # Late import so --help works even if mlx-lm has issues.
    from langsimp.inference.engine import load_model_with_adapter, make_generate_fn
    adapter_path = None if args.adapter == "base" else args.adapter
    model, tokenizer = load_model_with_adapter(args.model, adapter_path)
    gen = make_generate_fn(model, tokenizer, max_tokens=args.max_tokens)

    outputs = [gen(para) for para in paragraphs]
    sys.stdout.write(format_output(paragraphs, outputs, show_source=args.show_source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
