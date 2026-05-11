"""Sources of complex English text used as inputs for the simplification pipeline.

Currently only Wikipedia is wired up. Future sources (e.g. ArXiv abstracts) can
be added as additional iterators with the same `WikiParagraph`-shaped output.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import requests

REST_RANDOM = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
# Wikipedia's User-Agent policy asks for a tool name, version, and contact
# method. The contact can be a repo URL (no email required). Set
# WIKI_USER_AGENT_CONTACT to your fork/repo URL or your own email if you
# expect to scrape at volume.
_DEFAULT_CONTACT = "https://github.com/anonymous/language-simplification-llm"
USER_AGENT = (
    f"language-simplification-llm/0.1 "
    f"(+{os.environ.get('WIKI_USER_AGENT_CONTACT', _DEFAULT_CONTACT)})"
)


@dataclass
class WikiParagraph:
    title: str
    url: str
    text: str

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "text": self.text}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def _random_summary(session: requests.Session, max_retries: int = 5) -> Optional[dict]:
    """Return one random article summary dict, retrying on 429 / 5xx."""
    for attempt in range(max_retries):
        r = session.get(REST_RANDOM, timeout=30)
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 2 ** attempt))
            print(f"[wiki] 429 — sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if 400 <= r.status_code < 500:
            # 404 = redirect target gone; 400 = malformed redirected URL
            # (e.g. a title containing "/"). Either way, skip this draw.
            return None
        r.raise_for_status()
        return r.json()
    return None


_BAD_TITLE_RE = re.compile(r"\b(disambiguation|list of)\b", re.I)


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for chunk in re.split(r"\n\s*\n", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        paragraphs.append(re.sub(r"\s+", " ", chunk))
    return paragraphs


def fetch_random_paragraphs(
    n: int = 1,
    min_words: int = 80,
    max_words: int = 220,
    seed: Optional[int] = None,
    inter_request_delay: float = 0.4,
) -> Iterator[WikiParagraph]:
    """Yield up to `n` paragraphs in the [min_words, max_words] range.

    `seed` is accepted for API compatibility but the random article is chosen
    server-side, so it does not produce deterministic results.
    """
    _ = seed
    session = _session()
    yielded = 0
    consecutive_misses = 0
    while yielded < n:
        if consecutive_misses > 500:
            raise RuntimeError(
                f"Could not find enough qualifying paragraphs after 500 misses "
                f"(min_words={min_words}, max_words={max_words})"
            )
        summary = _random_summary(session)
        if summary is None:
            consecutive_misses += 1
            continue
        title = summary.get("title", "")
        if _BAD_TITLE_RE.search(title) or summary.get("type") == "disambiguation":
            consecutive_misses += 1
            continue
        extract = summary.get("extract", "") or ""
        paragraphs = [
            p for p in _split_paragraphs(extract)
            if min_words <= len(p.split()) <= max_words
        ]
        if not paragraphs:
            consecutive_misses += 1
            time.sleep(inter_request_delay)
            continue
        url = (
            summary.get("content_urls", {}).get("desktop", {}).get("page")
            or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        )
        yield WikiParagraph(title=title, url=url, text=paragraphs[0])
        yielded += 1
        consecutive_misses = 0
        if yielded < n:
            time.sleep(inter_request_delay)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--min-words", type=int, default=80)
    ap.add_argument("--max-words", type=int, default=220)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    for p in fetch_random_paragraphs(args.n, args.min_words, args.max_words, args.seed):
        print(json.dumps(p.to_dict(), ensure_ascii=False))
