import json
import re
import requests
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Dict, Any, Optional

# --- Sentence-level helpers ---

def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace.

    Naive but good enough for English prose at this size; robust parsing
    isn't worth the dependency cost for what we use it for.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def length_ratio_score(
    source: str,
    output: str,
    soft_cap: float = 1.3,
    hard_cap: float = 2.0,
) -> float:
    """Reward 1.0 while output/source word ratio ≤ soft_cap; decays linearly to
    0 by hard_cap. Catches simplifications that pad rather than condense.

    Defaults are loose: A2 simplification often unpacks clauses and grows
    ~1.2-1.3×, so penalty only kicks in past 1.3×.
    """
    src_words = len(_clean(source).split())
    if src_words == 0:
        return 0.0
    out_words = len(_clean(output).split())
    ratio = out_words / src_words
    if ratio <= soft_cap:
        return 1.0
    if ratio >= hard_cap:
        return 0.0
    return 1.0 - (ratio - soft_cap) / (hard_cap - soft_cap)

# --- Abstract Base Classes for Extensibility ---

class BaseJudge(ABC):
    @abstractmethod
    def evaluate(self, prompt: str) -> Dict[str, Any]:
        pass

class BaseTest(ABC):
    @abstractmethod
    def run(self, text: str, judge: BaseJudge) -> float:
        """Returns a reward score between 0.0 and 1.0"""
        pass

# --- Implementations ---

class LocalJudge(BaseJudge):
    """OpenAI-compatible chat endpoint. Works for LM Studio / vLLM / Ollama
    (no auth) and for hosted gateways like OpenRouter (pass `api_key` to
    send `Authorization: Bearer <key>`)."""
    def __init__(
        self,
        base_url: str,
        model_name: str,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ):
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model_name
        self.temperature = temperature
        self.api_key = api_key

    def evaluate(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=180)
            response.raise_for_status()
            raw_content = response.json()["choices"][0]["message"]["content"]
            return self._parse_json(raw_content)
        except Exception as e:
            print(f"Error communicating with judge at {self.endpoint}: {e}")
            return {"error": str(e)}

    def _parse_json(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        # Fallback: extract first {...} block.
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise


def _clean(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()


def truncate_to_words(text: str, n_words: int = 100) -> str:
    """Deterministic head-of-text excerpt."""
    words = _clean(text).split()
    if len(words) <= n_words:
        return " ".join(words)
    return " ".join(words[:n_words])


def windowed_excerpts(text: str, n_words: int = 100, n_windows: int = 3) -> List[str]:
    """Evenly-spaced n-word windows across the text. Used so that an A2-looking
    intro can't carry a text whose body is B1-complex."""
    words = _clean(text).split()
    if len(words) <= n_words:
        return [" ".join(words)]
    if n_windows <= 1:
        return [" ".join(words[:n_words])]
    last_start = len(words) - n_words
    starts = [round(i * last_start / (n_windows - 1)) for i in range(n_windows)]
    seen = set()
    out = []
    for s in starts:
        if s in seen:
            continue
        seen.add(s)
        out.append(" ".join(words[s:s + n_words]))
    return out


class PacingVarietyTest(BaseTest):
    """Score sentence-opening diversity, 0..1.

    Heuristic: take the first `opening_tokens` words of each sentence
    (lowercased). The more sentences share an opening, the lower the score.
    Uses 1 - sum(p_i^2) over opening counts (an HHI-style diversity index).

    Deliberately *general*: it doesn't pattern-match specific phrases, so the
    distillation prompt can't game it by token-substitution. A model that
    produces "X is Y. Z is W. P is Q." rhythms will score low regardless of
    which nouns it uses.
    """

    def __init__(self, opening_tokens: int = 2):
        self.opening_tokens = opening_tokens

    def run(self, text: str, judge: Optional[BaseJudge] = None) -> float:
        sentences = split_sentences(text)
        if len(sentences) < 2:
            return 1.0
        openings: list[tuple[str, ...]] = []
        for s in sentences:
            tokens = re.findall(r"\w+", s.lower())[: self.opening_tokens]
            openings.append(tuple(tokens))
        counts = Counter(openings)
        total = len(openings)
        return 1.0 - sum((c / total) ** 2 for c in counts.values())


class DifficultyRankingTest(BaseTest):
    """
    Classifies a candidate text as A1 / A2 / B1+ using few-shot reference samples.
    Reward is 1.0 only when the judge labels the text as A2.

    To stabilize the judgment, we:
      - truncate the candidate to a standard ~100-word excerpt,
      - ask the judge for an explicit level label and rubric-based reasoning,
      - run the judge n_votes times and majority-vote.
    """
    VALID_LEVELS = {"<A1", "A1", "A2", "B1", "B2+", "NA"}

    def __init__(
        self,
        a1_samples: List[str],
        b1_samples: List[str],
        a2_samples: Optional[List[str]] = None,
        n_words: int = 100,
    ):
        # Length-normalize the reference samples too — stops the judge from
        # using "this candidate is shorter than B1 examples" as a signal.
        self.a1_samples = [truncate_to_words(s, n_words) for s in a1_samples]
        self.b1_samples = [truncate_to_words(s, n_words) for s in b1_samples]
        self.a2_samples = [truncate_to_words(s, n_words) for s in (a2_samples or [])]
        self.n_words = n_words

    def _judge_call(self, text: str, judge: BaseJudge) -> Dict[str, Any]:
        excerpt = truncate_to_words(text, self.n_words)
        prompt = self._build_prompt(excerpt)
        return judge.evaluate(prompt)

    def _normalize_level(self, result: Dict[str, Any]) -> str:
        raw = result.get("level") if isinstance(result, dict) else None
        level = (raw or "").strip().upper().replace("B2", "B2+") if raw else ""
        if level in {"BELOW A1", "PRE-A1", "<<A1"}: level = "<A1"
        if level in {"B2", "B2+", "C1", "C2", "C1+", "C2+"}: level = "B2+"
        if level not in self.VALID_LEVELS:
            return "NA"
        return level

    def classify(self, text: str, judge: BaseJudge) -> str:
        """Return the CEFR level label ('A1', 'A2', 'B1', 'B2+', '<A1', or 'NA').

        Used by the eval harness: we want the actual label, not a binary score,
        so we can distinguish too-easy from too-hard failures.
        """
        return self._normalize_level(self._judge_call(text, judge))

    def run(self, text: str, judge: BaseJudge) -> float:
        result = self._judge_call(text, judge)
        level = self._normalize_level(result)
        feats = result.get("complex_features", "") if isinstance(result, dict) else ""
        print(f"[judge] level={level!r} feats={str(feats)[:120]}")
        if level == "NA":
            print(f"[warn] unrecognized level — scoring 0")
            return 0.0
        return 1.0 if level == "A2" else 0.0

    def _format_block(self, header: str, samples: List[str]) -> str:
        if not samples:
            return ""
        body = "\n\n".join(f"<sample>\n{s}\n</sample>" for s in samples)
        return f"### REFERENCE: {header}\n{body}\n"

    def _build_prompt(self, candidate_text: str) -> str:
        a1_block = self._format_block("A1 (Beginner)", self.a1_samples)
        a2_block = self._format_block("A2 (Elementary) — TARGET LEVEL", self.a2_samples)
        b1_block = self._format_block("B1 (Intermediate)", self.b1_samples)

        return f"""You are an expert CEFR examiner. Classify the CANDIDATE TEXT as exactly one of:
**<A1**, **A1**, **A2**, **B1**, **B2+**, or **NA**.

- Use **<A1** for text below the A1 floor (e.g. random tokens, single labels, gibberish, but still language-like).
- Use **B2+** for anything clearly at B2 or above (complex argument, advanced vocabulary, dense subordination).
- Use **NA** when the input is not classifiable as English prose (empty, code, non-English, pure numbers/symbols).

## CEFR rubric (concise)

**A1 — Beginner**
- Very short, simple sentences; mostly present tense.
- Concrete, high-frequency vocabulary (everyday objects, places, family).
- Lists, fragments, and direct instructions are common.
- Almost no subordinate clauses; coordination via "and"/"but" only.

**A2 — Elementary (TARGET)**
- Simple sentences, often joined with basic connectors (and, but, because, so, when, if).
- Past simple, present simple, present continuous, "going to" / "will" futures are normal.
- ALLOWED at A2 (do NOT count these against A2 by themselves):
  * Present perfect in simple forms ("I have lived", "we have forgotten").
  * Simple "if" conditionals ("if X, we Y").
  * Fixed-phrase passives like "is called", "is named", "is made of".
  * Short relative clauses with "that"/"who" referring to concrete things ("things that are important").
- Vocabulary is everyday + a few topic-specific concrete nouns.
- Mostly one idea per sentence; rare deep subordination; no extended abstract argument.

**B1 — Intermediate** — distinguished from A2 by *productive*, non-fixed complex grammar and lower-frequency vocabulary:
- MULTIPLE productive passive-voice constructions in the same passage ("was built with...", "were transported by...", "were damaged by...", "are decorated with"). One fixed-phrase passive does NOT count.
- Lower-frequency / domain-specific vocabulary clustered together (e.g. "mausoleum", "archways", "environmental damage", "rebellion", "conserve").
- Abstract argumentation or opinion that requires inference, not just simple description.
- Discourse markers like however, although, on the other hand, despite, according to legend.
- Idiomatic or figurative usage.

## Tiebreaker (IMPORTANT)
- Productive passive voice in 2+ different sentences → B1.
- Cluster of low-frequency content vocabulary (3+ uncommon words in a paragraph) → B1.
- Abstract argument or opinion + cohesive discourse markers → B1.
- A simple narrative or instructional text using ALLOWED A2 grammar is A2 even if a few sentences are slightly long.
- Do NOT label as A2 just because the text is "not as complex as the B1 reference samples." But also do NOT label as B1 just because a text has one passive ("is called") or a present perfect.

## Decision procedure
1. Read the candidate. Identify its 2–3 most complex sentences.
2. For each, note: tense, clause structure (simple / coordinated / subordinated / relative / passive), and any abstract or low-frequency vocabulary.
3. Compare those features to the rubric above and to the reference samples.
4. Be strict: if the text would be HARD for an A2 learner (subordinate clauses with abstract content, passive voice, idioms, multi-clause arguments), label it B1 — do NOT label it A2 just because it sits "between" A1 and B1.
5. Choose the single best label.

{a1_block}{a2_block}{b1_block}
### CANDIDATE TEXT (~{self.n_words} words)
{candidate_text}

## Output
Respond with ONLY a JSON object, no prose, no markdown fences:
{{
  "complex_features": "list the strongest complexity features you found (passives, subordination, abstract vocab, idioms, etc.) — empty string if none",
  "reasoning": "one or two sentences justifying the label using the rubric",
  "level": "<A1" | "A1" | "A2" | "B1" | "B2+" | "NA",
  "confidence": 0.0
}}
"""

class RewardVerifier:
    def __init__(self, judge: BaseJudge):
        self.judge = judge
        self.tests: List[BaseTest] = []

    def add_test(self, test: BaseTest):
        self.tests.append(test)

    def verify(self, text: str) -> float:
        if not self.tests:
            return 0.0
        scores = [test.run(text, self.judge) for test in self.tests]
        return sum(scores) / len(scores)

# --- Main Execution Script ---

if __name__ == "__main__":
    SAMPLES_FILE = "samples.jsonl"
    def load_samples(file_path: str):
        buckets: Dict[str, List[str]] = {"A1": [], "A2": [], "B1": []}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if data['level'] in buckets:
                    buckets[data['level']].append(data['text'])
        return buckets

    samples = load_samples(SAMPLES_FILE)

    LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
    MODEL_NAME = "google/gemma-4-26b-a4b"

    candidate_a2_text = "In 1885, Hermann Ebbinghaus studied his own memory. He wanted to know how quickly he forgot new information. He gave himself tests after different periods of time to see how much he forgot each time. The result is called The Forgetting Curve. [1] We forget the fastest in the first 24 hours. [2] And then we forget more and more but the speed slows down over time. [3] After about six days, we have forgotten most of the new information. [4] If we revise the information … [5] ... we make the memory stronger and we slow down the speed of forgetting. [6] If we revise again and again, we can leave longer and longer periods of time between revisions. Time is not the only thing that makes us forget something. If we are tired or under stress, we forget faster. If the information was difficult to understand we will forget it faster too. But things that are very important or meaningful to us are easier to remember."
    candidate_b1_text = """The Taj Mahal
The Taj Mahal (pronounced /ˌtɑːdʒ mə'hɑːl/) is a famous mausoleum next to the river Yamuna in the Indian city of Agra. A mausoleum is a building where people bury the dead. The name Taj Mahal means 'the crown of palaces'. 1. The most famous part of the Taj Mahal is the large white dome in the centre. It is 35 metres high and is surrounded by four smaller domes. The rooms inside the building are decorated with beautiful archways and precious stones in the walls. The buildings are surrounded by gardens with pathways, pools, fountains and green gardens. 2. The construction of the Taj Mahal began in 1632 and finished in 1653. It was built with materials from all over India and Asia, but the main material is white marble. Historians believe that the materials were transported by over 1,000 elephants for the construction. 3. The emperor Shah Jahan built the Taj Mahal as a burial place for his wife, Mumtaz Mahal. According to legend, he wanted to build another Taj Mahal in black on the other side of the river, but this never happened. During the Indian Rebellion of 1857, many parts of the Taj Mahal were damaged by British soldiers, who took some of the precious stones from its walls. Over the years, the Taj Mahal has suffered from environmental damage, and there have been many government attempts to conserve its beauty."""

    judge = LocalJudge(base_url=LM_STUDIO_URL, model_name=MODEL_NAME)
    verifier = RewardVerifier(judge)
    verifier.add_test(DifficultyRankingTest(
        a1_samples=samples["A1"],
        b1_samples=samples["B1"],
        a2_samples=samples["A2"],
        n_words=100,
    ))

    print(f"--- Running Verifier with {MODEL_NAME} ---")

    print("\n[TESTING A2 CANDIDATE] (expect PASS)")
    reward_a2 = verifier.verify(candidate_a2_text)
    print(f"Result: {'PASS' if reward_a2 == 1.0 else 'FAIL'} | Score: {reward_a2}")

    print("\n[TESTING B1 CANDIDATE] (expect FAIL — should not be classified as A2)")
    reward_b1 = verifier.verify(candidate_b1_text)
    print(f"Result: {'PASS' if reward_b1 == 0.0 else 'FAIL'} | Score: {reward_b1} (1.0 means wrongly classified as A2)")
