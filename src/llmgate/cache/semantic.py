"""Semantic response caching.

An exact-match cache on prompt strings catches almost nothing in production.
Users ask "what were Q3 revenues" and "how much revenue in Q3" and "Q3 revenue
figures" — three different strings, one question, three full-price API calls.

A semantic cache embeds the prompt and returns a stored response when a previous
prompt is close enough in vector space. Embedding is roughly four orders of
magnitude cheaper than generation, so the trade is almost always worth making.

**The threshold is the entire design problem.** Too loose and you return the Q2
answer to a Q3 question, which is worse than no cache at all — a wrong answer
served instantly and confidently. Too tight and it behaves like an exact-match
cache and saves nothing.

`tune_threshold` derives it from labelled pairs at a target precision rather than
picking a number that feels right. The default of 0.92 is deliberately
conservative: on a cost-saving feature, a false hit costs correctness and a miss
costs pennies. Those are not symmetric, so the threshold should not be set as if
they were.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class CacheEntry:
    prompt: str
    response: str
    embedding: np.ndarray
    input_tokens: int
    output_tokens: int
    model: str
    created_at: float = field(default_factory=time.time)
    hits: int = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


@dataclass
class CacheResult:
    hit: bool
    response: str | None = None
    similarity: float = 0.0
    matched_prompt: str | None = None
    # Tokens the hit avoided spending. This is the number the benchmark sums,
    # and keeping it per-hit rather than as a running total means a cache hit on
    # a long prompt is not scored the same as one on a short prompt.
    saved_input_tokens: int = 0
    saved_output_tokens: int = 0


class SemanticCache:
    """Vector-similarity cache over prompts.

    Linear scan over stored embeddings. At the scale a gateway cache operates —
    thousands of entries, not millions — a numpy dot product over the full set
    takes microseconds, and an ANN index would add a dependency and an
    approximation for no measurable gain. The interface allows swapping it later
    without touching callers.
    """

    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.92,
        max_entries: int = 1000,
        ttl_seconds: float | None = 3600.0,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.max_entries = max_entries
        # TTL exists because a cached answer about "this quarter" becomes wrong
        # when the quarter changes, and nothing in the prompt signals that.
        self.ttl_seconds = ttl_seconds

        self._entries: list[CacheEntry] = []
        self._matrix: np.ndarray | None = None
        self.lookups = 0
        self.hits = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def _rebuild_matrix(self) -> None:
        if not self._entries:
            self._matrix = None
            return
        self._matrix = np.vstack([e.embedding for e in self._entries])

    def _evict_expired(self) -> int:
        if self.ttl_seconds is None:
            return 0
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.age_seconds < self.ttl_seconds]
        removed = before - len(self._entries)
        if removed:
            self._rebuild_matrix()
        return removed

    def lookup(self, prompt: str) -> CacheResult:
        self.lookups += 1
        self._evict_expired()

        if not self._entries or self._matrix is None:
            return CacheResult(hit=False)

        query = self.embedder.encode([prompt])[0]
        # Embeddings are L2-normalised, so the dot product is cosine similarity.
        similarities = self._matrix @ query

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < self.threshold:
            return CacheResult(hit=False, similarity=best_score)

        entry = self._entries[best_idx]
        entry.hits += 1
        self.hits += 1

        return CacheResult(
            hit=True,
            response=entry.response,
            similarity=best_score,
            matched_prompt=entry.prompt,
            saved_input_tokens=entry.input_tokens,
            saved_output_tokens=entry.output_tokens,
        )

    def store(
        self,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> None:
        embedding = self.embedder.encode([prompt])[0]

        self._entries.append(CacheEntry(
            prompt=prompt,
            response=response,
            embedding=embedding,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        ))

        # Evict least-used first rather than oldest. A prompt asked fifty times
        # is worth keeping over one asked once, regardless of which arrived
        # first — LRU by insertion order would discard exactly the wrong entry.
        if len(self._entries) > self.max_entries:
            self._entries.sort(key=lambda e: (e.hits, -e.created_at))
            self._entries = self._entries[-self.max_entries:]

        self._rebuild_matrix()

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "lookups": self.lookups,
            "hits": self.hits,
            "hit_rate": round(self.hit_rate, 4),
            "threshold": self.threshold,
            "total_entry_hits": sum(e.hits for e in self._entries),
        }

    def clear(self) -> None:
        self._entries.clear()
        self._matrix = None
        self.lookups = 0
        self.hits = 0


def tune_threshold(
    embedder: Embedder,
    labelled_pairs: list[tuple[str, str, bool]],
    target_precision: float = 0.99,
) -> tuple[float, dict]:
    """Derive the similarity threshold from labelled prompt pairs.

    Each pair is (prompt_a, prompt_b, should_share_an_answer). The threshold is
    the lowest cutoff at which pairs above it are genuinely equivalent at least
    `target_precision` of the time.

    Target precision is 0.99 rather than something like 0.90 because the errors
    are not symmetric. A false cache hit returns a *wrong answer* — the Q2
    figure for a Q3 question, served instantly and with no indication anything
    is off. A miss costs one API call. Tuning for accuracy would trade many
    correct answers for a few cents.
    """
    texts = [p for pair in labelled_pairs for p in pair[:2]]
    embeddings = embedder.encode(texts)

    scored: list[tuple[float, bool]] = []
    for i, (_a, _b, equivalent) in enumerate(labelled_pairs):
        sim = float(embeddings[2 * i] @ embeddings[2 * i + 1])
        scored.append((sim, equivalent))

    scored.sort(key=lambda pair: -pair[0])

    threshold = 1.0
    correct = 0
    for i, (sim, equivalent) in enumerate(scored, start=1):
        correct += int(equivalent)
        if correct / i >= target_precision:
            threshold = sim
        else:
            break

    equivalent_sims = [s for s, eq in scored if eq]
    different_sims = [s for s, eq in scored if not eq]

    return round(threshold, 4), {
        "pairs": len(scored),
        "equivalent_mean_similarity": round(
            float(np.mean(equivalent_sims)), 4) if equivalent_sims else None,
        "different_max_similarity": round(
            float(max(different_sims)), 4) if different_sims else None,
        "recall_at_threshold": round(
            sum(1 for s in equivalent_sims if s >= threshold) / len(equivalent_sims), 4
        ) if equivalent_sims else None,
    }


def prompt_fingerprint(prompt: str) -> str:
    """Stable id for a prompt, for trace correlation."""
    return hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).hexdigest()
