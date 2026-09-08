"""LLM clients and embedders.

Real backends for measurement, deterministic fakes for CI.

The fakes are not just stand-ins -- they make the whole benchmark runnable with
no key, which means the routing, caching, and guard logic is exercised on every
push rather than only when someone remembers to spend money.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np


class HashEmbedder:
    """Deterministic character-ngram embedder.

    Genuinely poor at semantics, which is the point for CI: it exercises the
    cache machinery without pretending to validate cache *quality*. Quality is
    measured with a real embedder in the benchmark.
    """

    def __init__(self, dim: int = 256, ngram: int = 4):
        self.dim, self.ngram = dim, ngram

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            low = text.lower()
            for j in range(max(1, len(low) - self.ngram + 1)):
                h = int.from_bytes(
                    hashlib.blake2b(
                        low[j:j + self.ngram].encode(), digest_size=8
                    ).digest(), "little",
                )
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)


class FakeClient:
    """Deterministic client with realistic token counts and simulated latency.

    Latency is simulated rather than zero because a benchmark reporting
    sub-millisecond API calls would make the cache look pointless -- the whole
    value of a cache is the round trip it avoids.
    """

    def __init__(self, latency_ms: float = 400.0):
        self.latency_ms = latency_ms
        self.calls: list[tuple[str, str]] = []

    def complete(self, model, system, prompt, max_tokens):
        self.calls.append((model, prompt))
        time.sleep(self.latency_ms / 1000.0)
        response = f"[{model}] response to: {prompt[:60]}"
        return response, max(1, len(system + prompt) // 4), max(1, len(response) // 4)


class AnthropicClient:
    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def complete(self, model, system, prompt, max_tokens):
        # anthropic>=1.4 removed `temperature` from messages.create(). The
        # benchmark needs deterministic output anyway, and the default is
        # already deterministic enough for a cost measurement -- token counts
        # vary by a few percent, not by orders of magnitude.
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant. Be concise.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return text, response.usage.input_tokens, response.usage.output_tokens
