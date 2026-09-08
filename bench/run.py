"""Benchmark runner.

Measures each gateway feature in isolation and in combination, because
"the gateway saved 60%" is an aggregate, not a measurement. Knowing how much
came from caching versus routing is what makes the number actionable.

    python -m bench.run                    # fake client, no key needed
    python -m bench.run --real             # real API calls
    python -m bench.run --real --embedder sentence-transformers
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bench.workload import build_workload, cache_ground_truth
from llmgate.cache.semantic import SemanticCache
from llmgate.clients import AnthropicClient, FakeClient, HashEmbedder
from llmgate.gateway import Blocked, Gateway
from llmgate.trace.span import Tracer

ARMS = [
    ("baseline", {"enable_cache": False, "enable_routing": False, "enable_guard": True}),
    ("routing_only", {"enable_cache": False, "enable_routing": True, "enable_guard": True}),
    ("cache_only", {"enable_cache": True, "enable_routing": False, "enable_guard": True}),
    ("cache_and_routing", {"enable_cache": True, "enable_routing": True, "enable_guard": True}),
]


def run_arm(name, flags, requests, client, embedder, threshold):
    cache = SemanticCache(embedder, threshold=threshold, ttl_seconds=None)
    tracer = Tracer()
    gateway = Gateway(client, cache=cache, tracer=tracer, **flags)

    truth = cache_ground_truth(requests)
    false_hits = 0
    blocked = 0

    for req in requests:
        try:
            response = gateway.complete(
                req.prompt, context_tokens=req.context_tokens
            )
            # A cache hit against a prompt from a different semantic group is a
            # wrong answer served instantly. This is the number that decides
            # whether the threshold is safe, and it is worth more than hit rate.
            if response.cached and response.span.cache_similarity:
                matched = cache.lookup.__self__
                hit = None
                for entry in matched._entries:
                    if entry.response == response.text:
                        hit = entry
                        break
                if hit and truth.get(hit.prompt) != truth.get(req.prompt):
                    false_hits += 1
        except Blocked:
            blocked += 1

    summary = tracer.summary()
    summary["arm"] = name
    summary["false_cache_hits"] = false_hits
    summary["guard_blocked"] = blocked
    summary["cache"] = cache.stats()
    return summary, tracer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="use the Anthropic API")
    parser.add_argument("--embedder", default="hash",
                        help="hash | sentence-transformers")
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()

    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    client = AnthropicClient() if args.real else FakeClient()
    embedder = (
        HashEmbedder() if args.embedder == "hash"
        else __import__(
            "llmgate.clients", fromlist=["SentenceTransformerEmbedder"]
        ).SentenceTransformerEmbedder()
    )

    requests = build_workload(repeats=args.repeats)
    print(f"workload: {len(requests)} requests, "
          f"{len({r.prompt for r in requests})} unique prompts")
    print(f"client: {'anthropic (real)' if args.real else 'fake'} | "
          f"embedder: {args.embedder} | threshold: {args.threshold}\n")

    results = []
    for name, flags in ARMS:
        print(f"running {name}...")
        summary, _tracer = run_arm(
            name, flags, requests, client, embedder, args.threshold
        )
        results.append(summary)

    print(f"\n{'arm':<20}{'cost':>10}{'vs base':>10}{'hits':>7}{'false':>7}{'p95 ms':>9}")
    print("-" * 63)
    base_cost = results[0]["total_cost_usd"]
    for r in results:
        delta = (1 - r["total_cost_usd"] / base_cost) * 100 if base_cost else 0
        print(f"{r['arm']:<20}${r['total_cost_usd']:>9.5f}{delta:>9.1f}%"
              f"{r['cache_hits']:>7}{r['false_cache_hits']:>7}"
              f"{r['latency']['api_p95_ms']:>9.1f}")

    print(f"\nrouting distribution ({results[-1]['arm']}):")
    for tier, count in sorted(results[-1]["routed_by_tier"].items()):
        print(f"  {tier:<12} {count}")
    print(f"\nguard blocked: {results[-1]['guard_blocked']} injection attempts")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "real_api": args.real,
        "embedder": args.embedder,
        "threshold": args.threshold,
        "workload_size": len(requests),
        "unique_prompts": len({r.prompt for r in requests}),
        "arms": results,
    }, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
