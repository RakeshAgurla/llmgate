"""Request tracing.

Every request through the gateway produces a structured record: which model was
chosen and why, whether the cache was hit, what the guard found, tokens in and
out, latency, cost, and what the request would have cost without the gateway.

That last field is the point. "We spent $40" says nothing. "We spent $40 instead
of $115" is the claim, and it requires computing the counterfactual on every
request rather than estimating it afterwards.

Spans serialise to JSON in a shape that maps onto OpenTelemetry attributes, so
this exports to a real collector without restructuring. Adding the otel SDK here
would pull in a large dependency for a project whose interesting part is the
routing and caching logic, not the transport.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean


@dataclass
class Span:
    trace_id: str
    prompt_fingerprint: str
    started_at: float

    cache_hit: bool = False
    cache_similarity: float = 0.0

    routed_tier: str = ""
    routed_model: str = ""
    routing_reason: str = ""
    escalated: bool = False

    guard_severity: str = "none"
    guard_blocked: bool = False
    guard_patterns: list[str] = field(default_factory=list)

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    cost_usd: float = 0.0
    # What this request would have cost with no gateway: no cache, everything to
    # the standard model.
    baseline_cost_usd: float = 0.0

    error: str | None = None

    @property
    def saved_usd(self) -> float:
        return max(0.0, self.baseline_cost_usd - self.cost_usd)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["saved_usd"] = round(self.saved_usd, 8)
        return d

    def to_otel_attributes(self) -> dict:
        """Flattened into OTel semantic-convention-style keys."""
        return {
            "gen_ai.request.model": self.routed_model,
            "gen_ai.usage.input_tokens": self.input_tokens,
            "gen_ai.usage.output_tokens": self.output_tokens,
            "gen_ai.response.latency_ms": self.latency_ms,
            "llmgate.cache.hit": self.cache_hit,
            "llmgate.cache.similarity": self.cache_similarity,
            "llmgate.route.tier": self.routed_tier,
            "llmgate.route.escalated": self.escalated,
            "llmgate.guard.severity": self.guard_severity,
            "llmgate.cost.usd": self.cost_usd,
            "llmgate.cost.baseline_usd": self.baseline_cost_usd,
        }


class Tracer:
    def __init__(self):
        self.spans: list[Span] = []

    def start(self, prompt_fingerprint: str) -> Span:
        span = Span(
            trace_id=uuid.uuid4().hex[:12],
            prompt_fingerprint=prompt_fingerprint,
            started_at=time.time(),
        )
        self.spans.append(span)
        return span

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def total_baseline_cost(self) -> float:
        return sum(s.baseline_cost_usd for s in self.spans)

    @property
    def total_saved(self) -> float:
        return sum(s.saved_usd for s in self.spans)

    @property
    def savings_pct(self) -> float:
        base = self.total_baseline_cost
        return (self.total_saved / base * 100) if base else 0.0

    def latency_percentiles(self) -> dict[str, float]:
        """Percentiles, excluding cache hits.

        Mixing them makes the number meaningless: a cache hit returns in under a
        millisecond and an API call takes a second, so the blended figure just
        tracks the hit rate rather than describing either path.
        """
        api = sorted(s.latency_ms for s in self.spans if not s.cache_hit and not s.error)
        cached = sorted(s.latency_ms for s in self.spans if s.cache_hit)

        def pct(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            return round(values[min(len(values) - 1, round(p * (len(values) - 1)))], 2)

        return {
            "api_p50_ms": pct(api, 0.50),
            "api_p95_ms": pct(api, 0.95),
            "api_mean_ms": round(mean(api), 2) if api else 0.0,
            "cache_p50_ms": pct(cached, 0.50),
            "cache_p95_ms": pct(cached, 0.95),
            "api_calls": len(api),
            "cache_hits": len(cached),
        }

    def summary(self) -> dict:
        n = len(self.spans)
        by_tier: dict[str, int] = {}
        for s in self.spans:
            if s.routed_tier:
                by_tier[s.routed_tier] = by_tier.get(s.routed_tier, 0) + 1

        return {
            "requests": n,
            "cache_hits": sum(1 for s in self.spans if s.cache_hit),
            "cache_hit_rate": round(
                sum(1 for s in self.spans if s.cache_hit) / n, 4) if n else 0.0,
            "escalations": sum(1 for s in self.spans if s.escalated),
            "guard_blocks": sum(1 for s in self.spans if s.guard_blocked),
            "guard_flags": sum(1 for s in self.spans if s.guard_severity != "none"),
            "routed_by_tier": by_tier,
            "total_cost_usd": round(self.total_cost, 6),
            "baseline_cost_usd": round(self.total_baseline_cost, 6),
            "saved_usd": round(self.total_saved, 6),
            "savings_pct": round(self.savings_pct, 2),
            "latency": self.latency_percentiles(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "summary": self.summary(),
            "spans": [s.to_dict() for s in self.spans],
        }, indent=2))

    def render(self) -> str:
        s = self.summary()
        lines = [
            (
                f"requests {s['requests']}  cache hits {s['cache_hits']} "
                f"({s['cache_hit_rate']:.1%})  escalations {s['escalations']}"
            ),
            f"routed: {s['routed_by_tier']}",
            "",
            f"  cost with gateway    ${s['total_cost_usd']:.5f}",
            f"  cost without         ${s['baseline_cost_usd']:.5f}",
            f"  saved                ${s['saved_usd']:.5f}  ({s['savings_pct']:.1f}%)",
            "",
            (
                f"  api  p50 {s['latency']['api_p50_ms']:>8.1f} ms   "
                f"p95 {s['latency']['api_p95_ms']:>8.1f} ms   n={s['latency']['api_calls']}"
            ),
            (
                f"  cache p50 {s['latency']['cache_p50_ms']:>7.1f} ms   "
                f"p95 {s['latency']['cache_p95_ms']:>7.1f} ms   n={s['latency']['cache_hits']}"
            ),
        ]
        return "\n".join(lines)
