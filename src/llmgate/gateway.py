"""The gateway.

Ties the four pieces together in the order that matters:

    guard  ->  cache  ->  route  ->  call  ->  store  ->  trace

**Why guard first.** Scanning before the cache means injected content never
becomes a cache entry. If the cache came first, a poisoned response could be
stored and served to later users who never triggered the guard -- the injection
outlives the attack.

**Why cache before route.** A cache hit skips model selection entirely. Routing
first would compute a decision for a request that is never sent.

**Why the counterfactual is computed on every request.** Savings claims made
after the fact are estimates. Computing what each request would have cost
without the gateway, as it happens, makes the number auditable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from llmgate.cache.semantic import SemanticCache, prompt_fingerprint
from llmgate.guard.injection import ScanResult, scan
from llmgate.route.router import (
    RoutingDecision,
    Tier,
    classify,
    cost_if_all_premium,
    cost_usd,
    escalate,
)
from llmgate.trace.span import Span, Tracer


class Blocked(RuntimeError):
    """Raised when the guard blocks a request."""

    def __init__(self, result: ScanResult):
        self.scan = result
        super().__init__(f"blocked: {result.summary()}")


@dataclass
class Response:
    text: str
    model: str
    cached: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float
    baseline_cost_usd: float
    latency_ms: float
    span: Span

    @property
    def saved_usd(self) -> float:
        return max(0.0, self.baseline_cost_usd - self.cost_usd)


class Gateway:
    def __init__(
        self,
        client,
        cache: SemanticCache | None = None,
        tracer: Tracer | None = None,
        enable_cache: bool = True,
        enable_routing: bool = True,
        enable_guard: bool = True,
    ):
        self.client = client
        self.cache = cache
        self.tracer = tracer or Tracer()
        # Feature flags exist so the benchmark can isolate each contribution.
        # Reporting "the gateway saved 60%" without knowing how much came from
        # caching versus routing is not a measurement, it is an aggregate.
        self.enable_cache = enable_cache and cache is not None
        self.enable_routing = enable_routing
        self.enable_guard = enable_guard

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 512,
        context_tokens: int = 0,
        validate=None,
    ) -> Response:
        started = time.perf_counter()
        span = self.tracer.start(prompt_fingerprint(prompt))

        # 1. Guard, before anything is cached or sent.
        if self.enable_guard:
            result = scan(prompt, source="request")
            span.guard_severity = result.severity.value
            span.guard_patterns = sorted({d.pattern for d in result.detections})
            if result.should_block:
                span.guard_blocked = True
                span.latency_ms = (time.perf_counter() - started) * 1000
                raise Blocked(result)

        # 2. Cache.
        if self.enable_cache:
            hit = self.cache.lookup(prompt)
            span.cache_similarity = round(hit.similarity, 4)
            if hit.hit:
                span.cache_hit = True
                span.input_tokens = 0
                span.output_tokens = 0
                span.cost_usd = 0.0
                span.baseline_cost_usd = cost_if_all_premium(
                    hit.saved_input_tokens, hit.saved_output_tokens
                )
                span.latency_ms = (time.perf_counter() - started) * 1000
                return Response(
                    text=hit.response,
                    model="cache",
                    cached=True,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    baseline_cost_usd=span.baseline_cost_usd,
                    latency_ms=span.latency_ms,
                    span=span,
                )

        # 3. Route.
        if self.enable_routing:
            decision = classify(prompt, context_tokens)
        else:
            decision = RoutingDecision(
                Tier.STANDARD, "claude-sonnet-4-6", "routing disabled"
            )

        span.routed_tier = decision.tier.value
        span.routed_model = decision.model
        span.routing_reason = decision.reason

        # 4. Call, with one escalation if validation rejects the cheap answer.
        text, in_tok, out_tok = self.client.complete(
            decision.model, system, prompt, max_tokens
        )
        total_cost = cost_usd(decision.tier, in_tok, out_tok)

        if validate is not None and not validate(text):
            higher = escalate(decision)
            if higher is not None:
                span.escalated = True
                span.routing_reason += f" -> escalated: {higher.reason}"
                text, in2, out2 = self.client.complete(
                    higher.model, system, prompt, max_tokens
                )
                # The failed cheap call is still paid for. Counting only the
                # successful call would understate what routing actually costs
                # when it guesses wrong.
                total_cost += cost_usd(higher.tier, in2, out2)
                in_tok += in2
                out_tok += out2
                decision = higher
                span.routed_tier = higher.tier.value
                span.routed_model = higher.model

        # 5. Store.
        if self.enable_cache:
            self.cache.store(prompt, text, in_tok, out_tok, decision.model)

        latency = (time.perf_counter() - started) * 1000
        span.input_tokens = in_tok
        span.output_tokens = out_tok
        span.cost_usd = total_cost
        span.baseline_cost_usd = cost_if_all_premium(in_tok, out_tok)
        span.latency_ms = round(latency, 2)

        return Response(
            text=text,
            model=decision.model,
            cached=False,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=total_cost,
            baseline_cost_usd=span.baseline_cost_usd,
            latency_ms=round(latency, 2),
            span=span,
        )
