"""Model routing.

Not every request needs the expensive model. Extracting a date from a sentence
and reasoning about a contract clause are different problems, and paying Sonnet
prices for the first is the most common source of avoidable LLM spend.

The router classifies a request and sends it to the cheapest model likely to
handle it, with an escalation path when the cheap model's answer fails
validation.

**Why heuristic classification rather than a classifier model.** Calling a model
to decide which model to call adds latency and cost to every request, including
the ones that were going to be cheap. Heuristics are free, run in microseconds,
and are auditable — you can read why a request routed the way it did, which
matters when someone asks why their query cost what it did.

The tradeoff is real: heuristics misclassify. That is what the escalation path
is for, and what `escalation_rate` measures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    CHEAP = "cheap"
    STANDARD = "standard"
    PREMIUM = "premium"


# Published prices per million tokens. Kept as data, not inlined into the cost
# calculation, because prices change and a stale constant buried in a function
# is how a cost estimate silently becomes wrong.
MODELS = {
    Tier.CHEAP: {
        "name": "claude-haiku-4-5",
        "input": 1.00,
        "output": 5.00,
    },
    Tier.STANDARD: {
        "name": "claude-sonnet-4-6",
        "input": 3.00,
        "output": 15.00,
    },
    Tier.PREMIUM: {
        "name": "claude-opus-4-1",
        "input": 15.00,
        "output": 75.00,
    },
}

# Signals that a request needs reasoning rather than extraction. Ordered by how
# reliably each predicts difficulty in practice.
_REASONING_SIGNALS = (
    r"\bwhy\b", r"\bexplain\b", r"\bcompare\b", r"\banalyz", r"\banalys",
    r"\bevaluat", r"\brecommend", r"\bimplication", r"\btrade-?off",
    r"\bshould we\b", r"\bwhat if\b", r"\bassess\b",
)

_EXTRACTION_SIGNALS = (
    r"\bextract\b", r"\blist\b", r"\bwhat is the\b", r"\bhow much\b",
    r"\bwhen did\b", r"\bwho is\b", r"\bfind the\b", r"\bparse\b",
    r"\bconvert\b", r"\bclassif",
)

_STRUCTURED_OUTPUT = (r"\bjson\b", r"\bcsv\b", r"schema", r"\bformat as\b")


@dataclass
class RoutingDecision:
    tier: Tier
    model: str
    reason: str
    signals: dict[str, int] = field(default_factory=dict)
    escalated_from: Tier | None = None

    @property
    def was_escalated(self) -> bool:
        return self.escalated_from is not None


def _count(patterns: tuple[str, ...], text: str) -> int:
    lowered = text.lower()
    return sum(1 for p in patterns if re.search(p, lowered))


def classify(prompt: str, context_tokens: int = 0) -> RoutingDecision:
    """Choose a tier for this request.

    Length is a signal but a weak one on its own. A 10,000-token prompt asking
    "extract the invoice number" is still extraction; the length is context, not
    difficulty. So length only escalates when reasoning signals are also
    present.
    """
    reasoning = _count(_REASONING_SIGNALS, prompt)
    extraction = _count(_EXTRACTION_SIGNALS, prompt)
    structured = _count(_STRUCTURED_OUTPUT, prompt)

    signals = {
        "reasoning": reasoning,
        "extraction": extraction,
        "structured_output": structured,
        "context_tokens": context_tokens,
    }

    # Structured output with no reasoning ask is the clearest cheap case: the
    # task is shape-fitting, and the small model does it as well as the large
    # one for a fifth of the price.
    if structured and not reasoning:
        return RoutingDecision(
            Tier.CHEAP, MODELS[Tier.CHEAP]["name"],
            "structured output, no reasoning signals", signals,
        )

    if reasoning == 0 and extraction > 0:
        return RoutingDecision(
            Tier.CHEAP, MODELS[Tier.CHEAP]["name"],
            "extraction signals with no reasoning signals", signals,
        )

    # Long context plus reasoning is the case worth paying for: the model has to
    # hold a lot in mind and draw a conclusion from it.
    if reasoning >= 2 or (reasoning >= 1 and context_tokens > 8000):
        return RoutingDecision(
            Tier.STANDARD, MODELS[Tier.STANDARD]["name"],
            f"reasoning signals={reasoning}, context={context_tokens}", signals,
        )

    if reasoning == 1:
        return RoutingDecision(
            Tier.STANDARD, MODELS[Tier.STANDARD]["name"],
            "single reasoning signal", signals,
        )

    # Ambiguous. Default to cheap and let escalation catch the mistakes -- the
    # expected cost of one retry is lower than the expected cost of always
    # paying premium prices for requests that mostly did not need it.
    return RoutingDecision(
        Tier.CHEAP, MODELS[Tier.CHEAP]["name"],
        "no strong signals, defaulting to cheap with escalation available",
        signals,
    )


def escalate(decision: RoutingDecision) -> RoutingDecision | None:
    """Move one tier up. Returns None at the top."""
    order = [Tier.CHEAP, Tier.STANDARD, Tier.PREMIUM]
    idx = order.index(decision.tier)
    if idx + 1 >= len(order):
        return None
    higher = order[idx + 1]
    return RoutingDecision(
        tier=higher,
        model=MODELS[higher]["name"],
        reason=f"escalated from {decision.tier.value}",
        signals=decision.signals,
        escalated_from=decision.tier,
    )


def cost_usd(tier: Tier, input_tokens: int, output_tokens: int) -> float:
    prices = MODELS[tier]
    return (
        input_tokens / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
    )


def cost_if_all_premium(input_tokens: int, output_tokens: int) -> float:
    """What the same request would have cost without routing.

    This is the counterfactual the savings figure is measured against. Reporting
    'we spent $X' says nothing; reporting 'we spent $X instead of $Y' is the
    claim that matters, and it requires stating the baseline explicitly.
    """
    return cost_usd(Tier.STANDARD, input_tokens, output_tokens)
