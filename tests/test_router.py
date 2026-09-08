from llmgate.route.router import (
    Tier,
    classify,
    cost_if_all_premium,
    cost_usd,
    escalate,
)


def test_extraction_routes_cheap():
    assert classify("What is the invoice date").tier is Tier.CHEAP


def test_structured_output_routes_cheap():
    assert classify("Extract the total as JSON").tier is Tier.CHEAP


def test_reasoning_routes_standard():
    assert classify("Why did gross margin decline?").tier is Tier.STANDARD


def test_multiple_reasoning_signals_route_standard():
    d = classify("Compare the segments and explain the implications")
    assert d.tier is Tier.STANDARD and d.signals["reasoning"] >= 2


def test_long_context_alone_does_not_escalate():
    """Length is context, not difficulty. A 20k-token prompt asking to extract
    an invoice number is still extraction."""
    assert classify("Extract the invoice number", context_tokens=20000).tier is Tier.CHEAP


def test_long_context_with_reasoning_escalates():
    assert classify("Why did this happen?", context_tokens=20000).tier is Tier.STANDARD


def test_ambiguous_defaults_cheap_with_escalation_available():
    d = classify("Summarize this")
    assert d.tier is Tier.CHEAP and "escalation" in d.reason


def test_escalation_moves_one_tier():
    cheap = classify("Extract the date")
    higher = escalate(cheap)
    assert higher.tier is Tier.STANDARD and higher.escalated_from is Tier.CHEAP


def test_escalation_stops_at_top():
    from llmgate.route.router import RoutingDecision
    top = RoutingDecision(Tier.PREMIUM, "opus", "test")
    assert escalate(top) is None


def test_cheap_tier_costs_less():
    assert cost_usd(Tier.CHEAP, 10000, 500) < cost_usd(Tier.STANDARD, 10000, 500)


def test_baseline_is_the_standard_tier():
    """The savings counterfactual must be an explicit baseline, not an
    estimate made afterwards."""
    assert cost_if_all_premium(1000, 100) == cost_usd(Tier.STANDARD, 1000, 100)
