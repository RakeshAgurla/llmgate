import pytest

from llmgate.cache.semantic import SemanticCache, tune_threshold
from llmgate.clients import HashEmbedder


@pytest.fixture
def cache():
    return SemanticCache(HashEmbedder(), threshold=0.92, ttl_seconds=None)


def test_miss_on_empty_cache(cache):
    assert not cache.lookup("anything").hit


def test_exact_repeat_hits(cache):
    cache.store("What was Q3 revenue?", "4.82B", 100, 20, "haiku")
    result = cache.lookup("What was Q3 revenue?")
    assert result.hit and result.response == "4.82B"


def test_hit_reports_tokens_it_avoided(cache):
    """The saving is per-entry, not a flat count. A hit on a long prompt
    saves more than a hit on a short one."""
    cache.store("prompt", "response", input_tokens=5000, output_tokens=300, model="m")
    result = cache.lookup("prompt")
    assert result.saved_input_tokens == 5000
    assert result.saved_output_tokens == 300


def test_unrelated_prompt_misses(cache):
    cache.store("What was Q3 revenue?", "4.82B", 100, 20, "haiku")
    assert not cache.lookup("Describe the company's litigation history").hit


def test_threshold_governs_hits():
    loose = SemanticCache(HashEmbedder(), threshold=0.1, ttl_seconds=None)
    tight = SemanticCache(HashEmbedder(), threshold=0.999, ttl_seconds=None)
    for c in (loose, tight):
        c.store("What was Q3 revenue?", "4.82B", 100, 20, "m")
    probe = "What was Q2 revenue?"
    assert loose.lookup(probe).hit
    assert not tight.lookup(probe).hit


def test_ttl_expires_entries():
    c = SemanticCache(HashEmbedder(), threshold=0.5, ttl_seconds=-1)
    c.store("prompt", "response", 10, 10, "m")
    assert not c.lookup("prompt").hit


def test_eviction_keeps_most_used_not_most_recent():
    """LRU by insertion would discard exactly the wrong entry: a prompt asked
    fifty times matters more than one asked once, whenever it arrived."""
    c = SemanticCache(HashEmbedder(), threshold=0.99, max_entries=2, ttl_seconds=None)
    c.store("popular question here", "a", 10, 10, "m")
    for _ in range(5):
        c.lookup("popular question here")
    c.store("second entry text", "b", 10, 10, "m")
    c.store("third entry text", "c", 10, 10, "m")
    assert c.lookup("popular question here").hit


def test_hit_rate_tracking(cache):
    cache.store("q", "a", 10, 10, "m")
    cache.lookup("q")
    cache.lookup("completely different text entirely")
    assert cache.stats()["lookups"] == 2
    assert cache.stats()["hits"] == 1


def test_tune_threshold_excludes_the_false_pair():
    """Tuning targets precision, not accuracy: a threshold that admits a
    non-equivalent pair returns wrong answers instantly."""
    embedder = HashEmbedder()
    pairs = [
        ("What was Q3 revenue?", "What was Q3 revenue?", True),
        ("How much revenue in Q3?", "What was Q3 revenue?", True),
        ("What was Q3 revenue?", "What was Q2 revenue?", False),
    ]
    threshold, stats = tune_threshold(embedder, pairs, target_precision=1.0)
    assert threshold > stats["different_max_similarity"]
