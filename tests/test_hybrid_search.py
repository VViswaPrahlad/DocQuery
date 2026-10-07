"""
tests/test_hybrid_search.py

Unit tests for reciprocal_rank_fusion(). Pure logic, no embeddings/API
calls needed, so these run instantly and in CI.

Run:
    pytest tests/test_hybrid_search.py -v
"""

from hybrid_search import reciprocal_rank_fusion


def test_single_list_preserves_order():
    """With only one ranked list, fusion should just preserve its order."""
    result = reciprocal_rank_fusion([[5, 2, 8]])
    ranked_ids = [idx for idx, _ in result]
    assert ranked_ids == [5, 2, 8]


def test_agreement_boosts_rank():
    """A chunk ranked by BOTH methods should outrank one ranked by only one,
    even if the single-method chunk was ranked #1 in its own list."""
    semantic_rank = [10, 20, 30]   # chunk 10 is FAISS's top pick
    bm25_rank = [20, 30, 10]       # chunk 20 is BM25's top pick

    result = reciprocal_rank_fusion([semantic_rank, bm25_rank])
    top_id = result[0][0]

    # chunk 20 appears at rank 1 in bm25 and rank 1 in semantic (index 1) ->
    # chunk 10 appears at rank 0 in semantic and rank 2 in bm25.
    # chunk 20's combined rank positions (1,0) beat chunk 10's (0,2).
    assert top_id == 20


def test_missing_from_one_list_contributes_zero():
    """A chunk found by only one method should still be included, just
    ranked lower than chunks found by both."""
    result = reciprocal_rank_fusion([[1, 2], [1, 3]])
    ranked_ids = [idx for idx, _ in result]
    assert set(ranked_ids) == {1, 2, 3}
    assert ranked_ids[0] == 1  # found by both, and top of both


def test_empty_lists_return_empty():
    result = reciprocal_rank_fusion([[], []])
    assert result == []


def test_k_dampens_rank_1_vs_rank_2_gap():
    """Larger k should shrink the score gap between adjacent ranks."""
    small_k = reciprocal_rank_fusion([[1, 2]], k=1)
    large_k = reciprocal_rank_fusion([[1, 2]], k=1000)

    small_gap = small_k[0][1] - small_k[1][1]
    large_gap = large_k[0][1] - large_k[1][1]
    assert large_gap < small_gap