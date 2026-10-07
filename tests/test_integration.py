import pytest

from scripts.evaluate_retrieval import run_evaluation, METHODS


@pytest.mark.integration
def test_real_embeddings_synthetic_corpus():
    """Actual BGE -> FAISS/BM25/RRF -> page labels; no provider calls."""
    report = run_evaluation(repeats=1)
    assert len(report["rows"]) == 36
    assert sum(not row["case"].get("unanswerable") for row in report["rows"]) == 29
    for row in report["rows"]:
        assert set(row["methods"]) == set(METHODS)
        for method in METHODS:
            assert all(result["chunk"].page_number > 0 and result["chunk"].chunk_id
                       for result in row["methods"][method]["results"])
    assert all(0 <= report["summary"][method]["recall_at_k"] <= 1 for method in METHODS)
