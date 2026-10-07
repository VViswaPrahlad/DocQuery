import copy
from unittest.mock import Mock

import pytest

from fixture_utils import load_fixture
from scripts.evaluate_retrieval import DATASET, METHODS, validate_dataset, run_evaluation, render_report
import rag_engine


def test_evaluation_dataset_integrity():
    fixture = validate_dataset(load_fixture(DATASET))
    assert len(fixture["documents"]) == 6 and len(fixture["cases"]) == 36
    assert sum(bool(c.get("unanswerable")) for c in fixture["cases"]) == 7
    assert sum(c["kind"] == "follow_up" for c in fixture["cases"]) == 6
    assert sum(c["kind"] == "cross_document" for c in fixture["cases"]) == 4


def test_invalid_evidence_and_contradictory_answerability_fail():
    fixture = copy.deepcopy(load_fixture(DATASET))
    fixture["cases"][0]["relevant_pages"][0]["page"] = 99
    with pytest.raises(ValueError, match="Invalid evidence"):
        validate_dataset(fixture)
    fixture = copy.deepcopy(load_fixture(DATASET))
    fixture["cases"][0]["unanswerable"] = True
    with pytest.raises(ValueError, match="answerability"):
        validate_dataset(fixture)


def test_same_questions_denominators_and_no_live_generation(monkeypatch, fake_embedder):
    live = Mock(side_effect=AssertionError("Live generation must never be called"))
    monkeypatch.setattr(rag_engine, "get_client", live)
    report = run_evaluation(repeats=1)
    live.assert_not_called()
    assert len(report["rows"]) == 36 and report["chunks"] == 30
    for row in report["rows"]:
        assert set(row["methods"]) == set(METHODS)
        if row["case"].get("unanswerable"):
            assert all(row["methods"][m]["metrics"]["hit_at_k"] is None for m in METHODS)
        if row["case"].get("scope"):
            assert all(r["chunk"].doc_name in row["case"]["scope"] for m in METHODS for r in row["methods"][m]["results"])
    assert report["mock_calls"] == 7
    follow = next(r for r in report["rows"] if r["case"]["id"] == "C06")
    assert "What is the project funding amount?" in follow["query"]
    assert "Aurora" not in follow["query"]  # Preserve the actual heuristic's limitation.
    text = render_report(report)
    assert "not measured refusal accuracy" in text
    assert "excluding query embedding" in text
    assert "Citation reference validation" in text
