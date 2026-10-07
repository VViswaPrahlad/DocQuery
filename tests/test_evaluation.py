from evaluation import retrieval_ground_truth_metrics, summarize_retrieval, logs_to_dataframe
from fixture_utils import write_fixture_pdfs, load_fixture
from ingest import build_index_from_pdfs
from rag_engine import retrieve


def test_page_metrics_handle_duplicates_and_rank():
    sources = [{"doc_name": "other.pdf", "page": 1}, {"doc_name": "a.pdf", "page": 2}, {"doc_name": "a.pdf", "page": 2}]
    relevant = [{"doc_name": "a.pdf", "page": 2}, {"doc_name": "b.pdf", "page": 4}]
    metrics = retrieval_ground_truth_metrics(sources, relevant, k=3)
    assert metrics == {"hit_at_k": 1.0, "recall_at_k": 0.5, "mrr_at_k": 0.5, "document_recall_at_k": 0.5}
    assert retrieval_ground_truth_metrics(sources, [], k=3)["hit_at_k"] is None


def test_dashboard_describes_supplied_evidence():
    sources = [{"doc_name": "selected.pdf", "page": 8, "score": 0.02, "score_type": "rrf"}]
    result = summarize_retrieval("follow-up", sources, 0.25)
    assert result["docs_hit"] == ["selected.pdf"]
    assert result["pages_hit"] == [("selected.pdf", 8)]
    assert result["score_type"] == "rrf" and result["latency_sec"] == 0.25


def test_dashboard_logs_are_arrow_compatible_without_mutating_evidence():
    import pyarrow as pa
    sources = [{"doc_name": "a.pdf", "page": 1, "score": 0.025, "score_type": "rrf"}]
    log = summarize_retrieval("question", sources, 0.1)
    frame = logs_to_dataframe([log])
    assert pa.Table.from_pandas(frame).num_rows == 1
    assert log["pages_hit"] == [("a.pdf", 1)]
    assert frame["pages_hit"][0] == '[["a.pdf", 1]]'


def test_synthetic_pdf_to_index_to_retrieval(tmp_path, fake_embedder):
    fixture = load_fixture()
    paths = write_fixture_pdfs(tmp_path, fixture)
    store = build_index_from_pdfs(paths, persist=False)
    assert store.index.ntotal == len(store.chunks) == 4
    assert {(c.doc_name, c.page_number) for c in store.chunks} == {
        ("battery_brief.pdf", 1), ("battery_brief.pdf", 2), ("fleet_report.pdf", 1), ("fleet_report.pdf", 2)}
    result = retrieve("MF4 telemetry", store)
    assert "MF4" in result[0]["chunk"].text
