"""Small offline portfolio evaluation. Writes one current Markdown artifact.

python -m scripts.evaluate_retrieval
Requires existing BGE weights. No provider access, downloads, or algorithm tuning.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from config import ROOT
from evaluation import retrieval_ground_truth_metrics
from fixture_utils import load_fixture, write_fixture_pdfs
from hybrid_search import BM25Index, reciprocal_rank_fusion
from ingest import build_index_from_pdfs, EmbeddingModel, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
import rag_engine as rag

DATASET = ROOT / "tests" / "fixtures" / "retrieval_evaluation.json"
METHODS = ("Dense / FAISS", "BM25", "Hybrid RRF")


def validate_dataset(fixture):
    docs = {d["name"]: d["pages"] for d in fixture["documents"]}
    if len(docs) != len(fixture["documents"]) or len(docs) < 5:
        raise ValueError("Need at least five distinct documents")
    ids = [case["id"] for case in fixture["cases"]]
    if len(set(ids)) != len(ids) or not 30 <= len(ids) <= 40:
        raise ValueError("Need 30–40 distinct questions")
    for case in fixture["cases"]:
        relevant = case["relevant_pages"]
        if bool(case.get("unanswerable")) == bool(relevant):
            raise ValueError(f"Inconsistent answerability: {case['id']}")
        for reference in relevant:
            pages = docs[reference["doc_name"]]
            if not 1 <= reference["page"] <= len(pages):
                raise ValueError("Invalid evidence page")
        if case.get("scope") and any(r["doc_name"] not in case["scope"] for r in relevant):
            raise ValueError("Evidence lies outside the selected scope")
        if case.get("support") and not any(case["support"] in docs[r["doc_name"]][r["page"] - 1] for r in relevant):
            raise ValueError(f"Support text missing: {case['id']}")
        if case["kind"] == "follow_up" and (not case.get("history") or case.get("intended_document") not in docs):
            raise ValueError("Follow-ups require history and intended-document labels")
    return fixture


def search(method, store, bm25, query, vec, scope=None):
    if method == "Dense / FAISS":
        return store.search(vec, top_k=3, doc_filter=scope)
    if method == "BM25":
        allowed = [i for i, c in enumerate(store.chunks) if c.doc_name in scope] if scope else None
        return [{"chunk": store.chunks[i], "idx": i, "score": score} for i, score in bm25.search(query, top_k=3, allowed_indices=allowed)]
    if method == "Hybrid RRF":
        return store.hybrid_search(vec, query, top_k=3, doc_filter=scope)
    raise ValueError("Unknown retrieval method")


def sources(results):
    return [{"doc_name": r["chunk"].doc_name, "page": r["chunk"].page_number} for r in results]


def metrics(results, case):
    return retrieval_ground_truth_metrics(sources(results), case["relevant_pages"], k=3)


def mean_metric(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def reference_stats(results, intended):
    docs = [r["chunk"].doc_name for r in results]
    return {"document_hit": int(intended in docs), "top1": int(bool(docs) and docs[0] == intended),
            "purity": sum(d == intended for d in docs) / len(docs) if docs else 0.0}


def mock_refusal_contract(store, cases):
    """Test only the refusal instruction/response plumbing; not model judgment."""
    calls = 0

    def complete(*args, **kwargs):
        nonlocal calls
        assert "does not contain enough information" in kwargs["messages"][0]["content"]
        calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="The provided evidence does not contain enough information to answer."))])

    with patch.object(rag, "get_client", return_value=object()), patch.object(rag, "_safe_chat_completion", side_effect=complete):
        for case in cases:
            result = rag.answer_question(case["question"], store, chat_history=case.get("history"), doc_filter=case.get("scope"), top_k=3)
            assert "does not contain enough information" in result["answer"]
    return calls


def fusion_trace(store, bm25, query, vec, case):
    """Inspect unchanged 20/20 pools, not an alternative retrieval method."""
    semantic = store.search(vec, top_k=20, doc_filter=case.get("scope"))
    allowed = [i for i, c in enumerate(store.chunks) if c.doc_name in case["scope"]] if case.get("scope") else None
    lexical = bm25.search(query, top_k=20, allowed_indices=allowed)
    dense_rank = {r["idx"]: i + 1 for i, r in enumerate(semantic)}
    lexical_rank = {idx: i + 1 for i, (idx, _) in enumerate(lexical)}
    fused_rank = {idx: i + 1 for i, (idx, _) in enumerate(reciprocal_rank_fusion([list(dense_rank), list(lexical_rank)], k=60))}
    relevant = {(r["doc_name"], r["page"]) for r in case["relevant_pages"]}
    return [{"page": f"{c.doc_name.removesuffix('.pdf')}:{c.page_number}", "dense": dense_rank.get(i),
             "bm25": lexical_rank.get(i), "rrf": fused_rank.get(i)}
            for i, c in enumerate(store.chunks) if (c.doc_name, c.page_number) in relevant]


def run_evaluation(dataset=DATASET, repeats=20):
    fixture = validate_dataset(load_fixture(dataset))
    if repeats < 1:
        raise ValueError("Latency repeats must be positive")
    runtime = ROOT / ".runtime" / "retrieval-evaluation"
    runtime.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(runtime)
    with patch.object(rag, "get_client", side_effect=AssertionError("Live provider access is prohibited")):
        with tempfile.TemporaryDirectory(dir=runtime) as directory:
            store = build_index_from_pdfs(write_fixture_pdfs(directory, fixture), persist=False)
        bm25 = BM25Index.from_store(store)
        store._bm25 = bm25  # Warm the existing lazy index; no scoring changes.
        embedder = EmbeddingModel.get()
        encode_ms, vectors = [], {}

        def vector(query):
            if query not in vectors:
                started = time.perf_counter()
                vectors[query] = embedder.embed([query], is_query=True)
                encode_ms.append((time.perf_counter() - started) * 1000)
            return vectors[query]

        rows, followups, scope_checks, coverage = [], [], [], []
        for case in fixture["cases"]:
            query = rag.conversational_query(case["question"], case.get("history"))
            vec = vector(query)
            row = {"case": case, "query": query, "methods": {}}
            # Rotate timing order to reduce systematic first-method bias.
            order = METHODS[len(rows) % 3:] + METHODS[:len(rows) % 3]
            for method in order:
                result = search(method, store, bm25, query, vec, case.get("scope"))
                timings = []
                for _ in range(repeats):
                    started = time.perf_counter()
                    search(method, store, bm25, query, vec, case.get("scope"))
                    timings.append((time.perf_counter() - started) * 1000)
                row["methods"][method] = {"results": result, "metrics": metrics(result, case), "latency_ms": statistics.median(timings)}
            if not case.get("unanswerable") and row["methods"]["Hybrid RRF"]["metrics"]["mrr_at_k"] < row["methods"]["Dense / FAISS"]["metrics"]["mrr_at_k"]:
                row["fusion_trace"] = fusion_trace(store, bm25, query, vec, case)
            rows.append(row)
            if case["kind"] == "follow_up":
                raw_vec = vector(case["question"])
                for method in METHODS:
                    raw = search(method, store, bm25, case["question"], raw_vec)
                    current = row["methods"][method]["results"]
                    followups.append({"id": case["id"], "method": method, "expanded": query != case["question"],
                                      "raw": reference_stats(raw, case["intended_document"]),
                                      "current": reference_stats(current, case["intended_document"]),
                                      "raw_hit": metrics(raw, case)["hit_at_k"], "current_hit": metrics(current, case)["hit_at_k"]})
            if case.get("scope"):
                for method in METHODS:
                    unscoped = search(method, store, bm25, query, vec)
                    scope_checks.append({"method": method, "raw": metrics(unscoped, case), "scoped": row["methods"][method]["metrics"]})
            if case["kind"] == "cross_document":
                # Separate application policy check; not part of the three-way ablation.
                result = rag.retrieve_multi_document(case["question"], store, top_k=3)
                coverage.append({"id": case["id"], "global": row["methods"]["Hybrid RRF"]["metrics"], "app": metrics(result, case), "results": result})
        unanswerable = [r for r in rows if r["case"].get("unanswerable")]
        mock_calls = mock_refusal_contract(store, [r["case"] for r in unanswerable])
    summary = {}
    for method in METHODS:
        measured = [r["methods"][method]["metrics"] for r in rows if not r["case"].get("unanswerable")]
        summary[method] = {key: mean_metric(measured, key) for key in ("hit_at_k", "recall_at_k", "mrr_at_k", "document_recall_at_k")}
        summary[method]["latency_ms"] = statistics.median(r["methods"][method]["latency_ms"] for r in rows)
    return {"fixture": fixture, "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
            "chunks": len(store.chunks), "rows": rows, "followups": followups, "scopes": scope_checks,
            "coverage": coverage, "summary": summary, "mock_calls": mock_calls,
            "embedding_latency_ms": statistics.median(encode_ms), "repeats": repeats}


def pages(results):
    return ", ".join(f"{s['doc_name'].removesuffix('.pdf')}:{s['page']}" for s in sources(results)) or "empty"


def render_report(report):
    fixture, rows = report["fixture"], report["rows"]
    answerable = [r for r in rows if not r["case"].get("unanswerable")]
    unanswerable = [r for r in rows if r["case"].get("unanswerable")]
    counts = Counter(c["kind"] for c in fixture["cases"])
    lines = ["# DocQuery: current offline retrieval evaluation", "",
             f"Recorded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Fictional portfolio fixture; not a research benchmark.", "",
             f"**Dataset:** {len(fixture['documents'])} documents, {sum(len(d['pages']) for d in fixture['documents'])} pages, {report['chunks']} chunks; {len(rows)} questions, {len(answerable)} answerable, {len(unanswerable)} unanswerable, {counts['follow_up']} follow-ups, {counts['cross_document']} cross-document questions.",
             "", "Documents: " + ", ".join(d["name"] for d in fixture["documents"]) + ".",
             "", "[Questions, text, and page labels](../tests/fixtures/retrieval_evaluation.json) were fixed before scores were inspected. Similar fault identifiers and overlapping maintenance/funding topics provide distractors. No dataset, algorithm, model, or retrieval parameters were tuned after measuring results.",
             "", f"Dataset SHA-256: `{report['dataset_sha256']}`.", "", "## Same-question ablation", "",
             f"Model: `{EMBEDDING_MODEL_NAME}`; normalized embeddings and existing query prefix. Chunk size/overlap: {CHUNK_SIZE}/{CHUNK_OVERLAP} characters. K=3; existing hybrid candidate pools 20/20 and RRF k=60. All methods use identical queries, existing follow-up expansion, and user-selected scope. Cross-document questions use ordinary global top-3 in this table; the application coverage helper is checked separately below.",
             "", "Hit@3: any labelled page found. Recall@3: fraction of all labelled pages found, counting duplicate page chunks once. MRR@3: reciprocal rank of first relevant chunk, zero for a miss. Macro-average over answerable questions only. X04 requires four pages, so complete evidence recall is impossible at K=3 for every method.",
             "", "| Method | Hit@3 | Recall@3 | MRR@3 | Document recall@3 | Median retrieval latency (ms) |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        s = report["summary"][method]
        lines.append(f"| {method} | {s['hit_at_k']:.3f} | {s['recall_at_k']:.3f} | {s['mrr_at_k']:.3f} | {s['document_recall_at_k']:.3f} | {s['latency_ms']:.4f} |")
    lines += ["", f"Latency is warm in-memory search, including scope handling/tokenization/RRF but **excluding query embedding, PDF ingestion, model load, and index build**. Each query/method has {report['repeats']} timed repetitions; table reports median of per-query medians across all {len(rows)} questions. Query encoding median: **{report['embedding_latency_ms']:.2f} ms** (not included above). FAISS/BM25 are warmed before timing. Method order rotates. Host: {platform.system()} / {platform.machine()}, Python {platform.python_version()}. Local tiny-corpus timings are not production throughput claims.", "", "### Question groups", "", "| Group (answerable count) | Dense Hit / Recall / MRR | BM25 Hit / Recall / MRR | Hybrid Hit / Recall / MRR |", "|---|---|---|---|"]
    for kind in ("exact", "semantic", "factual", "scoped", "follow_up", "cross_document"):
        group = [r for r in answerable if r["case"]["kind"] == kind]
        cells = [" / ".join(f"{mean_metric([r['methods'][m]['metrics'] for r in group], k):.3f}" for k in ("hit_at_k", "recall_at_k", "mrr_at_k")) for m in METHODS]
        lines.append(f"| {kind} ({len(group)}) | " + " | ".join(cells) + " |")
    lines += ["", "### Where hybrid helps or loses", ""]
    for comparator in METHODS[:2]:
        wins = [r["case"]["id"] for r in answerable if r["methods"]["Hybrid RRF"]["metrics"]["mrr_at_k"] > r["methods"][comparator]["metrics"]["mrr_at_k"]]
        losses = [r["case"]["id"] for r in answerable if r["methods"]["Hybrid RRF"]["metrics"]["mrr_at_k"] < r["methods"][comparator]["metrics"]["mrr_at_k"]]
        lines.append(f"- Versus {comparator}: higher MRR on {len(wins)} ({', '.join(wins) or 'none'}); lower on {len(losses)} ({', '.join(losses) or 'none'}).")
    lines += ["", "Relevant-page candidate ranks for hybrid regressions (`None` means absent from that top-20 pool):", "", "| Case | Relevant page | Dense pool rank | BM25 pool rank | Fused rank |", "|---|---|---:|---:|---:|"]
    for row in answerable:
        for trace in row.get("fusion_trace", []):
            lines.append(f"| {row['case']['id']} | {trace['page']} | {trace['dense']} | {trace['bm25']} | {trace['rrf']} |")
    lines += ["", "RRF rewards agreement between ranked lists, including lexical overlap with distractors. It does not assess whether a page answers the question. In S05, the gauge-calibration page is strong in dense search but weaker lexically; unrelated pages supported by both lists displace it. E01's incident index merely mentions fault strings without defining them. The per-case ranks distinguish this fusion effect from extraction failure: the labelled pages survive PDF ingestion and dense retrieval. No algorithm or fixture was changed after observing scores.", "", "## Follow-up reference retention", "",
              "Six follow-ups compare the latest question alone with the existing user-question expansion. Intended-document labels are used only for scoring, never as automatic retrieval filters. The ambiguous funding → timeline case has no answerable timeline evidence, so it is excluded from evidence-hit averages but included in document-retention checks.", "", "| Method | Raw → current evidence Hit@3 (5 answerable) | Raw → current intended-doc Hit@3 (6 cases) | Raw → current top-1 intended doc | Current intended-doc share in results |", "|---|---|---|---|---|"]
    for method in METHODS:
        f = [r for r in report["followups"] if r["method"] == method]
        pair = lambda key: f"{statistics.mean(r['raw'][key] for r in f):.3f} → {statistics.mean(r['current'][key] for r in f):.3f}"
        lines.append(f"| {method} | {mean_metric(f, 'raw_hit'):.3f} → {mean_metric(f, 'current_hit'):.3f} | {pair('document_hit')} | {pair('top1')} | {statistics.mean(r['current']['purity'] for r in f):.3f} |")
    lines += ["", "| Follow-up | Expansion applied | Hybrid evidence Hit@3 | Hybrid intended doc found / top-1 | Hybrid retrieved pages |", "|---|---|---|---|---|"]
    for r in rows:
        if r["case"]["kind"] == "follow_up":
            result = r["methods"]["Hybrid RRF"]["results"]
            stats = reference_stats(result, r["case"]["intended_document"])
            lines.append(f"| {r['case']['id']}: {r['case']['question']} | {r['query'] != r['case']['question']} | {r['methods']['Hybrid RRF']['metrics']['hit_at_k']} | {stats['document_hit']} / {stats['top1']} | {pages(result)} |")
    lines += ["", "The current heuristic uses previous user questions, not the previous assistant's referenced document. It may mix projects, miss implicit follow-ups without recognized pronouns, or retrieve the intended document without the needed page. Finding the intended document is weaker than resolving the reference correctly.", "", "## Document scope and cross-document coverage", "", "| Method | Unscoped → scoped Hit@3 (4 scope questions) | Unscoped → scoped MRR@3 |", "|---|---|---|"]
    for method in METHODS:
        s = [r for r in report["scopes"] if r["method"] == method]
        lines.append(f"| {method} | {statistics.mean(r['raw']['hit_at_k'] for r in s):.3f} → {statistics.mean(r['scoped']['hit_at_k'] for r in s):.3f} | {statistics.mean(r['raw']['mrr_at_k'] for r in s):.3f} → {statistics.mean(r['scoped']['mrr_at_k'] for r in s):.3f} |")
    lines += ["", f"Cross-document page Recall@3: global hybrid **{statistics.mean(r['global']['recall_at_k'] for r in report['coverage']):.3f}**, existing application coverage helper **{statistics.mean(r['app']['recall_at_k'] for r in report['coverage']):.3f}** over four cases. Helper mode: {'hybrid' if rag.USE_HYBRID_SEARCH else 'dense'}, existing configured relevance heuristic; no changes. More document coverage is not proof that the right evidence was selected.", "", "## Unanswerable and refusal behavior", "", "| Method | Unanswerable questions returning nonempty evidence (7 cases) |", "|---|---:|"]
    for method in METHODS:
        lines.append(f"| {method} | {sum(bool(r['methods'][method]['results']) for r in unanswerable)}/{len(unanswerable)} |")
    lines += ["", f"**Zero live provider calls.** A mocked generation contract passed for {report['mock_calls']}/{len(unanswerable)} cases: the real answer flow supplied its refusal instruction and propagated the mock refusal. The mock deliberately supplies a refusal; this is a plumbing check, **not measured refusal accuracy**. Nonempty retrieval is not proof of answerability. The system has no tested semantic abstention gate, and similar grants/timelines can supply plausible but wrong evidence. Real model refusal correctness remains unmeasured in this evaluation.", "", "## Inspectable case results", "", "Page shorthand is `document:page`. Relevance labels are in the dataset. Empty BM25 results are retained as misses; no method gets a different denominator.", "", "| ID / group | Labelled pages | Dense top-3 | BM25 top-3 | Hybrid top-3 |", "|---|---|---|---|---|"]
    for r in rows:
        label = ", ".join(f"{p['doc_name'].removesuffix('.pdf')}:{p['page']}" for p in r["case"]["relevant_pages"]) or "unanswerable"
        lines.append(f"| {r['case']['id']} / {r['case']['kind']} | {label} | " + " | ".join(pages(r["methods"][m]["results"]) for m in METHODS) + " |")
    lines += ["", "## Interpretation and limits", "", "This fixture measures local page retrieval, exact-term/paraphrase trade-offs, selected-document filtering, and current follow-up/coverage policies. It does not measure end-to-end answer truth, semantic entailment, OCR quality, generalization to arbitrary PDFs, deployment throughput, or actual LLM refusal reliability. Citation reference validation checks only that cited document/page references occur in retrieved evidence; it cannot prove factual correctness.", "", "Reproduce: `python -m scripts.evaluate_retrieval` using the populated `venv` and existing offline model cache. Only this current Markdown artifact is written; generated PDFs are temporary. Older smoke reports are historical integration checks, not the main evaluation."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "evaluation.md")
    args = parser.parse_args()
    report = run_evaluation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("Current artifact:", args.output)


if __name__ == "__main__":
    main()
