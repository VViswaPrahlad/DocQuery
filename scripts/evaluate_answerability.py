"""Paired generation evaluation; baseline files are never overwritten.

Prepare fixed evidence once (requires cached BGE), then run live generation:
python -m scripts.evaluate_answerability prepare --dataset tests/fixtures/retrieval_evaluation.json --output .runtime/answerability/baseline-inputs.json
python -m scripts.evaluate_answerability run --inputs .runtime/answerability/baseline-inputs.json --output .runtime/answerability/baseline-results.json
Raw outputs require manual semantic review. No automatic truth claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone
import platform
import subprocess

from answerability import PROMPT, decision_metrics

BASELINE_HASH = "54ce50f7d70ae5bb61d990e6ae2da9d248561ebea15ce7369faf5635d505ae07"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects baseline artifacts and previous runs.
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def prepare(dataset, output, split, mode, method):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import tempfile
    from fixture_utils import write_fixture_pdfs
    from ingest import build_index_from_pdfs, EmbeddingModel
    from hybrid_search import BM25Index
    from scripts.evaluate_retrieval import search
    import rag_engine as rag

    fixture = json.loads(Path(dataset).read_text())
    selected = [c for c in fixture["cases"] if split == "all" or c.get("split") == split]
    if not selected:
        raise ValueError("No cases for selected split")
    if Path(dataset).name == "retrieval_evaluation.json" and digest(dataset) != BASELINE_HASH:
        raise ValueError("Original benchmark has changed")
    names = {d["name"] for d in fixture["documents"] if split == "all" or d.get("split") == split}
    documents = [d for d in fixture["documents"] if d["name"] in names]
    with tempfile.TemporaryDirectory() as directory:
        store = build_index_from_pdfs(write_fixture_pdfs(directory, {"documents": documents}), persist=False)
    bm25 = BM25Index.from_store(store)
    rows = []
    for case in selected:
        query = rag.conversational_query(case["question"], case.get("history"))
        if mode == "application" and case["kind"] == "cross_document":
            results = rag.retrieve_multi_document(query, store, top_k=8)
        else:
            vec = EmbeddingModel.get().embed([query], is_query=True)
            results = search(method, store, bm25, query, vec, case.get("scope"))
        rows.append({"case": case, "retrieval_query": query, "sources": [
            {"chunk_id": r["chunk"].chunk_id, "doc_name": r["chunk"].doc_name,
             "page": r["chunk"].page_number, "text": r["chunk"].text,
             "score": r["score"], "score_type": r.get("score_type", "unknown")} for r in results]})
    write_new(output, {"dataset_sha256": digest(dataset), "split": split, "mode": mode,
                       "method": method, "rows": rows})


def run(inputs, output):
    from types import SimpleNamespace
    import rag_engine as rag
    data = json.loads(Path(inputs).read_text())
    # Fail before creating misleading all-error results when credentials are absent.
    if Path(output).exists() or list(Path(output).parent.glob(Path(output).name + ".case-*.json")):
        raise FileExistsError("Choose a fresh output path; existing run/checkpoints are preserved")
    client = rag.get_client()
    rows = []
    for index, row in enumerate(data["rows"]):
        case, sources = row["case"], row["sources"]
        results = [{"chunk": SimpleNamespace(chunk_id=s["chunk_id"], doc_name=s["doc_name"],
                    page_number=s["page"], text=s["text"], source="native"),
                    "score": s["score"], "score_type": s["score_type"]} for s in sources]
        pair = {}
        # Alternate order to reduce systematic first-arm latency bias.
        arms = ["legacy", "structured"] if index % 2 == 0 else ["structured", "legacy"]
        for arm in arms:
            started = time.perf_counter()
            if arm == "structured":
                pair[arm] = rag.answer_from_results(case["question"], results, case.get("history"))
            elif not results:
                pair[arm] = {"answer": "No indexed evidence", "decision": "abstain"}
            else:
                messages = [{"role": "system", "content": rag.SYSTEM_PROMPT}]
                messages.extend(rag._history_messages(case.get("history")))
                messages.append({"role": "user", "content": f"Context:\n\n{rag._format_context(results)}\n\nQuestion: {case['question']}\n\nAnswer with citations:"})
                try:
                    response = rag._safe_chat_completion(client, model=rag.GROK_MODEL,
                        messages=messages, temperature=0.0, max_tokens=2000)
                    pair[arm] = {"answer": response.choices[0].message.content,
                                 "decision": "needs_manual_label",
                                 "usage": response.usage.model_dump() if getattr(response, "usage", None) else None}
                except RuntimeError:
                    pair[arm] = {"answer": "Provider failure", "decision": "error"}
            pair[arm]["latency_seconds"] = time.perf_counter() - started
        rows.append({**row, "outputs": pair, "review": {
            "legacy_decision": None, "structured_appropriate": None,
            "legacy_appropriate": None, "structured_fully_supported": None,
            "legacy_fully_supported": None, "evidence_sufficient": None, "notes": ""}})
        # Durable per-case checkpoint; never print document text to logs.
        write_new(str(output) + f".case-{index:03d}.json", rows[-1])
    write_new(output, {"recorded_utc": datetime.now(timezone.utc).isoformat(),
                       "python": platform.python_version(),
                       "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                       "model": rag.GROK_MODEL, "temperature": 0, "max_tokens": 2000,
                       "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                       "inputs_sha256": digest(inputs), "dataset_sha256": data["dataset_sha256"],
                       "split": data["split"], "mode": data["mode"], "method": data["method"],
                       "semantic_review": "pending", "rows": rows})


def summarize(path):
    data = json.loads(Path(path).read_text())
    summary = {}
    for arm in ("legacy", "structured"):
        rows = []
        for row in data["rows"]:
            decision = row["outputs"][arm]["decision"]
            if arm == "legacy" and decision == "needs_manual_label":
                decision = row["review"]["legacy_decision"]
            if decision not in {"answer", "abstain", "error"}:
                raise ValueError("Complete manual legacy decision labels before scoring")
            rows.append({"answerable": not row["case"].get("unanswerable", False), "decision": decision})
        summary[arm] = decision_metrics(rows)
        answered = [r for r in data["rows"] if (r["review"]["legacy_decision"] if arm == "legacy" and r["outputs"][arm]["decision"] == "needs_manual_label" else r["outputs"][arm]["decision"]) == "answer"]
        reviewed = [r["review"][arm + "_fully_supported"] for r in answered]
        appropriate = [r["review"][arm + "_appropriate"] for r in data["rows"]]
        summary[arm]["appropriate_decision_rate"] = sum(appropriate) / len(appropriate) if appropriate and all(type(v) is bool for v in appropriate) else None
        summary[arm]["supported_answer_accuracy"] = (sum(reviewed) / len(reviewed)) if reviewed and all(type(v) is bool for v in reviewed) else None
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["all", "dev", "heldout"], default="all")
    p.add_argument("--mode", choices=["top3", "application"], default="top3")
    p.add_argument("--method", choices=["Dense / FAISS", "BM25", "Hybrid RRF"], default="Hybrid RRF")
    p = sub.add_parser("run")
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("summarize")
    p.add_argument("results")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "summarize":
        summarize(args["results"])
    else:
        globals()[command](**args)


if __name__ == "__main__":
    main()
