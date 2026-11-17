"""
evaluation.py
-------------
Lightweight evaluation harness for the RAG pipeline, feeding the
Streamlit "Evaluation Dashboard" tab.

Two kinds of metrics:
  1. Retrieval metrics  -- computed directly, no LLM call needed:
       - avg / min / max rank scores (RRF or cosine, explicitly labelled)
       - retrieval latency
  2. Answer-quality metrics -- uses Grok itself as an LLM-judge to score
       groundedness (is the answer supported by the retrieved context?)
       and relevance (does it actually address the question?) on a 1-5 scale.

Log of every query run is kept in-memory (session_state in the app) and
can be exported to a DataFrame for charting.
"""

import time
import json
from typing import List, Dict
import pandas as pd
from answerability import decision_metrics  # Shared abstention metrics; no LLM judge dependency.

from rag_engine import retrieve, get_client, GROK_MODEL, _safe_chat_completion
from ingest import VectorStore


def evaluate_retrieval(query: str, store: VectorStore, top_k: int = 5) -> Dict:
    start = time.perf_counter()
    results = retrieve(query, store, top_k=top_k)
    latency = time.perf_counter() - start

    sources = [{"score": r["score"], "doc_name": r["chunk"].doc_name,
                "page": r["chunk"].page_number, "score_type": r.get("score_type", "rrf")} for r in results]
    return summarize_retrieval(query, sources, latency)


def summarize_retrieval(query, sources, latency=0.0):
    """Describe the exact evidence already used by an answer; never re-retrieve."""
    scores = [s["score"] for s in sources]
    return {
        "query": query,
        "latency_sec": round(latency, 3),
        "num_results": len(sources),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "min_score": round(min(scores), 4) if scores else 0.0,
        "max_score": round(max(scores), 4) if scores else 0.0,
        "docs_hit": sorted({s["doc_name"] for s in sources}),
        "pages_hit": sorted({(s["doc_name"], s["page"]) for s in sources}),
        "score_type": sources[0].get("score_type", "rrf") if sources else "none",
    }


def retrieval_ground_truth_metrics(sources, relevant_pages, k=5):
    """Page-level Hit@K, Recall@K and MRR@K, given explicit relevant pages.

    Repeated chunks on the same page count as one relevant page for recall.
    MRR uses the first relevant chunk rank. No score for absent ground truth.
    """
    relevant = {(r["doc_name"], int(r["page"])) for r in relevant_pages}
    if not relevant:
        return {"hit_at_k": None, "recall_at_k": None, "mrr_at_k": None, "document_recall_at_k": None}
    ranked = [(s["doc_name"], int(s["page"])) for s in sources[:k]]
    hits = set(ranked) & relevant
    first = next((i + 1 for i, pair in enumerate(ranked) if pair in relevant), None)
    docs = {doc for doc, page in relevant}
    return {"hit_at_k": float(bool(hits)), "recall_at_k": len(hits) / len(relevant),
            "mrr_at_k": 1 / first if first else 0.0,
            "document_recall_at_k": len({doc for doc, page in ranked} & docs) / len(docs)}


JUDGE_PROMPT = """You are grading a RAG system's answer.
Given the QUESTION, the retrieved CONTEXT, and the ANSWER, score two things from 1 (bad) to 5 (excellent):
- groundedness: is every claim in the answer actually supported by the context (not hallucinated)?
- relevance: does the answer actually address the question asked?

Return ONLY valid JSON, no markdown, in this exact shape:
{"groundedness": <int 1-5>, "relevance": <int 1-5>, "justification": "<one sentence>"}
"""


def evaluate_answer_quality(query: str, context_sources: List[Dict], answer: str) -> Dict:
    """LLM-as-judge scoring of a generated answer against its retrieved context."""
    context_text = "\n\n".join(
        f"[{s['doc_name']} p{s['page']}] {s.get('text', '')}" for s in context_sources
    )
    try:
        response = _safe_chat_completion(get_client(), max_retries=0,
            model=GROK_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"QUESTION: {query}\n\nCONTEXT:\n{context_text}\n\nANSWER:\n{answer}"},
            ], temperature=0.0, max_tokens=500,
        )
        raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or any(type(parsed.get(k)) is not int or not 1 <= parsed[k] <= 5 for k in ("groundedness", "relevance")):
            raise ValueError("Invalid judge score schema")
        parsed = {k: parsed.get(k) for k in ("groundedness", "relevance", "justification")}
        parsed["judge_status"] = "scored"
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        parsed = {"groundedness": None, "relevance": None,
                  "justification": "Optional judge unavailable or returned invalid scores.",
                  "judge_status": type(exc).__name__}
    parsed["query"] = query
    return parsed


def logs_to_dataframe(logs: List[Dict]) -> pd.DataFrame:
    """Convert the in-session evaluation log list into a DataFrame for charting."""
    if not logs:
        return pd.DataFrame(
            columns=["query", "latency_sec", "avg_score", "groundedness", "relevance"]
        )
    df = pd.DataFrame(logs)
    # Arrow cannot infer a homogeneous type for nested (filename, page) tuples.
    # Preserve structured session logs, but serialize collections for display/CSV.
    for column in df.columns:
        df[column] = df[column].map(lambda value: json.dumps(value, ensure_ascii=False)
                                   if isinstance(value, (list, tuple, dict)) else value)
    return df
