# DocQuery documentation and interview guide

## Current project at a glance

[Main README](../README.md) · [Live Streamlit demo](https://docquery-rag.streamlit.app) · [Retrieval evaluation](evaluation.md)

**Problem:** PDF question-answering systems often make retrieval and source evidence difficult to inspect. DocQuery supports questions over multiple PDFs, page-aware citations, scoped questions, follow-ups, OCR fallback and evidence inspection.

**Pipeline:** PDF → PyMuPDF (or Tesseract OCR) → page-bound chunks → BGE embeddings → FAISS dense / BM25 lexical retrieval → optional RRF fusion → Groq-hosted LLM → answer with source-page references.

**Evidence:** A small controlled fixture of 36 questions across six fictional documents was used to compare retrieval, not to establish production or general-domain answer correctness. Dense retrieval had Hit@3 1.000 / MRR@3 0.931; hybrid RRF had 0.931 / 0.810; BM25 had 0.793 / 0.638. The controlled benchmark is explicitly nonrepresentative of unseen PDFs. [Full measurement and failure cases](evaluation.md).

## Reading order

1. [README](../README.md) — problem, demo, features and setup.
2. [Offline evaluation](evaluation.md) — metrics, comparison, follow-ups and why RRF sometimes loses.
3. [Core retrieval](../ingest.py) and [BM25/RRF](../hybrid_search.py) — extraction, chunks, index and ranking.
4. [Generation and provider errors](../rag_engine.py), [citation checking](../citations.py) — answer construction and limits.
5. [Tests](../tests) and [CI workflow](../.github/workflows/tests.yml) — deterministic fixtures, mocked provider calls and checks.
6. [Privacy audit](../scripts/privacy_audit.py) — local publication check (heuristic, not a guarantee).

## Two-minute explanation

DocQuery is a multi-document RAG app built with Python and Streamlit. It first extracts text from each PDF and falls back to OCR when appropriate. It keeps page metadata with chunks, embeds them using BGE, and retrieves relevant chunks through FAISS semantic search or BM25 plus optional RRF hybrid fusion. Retrieved context goes to a Groq-hosted model; the UI shows the supporting document and page references, allowing inspection. I evaluated dense, lexical and hybrid retrieval on a frozen fictional fixture. Dense won on this small set, so I investigated lexical distractors rather than claiming hybrid always improves accuracy. Citation validation checks reference membership among retrieved pages, not whether the model's claim follows from those pages.

## Likely interview questions and answers

**Why RAG instead of fine-tuning?** RAG can incorporate newly uploaded documents without retraining, and retrieve inspectable passages for an answer. It does not eliminate hallucinations.

**Why page-boundary chunks?** Source page references remain traceable even after splitting text, at the cost of sometimes losing cross-page context.

**Why use BGE and FAISS?** BGE maps text to semantic vectors and FAISS efficiently ranks matching vectors for a local document collection. Quality depends on the embedding model, preprocessing, query and corpus.

**Why use BM25?** Exact terms, rare names and codes can be strong lexical signals when semantics blur them. BM25 does not understand paraphrases the way embeddings attempt to.

**What is RRF and why did it lose?** Reciprocal Rank Fusion merges *ranks*, avoiding directly mixing incompatible BM25 and cosine score scales. On the fixture, lexical agreement sometimes elevated irrelevant distractors and displaced relevant dense candidates.

**What do Hit@3, Recall@3 and MRR@3 mean?** Hit@3 asks whether at least one labelled relevant page appears in the top three. Recall@3 measures the fraction of labelled relevant pages found. MRR@3 rewards the first relevant result being ranked earlier. These are *retrieval* metrics, not generation accuracy.

**How reliable are citations?** The application checks that document/page references were among retrieved evidence. It does not independently verify claim-level entailment or guarantee factual accuracy.

**What about PDFs with scans?** PyMuPDF handles existing text; Tesseract OCR can process pages with little extractable text. Complex layouts and OCR errors remain challenges.

**What about privacy and provider failures?** The app uses separate in-memory session state and ignores local secrets/uploads; selected document excerpts are sent to the configured external model service. Provider errors are categorized and sanitized. Never promise private/offline generation when using a hosted LLM.

**What would you improve?** Evaluate real-world and adversarial PDFs, include answer faithfulness judged against evidence, add stronger citation grounding checks, study fusion parameters on held-out data, and improve large-document cost/latency measurement.

## Suggested short demo

1. Show an upload or the **Try Demo** fictional reports.
2. Ask a question and inspect the source pages; explain page-aware chunks.
3. Ask a cross-document or follow-up question; explain query context.
4. Open [retrieval evaluation](evaluation.md), acknowledge the dataset's small scale and show why hybrid can lose.

## Reproduction and scope

Use Python 3.13 and [the README setup](../README.md#setup). Tests use deterministic fake embeddings and blocked live provider calls where possible. Running the real embedding evaluation needs the BGE model available locally. Generation needs the configured provider key/network; initial embedding setup may download weights. The demo's availability and model provider status need a fresh manual check before an interview.
