# DocQuery: current offline retrieval evaluation

Recorded 2026-10-07 12:22 UTC. Fictional portfolio fixture; not a research benchmark.

**Dataset:** 6 documents, 30 pages, 30 chunks; 36 questions, 29 answerable, 7 unanswerable, 6 follow-ups, 4 cross-document questions.

Documents: battery_brief.pdf, fleet_report.pdf, river_operations.pdf, archive_handbook.pdf, building_energy.pdf, incident_playbook.pdf.

[Questions, text, and page labels](../tests/fixtures/retrieval_evaluation.json) were fixed before scores were inspected. Similar fault identifiers and overlapping maintenance/funding topics provide distractors. No dataset, algorithm, model, or retrieval parameters were tuned after measuring results.

Dataset SHA-256: `54ce50f7d70ae5bb61d990e6ae2da9d248561ebea15ce7369faf5635d505ae07`.

## Same-question ablation

Model: `BAAI/bge-base-en-v1.5`; normalized embeddings and existing query prefix. Chunk size/overlap: 800/120 characters. K=3; existing hybrid candidate pools 20/20 and RRF k=60. All methods use identical queries, existing follow-up expansion, and user-selected scope. Cross-document questions use ordinary global top-3 in this table; the application coverage helper is checked separately below.

Hit@3: any labelled page found. Recall@3: fraction of all labelled pages found, counting duplicate page chunks once. MRR@3: reciprocal rank of first relevant chunk, zero for a miss. Macro-average over answerable questions only. X04 requires four pages, so complete evidence recall is impossible at K=3 for every method.

| Method | Hit@3 | Recall@3 | MRR@3 | Document recall@3 | Median retrieval latency (ms) |
|---|---:|---:|---:|---:|---:|
| Dense / FAISS | 1.000 | 0.966 | 0.931 | 0.983 | 0.0242 |
| BM25 | 0.793 | 0.733 | 0.638 | 0.948 | 0.3548 |
| Hybrid RRF | 0.931 | 0.897 | 0.810 | 0.991 | 0.4670 |

Latency is warm in-memory search, including scope handling/tokenization/RRF but **excluding query embedding, PDF ingestion, model load, and index build**. Each query/method has 20 timed repetitions; table reports median of per-query medians across all 36 questions. Query encoding median: **75.57 ms** (not included above). FAISS/BM25 are warmed before timing. Method order rotates. Host: Windows / AMD64, Python 3.13.7. Local tiny-corpus timings are not production throughput claims.

### Question groups

| Group (answerable count) | Dense Hit / Recall / MRR | BM25 Hit / Recall / MRR | Hybrid Hit / Recall / MRR |
|---|---|---|---|
| exact (6) | 1.000 / 1.000 / 0.917 | 0.833 / 0.833 / 0.639 | 1.000 / 1.000 / 0.806 |
| semantic (6) | 1.000 / 1.000 / 0.917 | 0.667 / 0.667 / 0.583 | 0.833 / 0.833 / 0.722 |
| factual (4) | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 0.875 | 1.000 / 1.000 / 1.000 |
| scoped (4) | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 0.875 | 1.000 / 1.000 / 1.000 |
| follow_up (5) | 1.000 / 1.000 / 0.900 | 0.400 / 0.400 / 0.400 | 0.800 / 0.800 / 0.567 |
| cross_document (4) | 1.000 / 0.750 / 0.875 | 1.000 / 0.562 / 0.542 | 1.000 / 0.750 / 0.875 |

### Where hybrid helps or loses

- Versus Dense / FAISS: higher MRR on 1 (S01); lower on 6 (E01, E03, S02, S05, C03, C04).
- Versus BM25: higher MRR on 11 (E01, E03, E04, S02, S03, F01, D03, C01, C03, X02, X04); lower on 0 (none).

Relevant-page candidate ranks for hybrid regressions (`None` means absent from that top-20 pool):

| Case | Relevant page | Dense pool rank | BM25 pool rank | Fused rank |
|---|---|---:|---:|---:|
| E01 | battery_brief:4 | 2 | 4 | 3 |
| E03 | river_operations:5 | 1 | 3 | 2 |
| S02 | archive_handbook:1 | 1 | 12 | 3 |
| S05 | river_operations:2 | 1 | None | 14 |
| C03 | river_operations:5 | 1 | 8 | 3 |
| C04 | archive_handbook:3 | 1 | None | 14 |

RRF rewards agreement between ranked lists, including lexical overlap with distractors. It does not assess whether a page answers the question. In S05, the gauge-calibration page is strong in dense search but weaker lexically; unrelated pages supported by both lists displace it. E01's incident index merely mentions fault strings without defining them. The per-case ranks distinguish this fusion effect from extraction failure: the labelled pages survive PDF ingestion and dense retrieval. No algorithm or fixture was changed after observing scores.

## Follow-up reference retention

Six follow-ups compare the latest question alone with the existing user-question expansion. Intended-document labels are used only for scoring, never as automatic retrieval filters. The ambiguous funding → timeline case has no answerable timeline evidence, so it is excluded from evidence-hit averages but included in document-retention checks.

| Method | Raw → current evidence Hit@3 (5 answerable) | Raw → current intended-doc Hit@3 (6 cases) | Raw → current top-1 intended doc | Current intended-doc share in results |
|---|---|---|---|---|
| Dense / FAISS | 0.800 → 1.000 | 0.667 → 1.000 | 0.500 → 1.000 | 0.778 |
| BM25 | 0.400 → 0.400 | 0.500 → 1.000 | 0.333 → 0.833 | 0.500 |
| Hybrid RRF | 0.600 → 0.800 | 0.667 → 1.000 | 0.333 → 0.833 | 0.667 |

| Follow-up | Expansion applied | Hybrid evidence Hit@3 | Hybrid intended doc found / top-1 | Hybrid retrieved pages |
|---|---|---|---|---|
| C01: What about its limitations? | True | 1.0 | 1 / 1 | battery_brief:1, battery_brief:5, fleet_report:3 |
| C02: How often do they need checking? | True | 1.0 | 1 / 1 | fleet_report:2, fleet_report:5, fleet_report:1 |
| C03: How does it cope when contact is lost? | True | 1.0 | 1 / 1 | river_operations:1, building_energy:4, river_operations:5 |
| C04: And how are copies protected? | False | 0.0 | 1 / 0 | battery_brief:5, archive_handbook:1, incident_playbook:3 |
| C05: How long does that last? | True | 1.0 | 1 / 1 | building_energy:2, building_energy:1, building_energy:4 |
| C06: What about its timeline? | True | None | 1 / 1 | battery_brief:2, fleet_report:3, river_operations:4 |

The current heuristic uses previous user questions, not the previous assistant's referenced document. It may mix projects, miss implicit follow-ups without recognized pronouns, or retrieve the intended document without the needed page. Finding the intended document is weaker than resolving the reference correctly.

## Document scope and cross-document coverage

| Method | Unscoped → scoped Hit@3 (4 scope questions) | Unscoped → scoped MRR@3 |
|---|---|---|
| Dense / FAISS | 0.750 → 1.000 | 0.417 → 1.000 |
| BM25 | 0.750 → 1.000 | 0.333 → 0.875 |
| Hybrid RRF | 0.750 → 1.000 | 0.458 → 1.000 |

Cross-document page Recall@3: global hybrid **0.750**, existing application coverage helper **0.688** over four cases. Helper mode: hybrid, existing configured relevance heuristic; no changes. More document coverage is not proof that the right evidence was selected.

## Unanswerable and refusal behavior

| Method | Unanswerable questions returning nonempty evidence (7 cases) |
|---|---:|
| Dense / FAISS | 7/7 |
| BM25 | 7/7 |
| Hybrid RRF | 7/7 |

**Zero live provider calls.** A mocked generation contract passed for 7/7 cases: the real answer flow supplied its refusal instruction and propagated the mock refusal. The mock deliberately supplies a refusal; this is a plumbing check, **not measured refusal accuracy**. Nonempty retrieval is not proof of answerability. The system has no tested semantic abstention gate, and similar grants/timelines can supply plausible but wrong evidence. Real model refusal correctness remains unmeasured in this evaluation.

## Inspectable case results

Page shorthand is `document:page`. Relevance labels are in the dataset. Empty BM25 results are retained as misses; no method gets a different denominator.

| ID / group | Labelled pages | Dense top-3 | BM25 top-3 | Hybrid top-3 |
|---|---|---|---|---|
| E01 / exact | battery_brief:4 | incident_playbook:4, battery_brief:4, fleet_report:2 | incident_playbook:4, fleet_report:2, building_energy:4 | incident_playbook:4, fleet_report:2, battery_brief:4 |
| E02 / exact | fleet_report:2 | fleet_report:2, battery_brief:4, incident_playbook:4 | fleet_report:2, incident_playbook:4, battery_brief:4 | fleet_report:2, battery_brief:4, incident_playbook:4 |
| E03 / exact | river_operations:5 | river_operations:5, incident_playbook:4, building_energy:4 | incident_playbook:4, building_energy:4, river_operations:5 | incident_playbook:4, river_operations:5, building_energy:4 |
| E04 / exact | building_energy:4 | building_energy:4, incident_playbook:4, river_operations:5 | incident_playbook:4, building_energy:4, river_operations:5 | building_energy:4, incident_playbook:4, river_operations:5 |
| E05 / exact | archive_handbook:4 | archive_handbook:4, incident_playbook:4, archive_handbook:1 | archive_handbook:4, incident_playbook:1, building_energy:3 | archive_handbook:4, incident_playbook:4, incident_playbook:1 |
| E06 / exact | battery_brief:3 | battery_brief:3, incident_playbook:1, fleet_report:4 | battery_brief:3, archive_handbook:2, archive_handbook:1 | battery_brief:3, fleet_report:2, building_energy:4 |
| S01 / semantic | battery_brief:3 | battery_brief:5, battery_brief:3, building_energy:2 | battery_brief:3, building_energy:1, battery_brief:1 | battery_brief:3, building_energy:1, battery_brief:5 |
| S02 / semantic | archive_handbook:1 | archive_handbook:1, battery_brief:5, fleet_report:4 | incident_playbook:5, archive_handbook:3, incident_playbook:4 | archive_handbook:3, battery_brief:5, archive_handbook:1 |
| S03 / semantic | fleet_report:4 | fleet_report:4, battery_brief:5, fleet_report:2 | river_operations:1, fleet_report:4, fleet_report:3 | fleet_report:4, river_operations:1, battery_brief:3 |
| S04 / semantic | building_energy:1 | building_energy:1, building_energy:3, building_energy:2 | building_energy:1, battery_brief:1, archive_handbook:2 | building_energy:1, building_energy:3, building_energy:2 |
| S05 / semantic | river_operations:2 | river_operations:2, fleet_report:4, building_energy:4 | river_operations:1, battery_brief:4, building_energy:3 | river_operations:1, building_energy:4, battery_brief:4 |
| S06 / semantic | incident_playbook:2 | incident_playbook:2, battery_brief:3, archive_handbook:3 | incident_playbook:2, building_energy:5, river_operations:3 | incident_playbook:2, building_energy:4, fleet_report:4 |
| F01 / factual | battery_brief:1 | battery_brief:1, battery_brief:5, battery_brief:2 | building_energy:1, battery_brief:1, fleet_report:5 | battery_brief:1, fleet_report:5, battery_brief:5 |
| F02 / factual | fleet_report:1 | fleet_report:1, fleet_report:4, fleet_report:5 | fleet_report:1, fleet_report:3, incident_playbook:3 | fleet_report:1, fleet_report:3, fleet_report:4 |
| F03 / factual | archive_handbook:1 | archive_handbook:1, archive_handbook:4, archive_handbook:5 | archive_handbook:1, archive_handbook:5, incident_playbook:1 | archive_handbook:1, archive_handbook:5, archive_handbook:4 |
| F04 / factual | river_operations:3 | river_operations:3, river_operations:4, building_energy:4 | river_operations:3, river_operations:4, river_operations:2 | river_operations:3, river_operations:4, building_energy:4 |
| D01 / scoped | river_operations:4 | river_operations:4, river_operations:3, river_operations:1 | river_operations:4, river_operations:1, river_operations:3 | river_operations:4, river_operations:3, river_operations:1 |
| D02 / scoped | fleet_report:3 | fleet_report:3, fleet_report:4, fleet_report:1 | fleet_report:3, fleet_report:4, fleet_report:5 | fleet_report:3, fleet_report:4, fleet_report:1 |
| D03 / scoped | building_energy:3 | building_energy:3, building_energy:5, building_energy:4 | building_energy:4, building_energy:3, building_energy:2 | building_energy:3, building_energy:4, building_energy:5 |
| D04 / scoped | fleet_report:2 | fleet_report:2, fleet_report:4, fleet_report:1 | fleet_report:2, fleet_report:3, fleet_report:4 | fleet_report:2, fleet_report:4, fleet_report:3 |
| C01 / follow_up | battery_brief:5 | battery_brief:1, battery_brief:5, battery_brief:2 | battery_brief:1, fleet_report:3, building_energy:1 | battery_brief:1, battery_brief:5, fleet_report:3 |
| C02 / follow_up | fleet_report:2 | fleet_report:2, fleet_report:5, fleet_report:4 | fleet_report:2, archive_handbook:4, building_energy:4 | fleet_report:2, fleet_report:5, fleet_report:1 |
| C03 / follow_up | river_operations:5 | river_operations:5, river_operations:1, building_energy:4 | river_operations:1, building_energy:4, fleet_report:4 | river_operations:1, building_energy:4, river_operations:5 |
| C04 / follow_up | archive_handbook:3 | archive_handbook:3, archive_handbook:4, incident_playbook:3 | battery_brief:5, river_operations:2, archive_handbook:1 | battery_brief:5, archive_handbook:1, incident_playbook:3 |
| C05 / follow_up | building_energy:2 | building_energy:2, building_energy:1, building_energy:3 | building_energy:2, building_energy:4, building_energy:1 | building_energy:2, building_energy:1, building_energy:4 |
| C06 / follow_up | unanswerable | battery_brief:2, archive_handbook:5, river_operations:4 | battery_brief:2, fleet_report:3, battery_brief:1 | battery_brief:2, fleet_report:3, river_operations:4 |
| X01 / cross_document | battery_brief:4, fleet_report:2 | battery_brief:4, fleet_report:2, battery_brief:1 | battery_brief:4, river_operations:1, fleet_report:3 | battery_brief:4, fleet_report:3, battery_brief:1 |
| X02 / cross_document | archive_handbook:3, river_operations:5 | river_operations:5, archive_handbook:1, archive_handbook:4 | incident_playbook:3, river_operations:1, archive_handbook:3 | river_operations:5, incident_playbook:3, archive_handbook:3 |
| X03 / cross_document | battery_brief:4, fleet_report:2 | incident_playbook:4, fleet_report:2, battery_brief:4 | incident_playbook:4, fleet_report:2, battery_brief:4 | incident_playbook:4, fleet_report:2, battery_brief:4 |
| X04 / cross_document | battery_brief:4, fleet_report:2, river_operations:2, building_energy:3 | fleet_report:2, battery_brief:4, fleet_report:3 | building_energy:4, river_operations:4, building_energy:3 | battery_brief:4, fleet_report:2, building_energy:4 |
| U01 / unanswerable | unanswerable | battery_brief:2, fleet_report:3, battery_brief:1 | battery_brief:2, battery_brief:1, fleet_report:3 | battery_brief:2, fleet_report:3, battery_brief:1 |
| U02 / unanswerable | unanswerable | fleet_report:5, fleet_report:3, fleet_report:4 | fleet_report:2, building_energy:5, archive_handbook:4 | fleet_report:5, fleet_report:2, fleet_report:3 |
| U03 / unanswerable | unanswerable | river_operations:5, archive_handbook:4, river_operations:1 | building_energy:4, river_operations:5, archive_handbook:4 | river_operations:5, building_energy:4, archive_handbook:4 |
| U04 / unanswerable | unanswerable | archive_handbook:5, archive_handbook:1, archive_handbook:4 | archive_handbook:5, archive_handbook:4, archive_handbook:3 | archive_handbook:5, archive_handbook:4, archive_handbook:1 |
| U05 / unanswerable | unanswerable | building_energy:2, building_energy:5, building_energy:1 | building_energy:5, fleet_report:5, building_energy:2 | building_energy:5, building_energy:2, fleet_report:5 |
| U06 / unanswerable | unanswerable | battery_brief:3, archive_handbook:4, battery_brief:4 | incident_playbook:2, river_operations:3, incident_playbook:5 | incident_playbook:2, battery_brief:4, battery_brief:3 |

## Interpretation and limits

This fixture measures local page retrieval, exact-term/paraphrase trade-offs, selected-document filtering, and current follow-up/coverage policies. It does not measure end-to-end answer truth, semantic entailment, OCR quality, generalization to arbitrary PDFs, deployment throughput, or actual LLM refusal reliability. Citation reference validation checks only that cited document/page references occur in retrieved evidence; it cannot prove factual correctness.

Reproduce: `python -m scripts.evaluate_retrieval` using the populated `venv` and existing offline model cache. Only this current Markdown artifact is written; generated PDFs are temporary. Older smoke reports are historical integration checks, not the main evaluation.
