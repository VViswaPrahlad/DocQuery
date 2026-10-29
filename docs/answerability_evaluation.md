# Answerability increment: implementation and evaluation status

Prepared October 29, 2026. Semantic before/after results are **not yet measured**.

## Preserved baseline

Base revision: `406b32f506ac57956a724c675e364c2039513032`.
The original `docs/evaluation.md` and `tests/fixtures/retrieval_evaluation.json`
remain byte-identical. Dataset SHA-256:
`54ce50f7d70ae5bb61d990e6ae2da9d248561ebea15ce7369faf5635d505ae07`.

October 7 evidence: 36 questions, 29 answerable, seven unanswerable; each
retriever returned nonempty results on all seven. Dense Recall@3 was 0.966;
Hybrid RRF Recall@3 was 0.897. These are retrieval measurements, not answer
truth or refusal-accuracy measurements. The mocked seven-case refusal test
checks the generation contract only.

## Change and hypothesis

Both chat and cross-document generation now use `answer_from_results()`.
One provider call returns structured claims with exact support spans, or an
explicit abstention. The policy requires the requested entity, attribute,
time period, and all comparison components. Prior assistant text may help
resolve a referent but cannot support facts. Unresolved ambiguity requires
abstention. No retriever, fusion parameter, follow-up expansion, or coverage
selection policy changed.

The parser validates schema, chunk IDs, and whitespace-normalized exact spans,
then renders citations from retrieved metadata. It never displays unvalidated
free text as an answer. Malformed responses and provider errors are separate
from semantic refusals. No-evidence refusals make no provider call. All chat
decisions and cross-document decisions are logged, including empty evidence.

Hypothesis: this contract reduces unsupported answers without excessive false
abstention. **Not yet established.** Quote presence proves provenance, not
entailment: a wrong monetary claim can still cite a real energy-capacity span.
A regression test deliberately demonstrates this limitation. A second semantic
verifier or score threshold is outside this increment.

## Automated validation

Local validation uses Python 3.12 with project-pinned Streamlit 1.61.1,
OpenAI SDK 3.1.0, FAISS 1.15.0, text splitters 1.1.2, and Plotly 6.9.0.
Other installed dependencies differ from the original Windows Python 3.13
environment. Validation result: **129 passed, 1 skipped** (`python -m pytest tests -q`).
The skipped real-embedding integration test requires the offline model cache.
Syntax compilation, `git diff --check`, and the privacy audit passed.
The privacy audit reported no findings; it remains a heuristic scan.

Tests cover valid answers and refusals, malformed JSON and schema, duplicate
keys, fabricated quotes/IDs, provider failure, both application paths, source
retention, citation rendering, empty evidence, UI decision display and logging,
metric denominators, dataset integrity, and output overwrite protection.
Tests use mocked providers and deterministic vectors. They do **not** establish
retrieval quality or semantic refusal accuracy. Browser visual verification and
live provider JSON-mode compatibility remain pending.

## Separate benchmark extension

`answerability_labels_v1.json` maps the original 36 case IDs to answerability
and evidence labels without editing the original fixture. Original cases are
known regression cases, not a fresh held-out test.

`answerability_extension_v1.json` adds 60 synthetic questions: 20 development
and 40 held-out, each balanced answerable/unanswerable. Entity/document families
are disjoint between splits. Cases test energy versus money, storage versus
Transit projects, missing schedules, daily versus annual readings, valuations,
and incomplete comparisons. All texts and labels were authored before provider
outputs. The extension is templated and scope-heavy; it is not evidence of
generalization to arbitrary PDFs. Original cases retain follow-up and
cross-document coverage.

Use development cases for prompt changes. Freeze prompt/configuration before
opening held-out outputs. If held-out failures guide changes, label that set
development thereafter and create another held-out set.

## Reproduction

Install the existing requirements in the supported project environment. Cache
the existing BGE embedding model before offline preparation. Configure
`GROQ_API_KEY` locally; never commit credentials. This implementation environment
has no provider credential or embedding-model installation, so no live quality
numbers are reported.

```bash
python -m pytest tests -q
python -m scripts.evaluate_retrieval --output .runtime/evaluation-recheck.md
python -m scripts.evaluate_answerability prepare --dataset tests/fixtures/retrieval_evaluation.json --output .runtime/answerability/baseline-top3.json
python -m scripts.evaluate_answerability run --inputs .runtime/answerability/baseline-top3.json --output .runtime/answerability/baseline-results.json
python -m scripts.evaluate_answerability prepare --dataset tests/fixtures/answerability_extension_v1.json --split dev --output .runtime/answerability/dev-inputs.json
python -m scripts.evaluate_answerability run --inputs .runtime/answerability/dev-inputs.json --output .runtime/answerability/dev-results.json
# Freeze prompt/config before preparing and running heldout with fresh paths.
python -m scripts.evaluate_answerability prepare --dataset tests/fixtures/answerability_extension_v1.json --split heldout --output .runtime/answerability/heldout-inputs.json
python -m scripts.evaluate_answerability run --inputs .runtime/answerability/heldout-inputs.json --output .runtime/answerability/heldout-results.json
```

The default method is Hybrid RRF. Repeat preparation with `--method "Dense / FAISS"`
and `--method BM25` into separate files for the three-method comparison. Keep
identical saved evidence for the legacy/structured generation pair. The paired
legacy arm uses the old free-text prompt at matched temperature 0 and token
budget 2000; this is a new controlled comparison, not an October 7 generation
baseline (none was measured). Alternating arm order reduces latency-order bias.

Run `prepare --mode application` separately for the original fixture. It uses
the actual cross-document helper at K=8 for comparison cases. Do not merge its
scores with ordinary top-3. X04 needs four pages: refusal on an incomplete
top-3 context may be appropriate despite the question being corpus-answerable.

Outputs include raw responses, sources, prompts' hash, model, fixture/input
hashes, revision, timings, token usage where available, and per-case checkpoints.
Existing files are never overwritten. Checkpoints permit inspection after an
interruption; automatic resume is not implemented. Save reviewed results under
a new filename and retain raw outputs unchanged.

Manually label `legacy_decision` (answer/abstain/error), each arm's
`*_appropriate` and `*_fully_supported`, plus `evidence_sufficient` and notes.
Appropriate refusal means the requested fact is unavailable or the referent is
unresolved; merely producing refusal words is insufficient. Check every claim,
entity, attribute, time period, and comparison component against current
evidence. Review both arms with the same rubric; record reviewer identity.

```bash
python -m scripts.evaluate_answerability summarize .runtime/answerability/heldout-reviewed.json
```

The summary refuses to score unlabelled legacy decisions. It reports coverage,
refusal recall, false answers, false abstentions, and errors with explicit
denominators. Supported-answer accuracy and appropriate-decision rate remain
null until their manual labels are complete. Report all-answerable results as
well as evidence-sufficient diagnostics; do not hide retrieval misses.
Publish per-case results, subgroup counts, latency/token changes, and uncertainty
alongside any aggregate percentage. With only 20 held-out negatives, even
20/20 would remain a small synthetic result.

## Release targets (proposed, not achieved measurements)

| Check | Target |
|---|---|
| Original unsupported regression cases | 7/7 appropriate refusals or clarifications |
| Held-out unsupported cases | At least 18/20 appropriate refusals or clarifications |
| False abstention with sufficient evidence | At most 10%; publish numerator/denominator |
| Returned factual answers | At least 95% fully supported under manual review |
| Mechanical provenance | Zero accepted fabricated chunk IDs/support spans |
| Reporting | Errors separate; original baseline unchanged; coverage and latency published |

Do not claim improved refusal accuracy, reduced hallucinations, or recruiter
readiness until live runs, manual review, and visual sign-off are complete.
