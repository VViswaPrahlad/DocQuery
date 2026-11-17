import hashlib
import json
from pathlib import Path

import pytest
from scripts.evaluate_answerability import BASELINE_HASH, write_new, prepare, run, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_is_byte_identical_and_extension_split_is_disjoint():
    assert hashlib.sha256((ROOT / 'tests/fixtures/retrieval_evaluation.json').read_bytes()).hexdigest() == BASELINE_HASH
    fixture = json.loads((ROOT / 'tests/fixtures/answerability_extension_v1.json').read_text())
    cases = fixture['cases']
    assert len(cases) == 60 and len({c['id'] for c in cases}) == 60
    names = {}
    for split, count in [('dev', 20), ('heldout', 40)]:
        selected = [c for c in cases if c['split'] == split]
        assert len(selected) == count
        assert sum(c['unanswerable'] for c in selected) == count // 2
        names[split] = {n for c in selected for n in c['scope']}
    assert not names['dev'] & names['heldout']


def test_output_cannot_overwrite_baseline_or_existing_results(tmp_path):
    path = tmp_path / 'results.json'
    write_new(path, {'first': True})
    with pytest.raises(FileExistsError):
        write_new(path, {'second': True})


def test_prepare_real_ingestion_with_fake_vectors_is_only_a_contract_check(tmp_path, fake_embedder):
    output = tmp_path / 'inputs.json'
    prepare(ROOT / 'tests/fixtures/retrieval_evaluation.json', output, 'all', 'top3', 'Hybrid RRF')
    data = json.loads(output.read_text())
    assert len(data['rows']) == 36
    assert data['dataset_sha256'] == BASELINE_HASH
    assert sum(r['case'].get('unanswerable', False) for r in data['rows']) == 7
    for row in data['rows']:
        if row['case'].get('scope'):
            assert all(s['doc_name'] in row['case']['scope'] for s in row['sources'])


def test_run_requires_credentials_without_creating_false_results(tmp_path, monkeypatch):
    import rag_engine as rag
    inputs = tmp_path / 'inputs.json'
    write_new(inputs, {'rows': []})
    monkeypatch.setattr(rag, 'get_client', lambda: (_ for _ in ()).throw(RuntimeError('missing credential')))
    with pytest.raises(RuntimeError):
        run(inputs, tmp_path / 'outputs.json')
    assert not (tmp_path / 'outputs.json').exists()


def test_summary_requires_manual_legacy_labels(tmp_path):
    path = tmp_path / 'outputs.json'
    write_new(path, {'rows': [{'case': {'unanswerable': True}, 'outputs': {'legacy': {'decision': 'needs_manual_label'}}, 'review': {'legacy_decision': None}}]})
    with pytest.raises(ValueError, match='manual'):
        summarize(path)


def test_paired_runner_reuses_evidence_and_records_raw_outputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import rag_engine as rag
    inputs = tmp_path / 'inputs.json'
    write_new(inputs, {'dataset_sha256': 'fixture', 'split': 'dev', 'mode': 'top3', 'method':'Hybrid RRF', 'rows':[
        {'case': {'id':'U', 'question':'Grant amount?', 'unanswerable':True},
         'sources':[{'chunk_id':'a','doc_name':'a.pdf','page':1,'text':'No amount recorded.','score':0.03,'score_type':'rrf'}]}]})
    requests = []
    monkeypatch.setattr(rag, 'get_client', lambda: object())
    def complete(*args, **kwargs):
        requests.append(kwargs)
        content = '{"decision":"abstain","claims":[],"reason":"insufficient_evidence"}' if kwargs.get('response_format') else 'Not enough evidence.'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    monkeypatch.setattr(rag, '_safe_chat_completion', complete)
    output = tmp_path / 'results.json'
    run(inputs, output)
    data = json.loads(output.read_text())
    assert len(requests) == 2
    assert all('No amount recorded.' in r['messages'][-1]['content'] for r in requests)
    assert all(r['temperature'] == 0 and r['max_tokens'] == 2000 for r in requests)
    assert data['rows'][0]['outputs']['structured']['decision'] == 'abstain'
    assert data['rows'][0]['outputs']['structured']['raw_response']
    assert data['semantic_review'] == 'pending'
    with pytest.raises(FileExistsError):
        run(inputs, output)
