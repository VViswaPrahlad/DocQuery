from scripts.privacy_audit import ignore_reason


def test_ignored_runtime_and_explicit_public_configuration():
    rules = ['.env', '.env.*', '!.env.example', 'venv/', '.runtime/', 'data/uploads/*',
             'data/index/*', '.streamlit/*', '!.streamlit/config.toml']
    for path in ('.env', '.env.production', 'venv/Lib/library.py', '.runtime/demo/file.pdf',
                 'data/uploads/private.pdf', 'data/index/chunks.json', '.streamlit/secrets.toml'):
        assert ignore_reason(path, rules)
    assert ignore_reason('.env.example', rules) is None
    assert ignore_reason('.streamlit/config.toml', rules) is None
    assert ignore_reason('tests/fixtures/corpus.json', rules) is None


def test_root_anchored_cleanup_rules():
    rules = ['/README_OCR.md', '/docs/architecture.md', '/evaluation*.json', '/.idea/']
    assert ignore_reason('README_OCR.md', rules)
    assert ignore_reason('docs/architecture.md', rules)
    assert ignore_reason('evaluation-local.json', rules)
    assert ignore_reason('.idea/settings.xml', rules)
    assert ignore_reason('nested/README_OCR.md', rules) is None
    assert ignore_reason('nested/evaluation-local.json', rules) is None
    assert ignore_reason('nested/.idea/settings.xml', rules) is None
