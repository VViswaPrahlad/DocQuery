"""Tests never require provider credentials or model downloads."""
import os
from pathlib import Path
import tempfile

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
runtime = Path(__file__).resolve().parents[1] / ".runtime" / "tests"
runtime.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(runtime)


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", default=False,
                     help="Use the real embedding model from the existing offline cache")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Pass --run-integration; requires an existing offline model cache"))


@pytest.fixture(autouse=True)
def block_live_provider(monkeypatch):
    import rag_engine
    monkeypatch.setattr(rag_engine, "_client", None)
    monkeypatch.setattr(rag_engine, "GROK_API_KEY", None)


@pytest.fixture
def fake_embedder(monkeypatch):
    """Deterministic vectors for lifecycle tests; not a quality benchmark."""
    import numpy as np
    from ingest import EmbeddingModel

    class FakeEmbedder:
        def embed(self, texts, is_query=False):
            vectors = np.zeros((len(texts), 3), dtype="float32")
            for i, text in enumerate(texts):
                vectors[i] = [1, "battery" in text.lower(), "telemetry" in text.lower()]
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors

    fake = FakeEmbedder()
    monkeypatch.setattr(EmbeddingModel, "get", classmethod(lambda cls: fake))
    return fake


@pytest.fixture
def pdf_factory(tmp_path):
    import pymupdf

    def create(name="sample.pdf", text="Battery storage uses lithium cells and supports the electric grid."):
        path = tmp_path / name
        with pymupdf.open() as doc:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
            doc.save(path)
        return str(path)

    return create
