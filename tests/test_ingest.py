"""
tests/test_ingest.py

Unit tests for chunk_document() and ocr_utils.needs_ocr(). Both are pure
logic and don't need a real PDF, embeddings, or an API key to test.

Run:
    pytest tests/test_ingest.py -v
"""

from ingest import chunk_document
from ocr_utils import needs_ocr, MIN_NATIVE_CHARS


def test_chunk_never_spans_pages():
    """A chunk built from page 1's text should never carry page 2's text,
    since chunk_document() splits per-page before running the splitter."""
    pages = [
        {"page_number": 1, "text": "Alpha content. " * 50, "source": "native"},
        {"page_number": 2, "text": "Beta content. " * 50, "source": "native"},
    ]
    chunks = chunk_document("doc.pdf", pages)

    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]

    assert all("Beta" not in c.text for c in page1_chunks)
    assert all("Alpha" not in c.text for c in page2_chunks)


def test_chunk_ids_are_unique_and_traceable():
    pages = [{"page_number": 1, "text": "Some content. " * 100, "source": "native"}]
    chunks = chunk_document("report.pdf", pages)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))          # all unique
    assert all("report.pdf" in cid for cid in ids)
    assert all(c.doc_name == "report.pdf" for c in chunks)


def test_ocr_source_flag_propagates_to_chunks():
    """A page extracted via OCR should produce chunks flagged source='ocr',
    so the UI can show which answers came from a scanned page."""
    pages = [{"page_number": 1, "text": "Scanned text. " * 50, "source": "ocr"}]
    chunks = chunk_document("scan.pdf", pages)
    assert all(c.source == "ocr" for c in chunks)


def test_short_page_needs_ocr():
    """A page with only a stray page number isn't real content."""
    assert needs_ocr("7") is True
    assert needs_ocr("") is True


def test_normal_page_does_not_need_ocr():
    long_text = "This page has plenty of real extracted text content. " * 3
    assert len(long_text) > MIN_NATIVE_CHARS
    assert needs_ocr(long_text) is False


def test_needs_ocr_boundary():
    """Text exactly at the threshold should not trigger OCR; one character
    short of it should."""
    at_threshold = "x" * MIN_NATIVE_CHARS
    below_threshold = "x" * (MIN_NATIVE_CHARS - 1)
    assert needs_ocr(at_threshold) is False
    assert needs_ocr(below_threshold) is True