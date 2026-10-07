"""
ingest.py
---------
Handles the ingestion side of the RAG pipeline:
  1. PDF text extraction (PyMuPDF), with automatic OCR fallback for
     scanned/image-only pages (see ocr_utils.py)
  2. Intelligent chunking (recursive, page-boundary preserving)
  3. Embedding generation (BGE via sentence-transformers)
  4. FAISS vector index build / save / load

Each chunk stores metadata: {doc_name, page_number, chunk_id, source}
so answers can always be traced back to an exact page in an exact PDF,
and the UI can show whether a page's text came from OCR.
"""

import os
import pickle
import json
import hashlib
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

import pymupdf as fitz
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ocr_utils import extract_page_text, check_tesseract_available
from config import ROOT, get_config, positive_int

INDEX_DIR = str(ROOT / "data" / "index")
CHUNK_SIZE = positive_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = int(get_config("CHUNK_OVERLAP", 120))
if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be nonnegative and smaller than CHUNK_SIZE")
EMBEDDING_MODEL_NAME = get_config("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
OCR_FALLBACK_ENABLED = str(get_config("OCR_FALLBACK", "true")).lower() == "true"


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    source: str = "native"  # "native" or "ocr" -- which extraction path produced this page's text


def extract_pages(pdf_path: str, ocr_fallback: bool = OCR_FALLBACK_ENABLED) -> List[dict]:
    """
    Extract text per page from a PDF using PyMuPDF, falling back to
    Tesseract OCR (via ocr_utils) for pages whose native text layer is
    empty or too thin to be real content -- e.g. scanned pages, or
    pages that are mostly a table/diagram image.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages = []
    ocr_page_count = 0

    try:
        for page_num in range(total_pages):
            page = doc[page_num]
            text, source = extract_page_text(page, ocr_fallback=ocr_fallback)
            if source == "ocr":
                ocr_page_count += 1
            if text.strip():
                pages.append({"page_number": page_num + 1, "text": text, "source": source})
    finally:
        doc.close()

    if ocr_page_count:
        print(f"[ingest] {os.path.basename(pdf_path)}: OCR fallback used on "
              f"{ocr_page_count}/{total_pages} page(s)")

    return pages


def chunk_document(doc_name: str, pages: List[dict]) -> List[Chunk]:
    """
    Chunk text per page (so a chunk never spans across page numbers,
    keeping citations accurate) using a recursive character splitter.
    Each resulting chunk inherits the page's extraction source
    ("native" or "ocr") so it can be surfaced in the UI.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    counter = 0
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for pc in page_chunks:
            counter += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_name}_p{page['page_number']}_c{counter}",
                    doc_name=doc_name,
                    page_number=page["page_number"],
                    text=pc.strip(),
                    source=page.get("source", "native"),
                )
            )
    return chunks


class EmbeddingModel:
    """Thin wrapper so the model is loaded once and reused."""

    _instance = None

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        offline = str(get_config("HF_HUB_OFFLINE", "0")).lower() in {"1", "true"}
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=offline)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = EmbeddingModel()
        return cls._instance

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        # BGE models recommend an instruction prefix for queries (not for docs)
        if is_query:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,  # so inner product == cosine similarity
            show_progress_bar=False,
            batch_size=32,
        )
        return np.array(vecs, dtype="float32")


class VectorStore:
    """FAISS-backed vector store with a metadata sidecar, plus an
    optional lazily-built BM25 index for hybrid search."""

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized inner product
        self.chunks: List[Chunk] = []
        self._bm25 = None  # built on first hybrid_search() call, or invalidated by add()
        self.document_hashes = {}
        self.embedding_model = EMBEDDING_MODEL_NAME

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]):
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape != (len(chunks), self.index.d):
            raise ValueError("Embedding shape must match chunk count and index dimension")
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings must contain finite values")
        ids = [c.chunk_id for c in self.chunks] + [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk IDs; replace the document instead of appending it")
        if not chunks:
            return
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self._bm25 = None  # stale now that chunk list changed -- rebuild lazily next search

    def remove_document(self, doc_name: str):
        """
        Remove all chunks belonging to doc_name and rebuild the FAISS
        index from the REMAINING chunks' original embeddings. No
        re-embedding needed: IndexFlatIP stores raw vectors directly, so
        we can reconstruct them and rebuild a smaller index in place.
        """
        if not any(c.doc_name == doc_name for c in self.chunks):
            return  # nothing to remove

        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        keep_indices = [i for i, c in enumerate(self.chunks) if c.doc_name != doc_name]

        dim = self.index.d
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = [self.chunks[i] for i in keep_indices]
        if keep_indices:
            self.index.add(all_vectors[keep_indices])
        self._bm25 = None  # stale -- rebuild lazily next search
        self.document_hashes.pop(doc_name, None)

    def list_documents(self) -> List[str]:
        return sorted(set(c.doc_name for c in self.chunks))

    def search(self, query_vec: np.ndarray, top_k: int = 5, doc_filter=None):
        """Pure semantic (FAISS) search. Each result also carries "idx"
        (the chunk's position in self.chunks) so hybrid_search() can
        fuse this with BM25 ranks without re-looking-up chunks."""
        if not self.chunks or top_k <= 0:
            return []
        if doc_filter:
            candidates = [i for i, c in enumerate(self.chunks) if c.doc_name in doc_filter]
            if not candidates:
                return []
            # IndexFlatIP retains normalized vectors. Rank only the selected
            # subset, so globally dominant documents cannot hide scoped evidence.
            vectors = self.index.reconstruct_n(0, self.index.ntotal)[candidates]
            similarities = vectors @ query_vec[0]
            ranked = np.argsort(-similarities, kind="stable")[:top_k]
            return [{"chunk": self.chunks[candidates[i]], "score": float(similarities[i]),
                     "semantic_score": float(similarities[i]), "score_type": "cosine", "idx": candidates[i]} for i in ranked]
        scores, idxs = self.index.search(query_vec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append({"chunk": self.chunks[idx], "score": float(score), "semantic_score": float(score), "score_type": "cosine", "idx": int(idx)})
        return results

    def hybrid_search(self, query_vec: np.ndarray, query_text: str, top_k: int = 5,
                       semantic_k: int = 20, bm25_k: int = 20, rrf_k: int = 60, doc_filter=None):
        """
        Semantic (FAISS) + BM25 fused via Reciprocal Rank Fusion.
        semantic_k/bm25_k control how many candidates each method
        contributes to the fusion pool before cutting down to top_k --
        wider than top_k so a chunk that's strong in only one method
        still gets a chance to be considered.
        """
        from hybrid_search import BM25Index, reciprocal_rank_fusion

        if not self.chunks or top_k <= 0:
            return []

        if self._bm25 is None:
            self._bm25 = BM25Index.from_store(self)

        semantic_results = self.search(query_vec, top_k=max(semantic_k, top_k), doc_filter=doc_filter)
        semantic_rank = [r["idx"] for r in semantic_results]
        allowed = [i for i, c in enumerate(self.chunks) if c.doc_name in doc_filter] if doc_filter else None
        lexical_results = self._bm25.search(query_text, top_k=max(bm25_k, top_k), allowed_indices=allowed)
        bm25_rank = [idx for idx, _ in lexical_results]
        lexical_scores = dict(lexical_results)
        semantic_scores = {r["idx"]: r["score"] for r in semantic_results}

        fused = reciprocal_rank_fusion([semantic_rank, bm25_rank], k=rrf_k)[:top_k]
        return [
            {"chunk": self.chunks[idx], "score": rrf_score, "score_type": "rrf", "idx": idx,
             "semantic_score": semantic_scores.get(idx), "lexical_score": lexical_scores.get(idx, 0.0)}
            for idx, rrf_score in fused
        ]

    def save(self, path: str = INDEX_DIR):
        """Save JSON metadata and a manifest; manifest checks detect partial writes."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        metadata_path = Path(path) / "chunks.json"
        metadata_path.write_text(json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False), encoding="utf-8")
        manifest = {
            "version": 1, "embedding_model": self.embedding_model,
            "dimension": self.index.d, "count": len(self.chunks),
            "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
            "document_hashes": self.document_hashes,
            "sha256": {name: hashlib.sha256((Path(path) / name).read_bytes()).hexdigest()
                       for name in ("faiss.index", "chunks.json")},
        }
        (Path(path) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str = INDEX_DIR, allow_legacy: bool = False):
        path = Path(path)
        if not (path / "faiss.index").is_file():
            raise FileNotFoundError("No saved FAISS index. Build and save an index with ingest.build_index_from_pdfs(..., persist=True).")
        if not (path / "manifest.json").is_file():
            if not allow_legacy:
                raise ValueError("Legacy index has no model manifest. Rebuild with ingest.build_index_from_pdfs(..., persist=True).")
            warnings.warn("Legacy index: embedding model identity cannot be verified", RuntimeWarning)
            class DataOnlyUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    raise ValueError("Legacy metadata may contain only plain data")
            with (path / "chunks.pkl").open("rb") as f:
                raw_chunks = DataOnlyUnpickler(f).load()
            manifest = {}
        else:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("version") != 1 or manifest.get("embedding_model") != EMBEDDING_MODEL_NAME:
                raise ValueError("Saved index version/model differs from configuration; rebuild the index")
            for name in ("faiss.index", "chunks.json"):
                if not (path / name).is_file():
                    raise FileNotFoundError(f"Saved index is incomplete: missing {name}")
                if hashlib.sha256((path / name).read_bytes()).hexdigest() != manifest.get("sha256", {}).get(name):
                    raise ValueError(f"Saved index integrity check failed: {name}")
            raw_chunks = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(path / "faiss.index"))
        store = cls(dim=index.d)
        store.index = index
        # Chunk gained a "source" field after some indexes may have been
        # built; default it for any pickled chunk that predates the field.
        store.chunks = [Chunk(**{**{"source": "native"}, **c}) for c in raw_chunks]
        if index.ntotal != len(store.chunks) or len({c.chunk_id for c in store.chunks}) != len(store.chunks):
            raise ValueError("Index vectors and chunk metadata are inconsistent")
        if manifest and (manifest.get("dimension") != index.d or manifest.get("count") != index.ntotal):
            raise ValueError("Index dimensions/count do not match the manifest")
        if any(not c.text.strip() or c.page_number < 1 or c.source not in {"native", "ocr"} for c in store.chunks):
            raise ValueError("Invalid chunk metadata in saved index")
        store.document_hashes = manifest.get("document_hashes", {})
        return store


def build_index_from_pdfs(pdf_paths: List[str], persist: bool = True) -> VectorStore:
    """
    Full ingestion pipeline: extract (with OCR fallback) -> chunk -> embed -> index.

    persist=False keeps everything in memory only (no data/index/ write).
    Use this when running as a shared public demo (e.g. Streamlit Community
    Cloud), since that platform runs ONE process for every visitor --
    writing to a fixed disk path would let concurrent users silently
    overwrite each other's index. In-memory + st.session_state already
    gives correct per-visitor isolation without touching disk at all.
    """
    if OCR_FALLBACK_ENABLED and not check_tesseract_available():
        print("[ingest] WARNING: OCR_FALLBACK is enabled but Tesseract is not installed "
              "or not on PATH. Native text will be kept; image-only pages cannot be indexed. "
              "See the OCR setup section in README.md for install instructions.")

    all_chunks: List[Chunk] = []
    hashes = {}

    for pdf_path in pdf_paths:
        doc_name = os.path.basename(pdf_path)
        digest = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
        if doc_name in hashes:
            if hashes[doc_name] == digest:
                continue
            raise ValueError(f"Two different PDFs have the same filename: {doc_name}")
        pages = extract_pages(pdf_path)
        doc_chunks = chunk_document(doc_name, pages)
        if not doc_chunks:
            raise ValueError(f"No extractable text in {doc_name}. Scanned PDFs require Tesseract OCR.")
        all_chunks.extend(doc_chunks)
        hashes[doc_name] = digest

    if not all_chunks:
        raise ValueError("No extractable text found in the uploaded PDF(s), even after OCR fallback.")

    texts = [c.text for c in all_chunks]
    embeddings = EmbeddingModel.get().embed(texts, is_query=False)

    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, all_chunks)
    store.document_hashes = hashes
    if persist:
        store.save()
    return store


def add_pdfs_to_existing_index(pdf_paths: List[str], store: VectorStore, persist: bool = True) -> VectorStore:
    """Incrementally add new PDFs to an already-built index (multi-PDF upload)."""
    if store.embedding_model != EMBEDDING_MODEL_NAME:
        raise ValueError("Embedding model changed; rebuild the index before adding PDFs")
    changed_paths = [p for p in pdf_paths if store.document_hashes.get(os.path.basename(p)) != hashlib.sha256(Path(p).read_bytes()).hexdigest()]
    if not changed_paths:
        return store
    # Prepare all replacements before touching the live store. Failed extraction,
    # embedding, or validation must leave the existing index intact.
    incoming = build_index_from_pdfs(changed_paths, persist=False)
    if incoming.index.d != store.index.d:
        raise ValueError("New embedding dimension differs from the existing index")
    replacements = set(incoming.list_documents())
    keep = [i for i, c in enumerate(store.chunks) if c.doc_name not in replacements]
    candidate = VectorStore(store.index.d)
    if keep:
        candidate.add(store.index.reconstruct_n(0, store.index.ntotal)[keep], [store.chunks[i] for i in keep])
    candidate.add(incoming.index.reconstruct_n(0, incoming.index.ntotal), incoming.chunks)
    candidate.document_hashes = {k: v for k, v in store.document_hashes.items() if k not in replacements}
    candidate.document_hashes.update(incoming.document_hashes)
    if persist:
        candidate.save()
    store.index, store.chunks = candidate.index, candidate.chunks
    store.document_hashes = candidate.document_hashes
    store._bm25 = None
    return store
