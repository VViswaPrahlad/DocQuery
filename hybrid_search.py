"""
hybrid_search.py
-----------------
BM25 + FAISS hybrid retrieval via Reciprocal Rank Fusion (RRF).

Why hybrid: semantic search (FAISS over BGE embeddings) is strong at
paraphrase and conceptual matches, but can dilute exact tokens --
acronyms, part numbers, model names, PGNs -- into "just another word"
in vector space. BM25 is the opposite: exact/near-exact term overlap
weighted by term rarity (IDF), with zero semantic understanding.
Combining them catches both cases.

Why RRF instead of a weighted score average: BM25 scores are
unbounded and corpus-dependent; FAISS cosine similarity is bounded in
[-1, 1]. Averaging two different scales requires normalization
choices that are easy to get subtly wrong. RRF sidesteps this
entirely -- it only looks at each chunk's RANK within each list, which
is scale-free by construction.
"""

from typing import List, Tuple
import re
import math
import unicodedata
from collections import Counter
from rank_bm25 import BM25Okapi  # pip install rank-bm25


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


class BM25Index:
    """
    Thin wrapper so BM25 is built once and reused. Indices returned by
    .search() are positions into the SAME chunk list a VectorStore
    holds -- the two must stay in sync (see BM25Index.from_store).
    """

    def __init__(self, chunks):
        self.chunks = chunks
        corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus) if corpus and any(corpus) else None
        if self.bm25 is not None:
            # Positive-IDF BM25 avoids negative scores in tiny PDF collections.
            frequencies = Counter(token for tokens in corpus for token in set(tokens))
            n = len(corpus)
            self.bm25.idf = {token: math.log1p((n - count + 0.5) / (count + 0.5))
                             for token, count in frequencies.items()}

    @classmethod
    def from_store(cls, store):
        """
        Build a BM25Index over whatever chunks are currently in a
        VectorStore. Rebuilding is cheap (milliseconds for a few
        thousand chunks) so there's no need to persist this separately
        from the FAISS index -- just rebuild it after load() or after
        adding new documents.
        """
        return cls(store.chunks)

    def search(self, query: str, top_k: int = 10, allowed_indices=None) -> List[Tuple[int, float]]:
        """Returns [(chunk_index, bm25_score), ...] sorted best-first."""
        if self.bm25 is None or top_k <= 0 or not _tokenize(query):
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        candidates = range(len(scores)) if allowed_indices is None else allowed_indices
        ranked = sorted((i for i in candidates if scores[i] > 0), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked[:top_k]]


def reciprocal_rank_fusion(
    rank_lists: List[List[int]], k: int = 60
) -> List[Tuple[int, float]]:
    """
    rank_lists: one ranked list of chunk indices (best-first) per
    retrieval method -- e.g. one from FAISS, one from BM25.

    Returns [(chunk_index, fused_score), ...] sorted best-first.

    Fused score for a chunk = sum, over every list it appears in, of
    1 / (k + rank_in_that_list + 1). A chunk missing from a list
    contributes 0 for that list -- so a chunk found by BOTH methods
    naturally outranks one found by only one, without any manual
    weighting. k=60 is the standard damping constant from the original
    RRF paper: large enough that rank 1 vs rank 2 within a single list
    doesn't swing the fused result too aggressively.
    """
    if k < 0:
        raise ValueError("RRF k must be nonnegative")
    scores = {}
    for rank_list in rank_lists:
        for rank, idx in enumerate(dict.fromkeys(rank_list)):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
