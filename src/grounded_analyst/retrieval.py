from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from grounded_analyst.models import EvidenceChunk, RetrievalResult


@dataclass
class RetrievalConfig:
    top_k: int = 10


class Retriever:
    """
    Lexical retrieval: TF-IDF cosine similarity plus small hybrid bonuses
    (overlap, numeric consistency, table structure) — no corpus-specific keywords.
    """

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            stop_words="english",
        )
        self._chunks: list[EvidenceChunk] = []
        self._matrix = None

    def build_index(self, chunks: list[EvidenceChunk]) -> None:
        self._chunks = chunks
        corpus = [c.text for c in chunks]
        self._matrix = self._vectorizer.fit_transform(corpus) if corpus else None

    def search(
        self,
        query: str,
        top_k: int | None = None,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        if self._matrix is None or not self._chunks:
            return []

        k = top_k or self.config.top_k
        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix).flatten()

        query_tokens = set(re.findall(r"[A-Za-z0-9\-]+", query.lower()))
        query_has_numeric = any(any(ch.isdigit() for ch in tok) for tok in query_tokens)

        scored: list[tuple[float, EvidenceChunk]] = []

        for i, tfidf_score in enumerate(sims):
            if tfidf_score <= 0:
                continue
            chunk = self._chunks[i]
            if source_types and chunk.source_type not in source_types:
                continue

            score = float(tfidf_score)
            text_lower = chunk.text.lower()
            chunk_tokens = set(re.findall(r"[A-Za-z0-9\-]+", text_lower))

            overlap = len(query_tokens.intersection(chunk_tokens))
            score += 0.03 * overlap

            for tok in query_tokens:
                if len(tok) > 4 and tok in chunk_tokens:
                    score += 0.02

            chunk_has_numeric = any(any(ch.isdigit() for ch in tok) for tok in chunk_tokens)
            if query_has_numeric and chunk_has_numeric:
                score += 0.05

            if query_has_numeric and chunk.source_type == "table":
                score += 0.05

            if "budget" in query_tokens:
                for tok in chunk_tokens:
                    if tok.isdigit() and len(tok) >= 5:
                        score += 0.15
                        break

            # Filename hint: questions often cite `something.csv` / doc names — prefer those chunks.
            ql = query.lower()
            stem = chunk.filename.rsplit(".", 1)[0].lower()
            if len(stem) >= 4 and stem.replace("_", "") in ql.replace("_", "").replace("-", ""):
                score += 0.2
            if stem in ql or chunk.filename.lower() in ql:
                score += 0.12
            if ("csv" in ql or "xlsx" in ql) and chunk.source_type == "table":
                score += 0.06

            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(chunk=chunk, score=score)
            for score, chunk in scored[:k]
        ]
