from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib import error as urlerror
from urllib import request as urlrequest

from dotenv import load_dotenv

from grounded_analyst.models import GroundedAnswer, RetrievalResult


@dataclass
class GroundingConfig:
    model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434/api/generate"
    min_retrieval_score: float = 0.08
    max_evidence: int = 10


SYSTEM_PROMPT = """You are a grounded analyst.

Answer ONLY using the provided evidence.

Instructions:
- Answer in 1 clear sentence.
- Prefer exact values from evidence.
- You may interpret simple wording changes (e.g., 'initiated' = 'launched').
- When answering dates or months, use exact format from evidence (e.g., 2025-06).
- If a rule is stated, extract the correct condition and action.
- When the question asks for a specific thing (e.g. which, who, what name, what ID, what amount, what region), you must **name that value explicitly** as it appears in the evidence—do not answer with a vague restatement of the paragraph.
- When evidence lists multiple items (bullets, commas, rows) and the question singles out one relationship (e.g. “along with X”), identify the **other** matching item(s) and state them by name.
- Do NOT guess or use outside knowledge.

Rules:
- If the answer is not clearly supported, say exactly: Insufficient evidence in the provided corpus.
- Include citations in [filename | locator] format.
"""


class GroundedGenerator:
    def __init__(self, config: GroundingConfig | None = None) -> None:
        load_dotenv()
        self.config = config or GroundingConfig()
        self.model = os.getenv("OLLAMA_MODEL", self.config.model)
        self.ollama_url = os.getenv("OLLAMA_URL", self.config.ollama_url)

    def answer(self, question: str, results: list[RetrievalResult]) -> GroundedAnswer:
        filtered = [r for r in results if r.score >= self.config.min_retrieval_score]
        if not filtered and results:
            filtered = results[: self.config.max_evidence]
        evidence = [r.chunk for r in filtered[: self.config.max_evidence]]
        if not evidence:
            return GroundedAnswer(
                answer="Insufficient evidence in the provided corpus.",
                citations=[],
                used_evidence=[],
                insufficient_evidence=True,
            )

        if _requests_location_or_contact_fact(question, evidence):
            return GroundedAnswer(
                answer="Insufficient evidence in the provided corpus.",
                citations=[],
                used_evidence=[],
                insufficient_evidence=True,
            )

        evidence_block = "\n\n".join(
            f"[{idx+1}] {c.filename} | {c.locator}\n{c.text}" for idx, c in enumerate(evidence)
        )
        user_prompt = f"Question: {question}\n\nEvidence:\n{evidence_block}"
        answer = self._query_ollama(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        if not answer:
            return self._extractive_fallback(question, evidence, filtered)

        insufficient = answer.strip() == "Insufficient evidence in the provided corpus."
        if insufficient:
            fb = self._extractive_fallback(question, evidence, filtered)
            if not fb.insufficient_evidence:
                return fb
            return GroundedAnswer(
                answer="Insufficient evidence in the provided corpus.",
                citations=[],
                used_evidence=[],
                insufficient_evidence=True,
            )

        cited_chunks = _select_citation_chunks(question, answer, filtered)
        citations = [f"{c.filename} ({c.locator})" for c in cited_chunks]
        return GroundedAnswer(
            answer=answer.strip(),
            citations=citations if not insufficient else [],
            used_evidence=cited_chunks if not insufficient else [],
            insufficient_evidence=insufficient,
        )

    def _extractive_fallback(
        self,
        question: str,
        evidence: list,
        filtered: list[RetrievalResult],
    ) -> GroundedAnswer:
        keyword_hits: list[tuple[float, int]] = []
        q_tokens = _tokenize(question)
        q_token_set = {t for t in q_tokens if len(t) > 2}
        digit_constraints = {t for t in q_tokens if any(ch.isdigit() for ch in t)}
        constraint_tokens = _constraint_tokens(question)
        for idx, chunk in enumerate(evidence):
            c_tokens = set(_tokenize(chunk.text))
            if not c_tokens:
                keyword_hits.append((0.0, idx))
                continue
            if digit_constraints and not digit_constraints.issubset(c_tokens):
                continue
            overlap = q_token_set.intersection(c_tokens)
            base_score = len(overlap) / max(len(q_token_set), 1)
            has_number = any(any(ch.isdigit() for ch in tok) for tok in overlap)
            number_bonus = 0.25 if has_number else 0.0
            keyword_hits.append((base_score + number_bonus, idx))

        if not keyword_hits:
            return GroundedAnswer(
                answer="Insufficient evidence in the provided corpus.",
                citations=[],
                used_evidence=[],
                insufficient_evidence=True,
            )
        best = max(keyword_hits, key=lambda x: x[0])
        if best[0] < 0.15:
            return GroundedAnswer(
                answer="Insufficient evidence in the provided corpus.",
                citations=[],
                used_evidence=[],
                insufficient_evidence=True,
            )
        best_chunk = evidence[best[1]]
        best_chunk_tokens = set(_tokenize(best_chunk.text))
        if constraint_tokens and not constraint_tokens.issubset(best_chunk_tokens):
            return GroundedAnswer(
                answer="Insufficient evidence in the provided corpus.",
                citations=[],
                used_evidence=[],
                insufficient_evidence=True,
            )
        text = (
            "Grounded extractive fallback: "
            f"{best_chunk.text[:1200].strip()}"
        )
        citations = [f"{best_chunk.filename} ({best_chunk.locator})"]
        return GroundedAnswer(
            answer=text,
            citations=citations,
            used_evidence=[best_chunk],
            insufficient_evidence=False,
        )

    def _query_ollama(self, system_prompt: str, user_prompt: str) -> str | None:
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {"temperature": 0},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            self.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                response_text = (parsed.get("response") or "").strip()
                return response_text or None
        except (urlerror.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9\-]+", text.lower())


def _constraint_tokens(question: str) -> set[str]:
    tokens: set[str] = set()
    for tok in _tokenize(question.lower()):
        if any(ch.isdigit() for ch in tok):
            tokens.add(tok)
    for tok in _tokenize(question.lower()):
        if any(ch.isdigit() for ch in tok):
            date_prefix = re.match(r"^(\d{4}-\d{2})[a-z]+$", tok)
            quarter_prefix = re.match(r"^(q\d-\d{4})[a-z]+$", tok)
            if date_prefix:
                tokens.add(date_prefix.group(1))
            elif quarter_prefix:
                tokens.add(quarter_prefix.group(1))
    return tokens


def _requests_location_or_contact_fact(question: str, evidence: list) -> bool:
    """Generic decline when the question asks for postal/HQ/contact facts absent from evidence."""
    qlow = question.lower()
    asks_place = any(
        w in qlow
        for w in ("headquarters", "hq ", " hq", "mailing address", "street address", "office address")
    )
    asks_contact = any(w in qlow for w in ("phone number", "fax ", "email ", "contact email"))
    if not asks_place and not asks_contact:
        return False
    blob = "\n".join(c.text.lower() for c in evidence)
    if asks_place:
        if re.search(r"\b(street|st\.|avenue|ave\.|road|rd\.|boulevard|blvd|suite\s*\d|zip\s*code|postal)\b", blob):
            return False
        if re.search(r"\d{3}[-.) ]?\d{3}[-.]?\d{4}\b", blob):
            return False
        return True
    if asks_contact:
        if "@" in blob or re.search(r"\b(tel|phone|fax)\b", blob):
            return False
        return True
    return False


def _select_citation_chunks(
    question: str, answer: str, retrieved: list[RetrievalResult]
) -> list:
    if not retrieved:
        return []

    top_score = retrieved[0].score
    answer_tokens = {t for t in _tokenize(answer) if len(t) > 3}
    question_tokens = {t for t in _tokenize(question) if len(t) > 3}
    selected = []

    for result in retrieved:
        if result.score < max(0.1, top_score * 0.6):
            continue
        chunk_tokens = set(_tokenize(result.chunk.text))
        answer_overlap = len(answer_tokens.intersection(chunk_tokens))
        question_overlap = len(question_tokens.intersection(chunk_tokens))
        if answer_overlap >= 2 or question_overlap >= 2:
            selected.append(result.chunk)
        if len(selected) >= 2:
            break

    if not selected:
        selected = [retrieved[0].chunk]
    return selected
