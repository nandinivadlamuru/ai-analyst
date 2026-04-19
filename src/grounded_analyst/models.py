from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceChunk:
    chunk_id: str
    source_type: str
    filename: str
    locator: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk: EvidenceChunk
    score: float


@dataclass
class GroundedAnswer:
    answer: str
    citations: list[str]
    used_evidence: list[EvidenceChunk]
    insufficient_evidence: bool
    clarification_requested: bool = False
