from __future__ import annotations

from pathlib import Path

from grounded_analyst.grounding import GroundedGenerator
from grounded_analyst.ingestion import IngestionPipeline
from grounded_analyst.logging_utils import get_logger, new_trace_id
from grounded_analyst.models import GroundedAnswer
from grounded_analyst.retrieval import Retriever
from grounded_analyst.vague_question import CLARIFICATION_MESSAGE, is_vague_question


class GroundedAnalystPipeline:
    def __init__(self) -> None:
        self.ingestion = IngestionPipeline()
        self.retriever = Retriever()
        self.generator = GroundedGenerator()
        self._ingested = False
        self.logger = get_logger("grounded_analyst.pipeline")

    def ingest(self, data_dir: Path) -> int:
        chunks = self.ingestion.ingest_directory(data_dir)
        self.retriever.build_index(chunks)
        self._ingested = True
        self.logger.info("Ingested %s chunks from %s", len(chunks), data_dir)
        return len(chunks)

    def ask(self, question: str, source_types: set[str] | None = None, trace_id: str | None = None) -> GroundedAnswer:
        if not self._ingested:
            raise RuntimeError("Pipeline has not ingested data yet.")
        trace = trace_id or new_trace_id()
        if is_vague_question(question):
            self.logger.info("trace=%s vague_question=True", trace)
            return GroundedAnswer(
                answer=CLARIFICATION_MESSAGE,
                citations=[],
                used_evidence=[],
                insufficient_evidence=False,
                clarification_requested=True,
            )
        retrieved = self.retriever.search(question, source_types=source_types)
        self.logger.info(
            "trace=%s retrieved=%s source_types=%s question=%s",
            trace,
            len(retrieved),
            sorted(source_types) if source_types else "all",
            question,
        )
        answer = self.generator.answer(question, retrieved)
        self.logger.info(
            "trace=%s insufficient=%s citations=%s",
            trace,
            answer.insufficient_evidence,
            len(answer.citations),
        )
        return answer
