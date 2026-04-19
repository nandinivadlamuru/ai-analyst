from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from docx import Document as DocxDocument
from pypdf import PdfReader

from grounded_analyst.models import EvidenceChunk


class IngestionPipeline:
    """Parse documents and tabular files into evidence chunks."""

    supported_doc_exts = {".pdf", ".docx", ".md", ".txt"}
    supported_table_exts = {".csv", ".xlsx", ".xls"}

    def ingest_paths(self, paths: Iterable[Path]) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        for path in paths:
            suffix = path.suffix.lower()
            if suffix in self.supported_doc_exts:
                chunks.extend(self._ingest_document(path))
            elif suffix in self.supported_table_exts:
                chunks.extend(self._ingest_table(path))
        return chunks

    def ingest_directory(self, directory: Path) -> list[EvidenceChunk]:
        files = [p for p in directory.rglob("*") if p.is_file()]
        return self.ingest_paths(files)

    def _ingest_document(self, path: Path) -> list[EvidenceChunk]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._ingest_pdf(path)
        if suffix == ".docx":
            return self._ingest_docx(path)
        return self._ingest_text(path)

    def _ingest_pdf(self, path: Path) -> list[EvidenceChunk]:
        reader = PdfReader(str(path))
        chunks: list[EvidenceChunk] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            for part_idx, part in enumerate(_split_text(text), start=1):
                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"{path.name}-p{idx}-c{part_idx}",
                        source_type="document",
                        filename=path.name,
                        locator=f"page {idx}",
                        text=part,
                    )
                )
        return chunks

    def _ingest_docx(self, path: Path) -> list[EvidenceChunk]:
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return self._build_text_chunks(path, text, "paragraph range")

    def _ingest_text(self, path: Path) -> list[EvidenceChunk]:
        text = path.read_text(encoding="utf-8")
        return self._build_text_chunks(path, text, "section")

    def _build_text_chunks(self, path: Path, text: str, label: str) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        for idx, part in enumerate(_split_text(text), start=1):
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{path.name}-c{idx}",
                    source_type="document",
                    filename=path.name,
                    locator=f"{label} {idx}",
                    text=part,
                )
            )
        return chunks

    def _ingest_table(self, path: Path) -> list[EvidenceChunk]:
        ext = path.suffix.lower()
        if ext == ".csv":
            dataframes = {"Sheet1": pd.read_csv(path)}
        else:
            dataframes = pd.read_excel(path, sheet_name=None)

        chunks: list[EvidenceChunk] = []
        for sheet_name, df in dataframes.items():
            headers = [str(c) for c in df.columns]
            for i, row in df.fillna("").iterrows():
                row_text_pairs = [f"{h}: {row[h]}" for h in df.columns]
                text = f"Sheet={sheet_name}; " + "; ".join(row_text_pairs)
                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"{path.name}-{sheet_name}-r{i+2}",
                        source_type="table",
                        filename=path.name,
                        locator=f"sheet {sheet_name}, row {i+2}",
                        text=text,
                        metadata={"headers": headers, "sheet": sheet_name, "row": int(i + 2)},
                    )
                )
        return chunks


def _split_text(text: str, max_chars: int = 1200) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue
        if size + len(para) > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para)
    if buf:
        chunks.append("\n".join(buf))
    return chunks
