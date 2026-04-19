from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from grounded_analyst.pipeline import GroundedAnalystPipeline

_FALLBACK_PREFIX = "Grounded extractive fallback:"


def _answer_display_parts(raw: str) -> tuple[str, str | None]:
    """
    Match eval-style concise reading in the UI: one clear sentence up front,
    full text available when the model returned a long excerpt (e.g. extractive fallback).
    """
    raw = (raw or "").strip()
    if not raw:
        return "", None

    if raw.startswith(_FALLBACK_PREFIX):
        body = raw[len(_FALLBACK_PREFIX) :].strip()
        headline = _first_sentence_or_line(body)
        return headline, body

    if "\n" in raw and len(raw) > 360:
        first_block = raw.split("\n\n", 1)[0].strip()
        return first_block, raw

    parts = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)
    if len(parts) == 2 and len(raw) > 320:
        return parts[0].strip(), raw.strip()

    return raw, None


def _first_sentence_or_line(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    line = text.split("\n", 1)[0].strip()
    if "." in line:
        idx = line.find(".")
        if idx != -1 and idx < 400:
            return line[: idx + 1].strip()
    return line[:350] + ("..." if len(line) > 350 else "")


@st.cache_resource(show_spinner=False)
def load_pipeline(data_dir: str) -> tuple[GroundedAnalystPipeline, int]:
    pipeline = GroundedAnalystPipeline()
    chunk_count = pipeline.ingest(Path(data_dir))
    return pipeline, chunk_count


def main() -> None:
    st.set_page_config(page_title="Grounded Analyst", page_icon=":mag:", layout="centered")
    st.title("Grounded Multi-Modal Analyst")
    st.caption("Ask questions over your document + table corpus with grounded citations.")

    default_dir = "data/sample"
    data_dir = st.text_input("Data directory", value=default_dir)

    if not data_dir.strip():
        st.info("Enter a data directory to continue.")
        return

    dir_path = Path(data_dir)
    if not dir_path.exists():
        st.error(f"Directory not found: {dir_path}")
        return

    try:
        with st.spinner("Indexing data..."):
            pipeline, chunk_count = load_pipeline(str(dir_path))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to ingest data: {exc}")
        return

    st.success(f"Indexed {chunk_count} evidence chunks.")
    source_filter = st.multiselect(
        "Source filter",
        options=["document", "table"],
        default=["document", "table"],
        help="Optional metadata filter for retrieval.",
    )
    question = st.text_area(
        "Your question",
        placeholder="e.g. What was North region revenue in Q2-2025?",
        height=100,
    )

    if st.button("Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
            return
        with st.spinner("Generating grounded answer..."):
            source_types = set(source_filter) if source_filter else None
            answer = pipeline.ask(question.strip(), source_types=source_types)

        st.subheader("Answer")
        if answer.clarification_requested:
            st.info("Your question was too short or unclear to run retrieval. Add a clear question (e.g. what / which / how much).")
        headline, _detail = _answer_display_parts(answer.answer)
        full_text = answer.answer.strip()
        if headline:
            st.markdown(f"> {headline}")
        if _detail is not None or (headline and headline != full_text):
            with st.expander("Full response (same string as CLI / eval JSON)", expanded=False):
                st.text(answer.answer)

        st.subheader("Citations")
        if answer.citations:
            for citation in answer.citations:
                st.markdown(f"- `{citation}`")
        else:
            st.write("No citations returned.")


if __name__ == "__main__":
    main()
