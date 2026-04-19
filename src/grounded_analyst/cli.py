from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounded_analyst.evaluator import load_eval_set, run_eval
from grounded_analyst.logging_utils import setup_logging
from grounded_analyst.pipeline import GroundedAnalystPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounded-analyst",
        description="Grounded QA over mixed docs and tables.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask a single question.")
    ask.add_argument("--data-dir", type=Path, required=True, help="Directory with source files.")
    ask.add_argument("--question", type=str, required=True, help="Natural language question.")
    ask.add_argument(
        "--source-types",
        type=str,
        default="all",
        help="Comma-separated source filter: all,document,table",
    )

    ev = sub.add_parser("evaluate", help="Run evaluation set.")
    ev.add_argument("--data-dir", type=Path, required=True, help="Directory with source files.")
    ev.add_argument("--eval-file", type=Path, required=True, help="JSON file of eval questions.")
    ev.add_argument("--out", type=Path, default=Path("eval/results.json"), help="Output report path.")
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    pipeline = GroundedAnalystPipeline()
    chunk_count = pipeline.ingest(args.data_dir)

    if args.command == "ask":
        source_types = None
        if args.source_types.lower() != "all":
            source_types = {s.strip() for s in args.source_types.split(",") if s.strip()}
        answer = pipeline.ask(args.question, source_types=source_types)
        print(f"Ingested chunks: {chunk_count}\n")
        print(f"Q: {args.question}\n")
        print(f"A: {answer.answer}\n")
        if answer.clarification_requested:
            print("(Clarification requested — question was too vague for retrieval.)\n")
        if answer.citations:
            print("Citations:")
            for c in answer.citations:
                print(f"- {c}")
        else:
            print("Citations: none")
        return

    eval_set = load_eval_set(args.eval_file)
    report = run_eval(pipeline, eval_set)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Ingested chunks: {chunk_count}")
    print(f"Eval: {report['passed']}/{report['total']} passed")
    print(f"Saved report: {args.out}")


if __name__ == "__main__":
    main()
