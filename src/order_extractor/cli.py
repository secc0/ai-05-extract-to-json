"""Command line: `extract-order FILE` or `extract-order DIR --batch`."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from llm_kit import ConfigError, LLMClient, LLMKitError

from order_extractor.batch import LOW_CONFIDENCE, format_table, read_email, run_batch
from order_extractor.extract import extract_order


def _utf8_console() -> None:
    """Windows consoles may default to a legacy code page; never crash on 'ç' or 'ã'."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an ISO date (YYYY-MM-DD): {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-order",
        description="Extract a validated JSON order from a Portuguese order email.",
    )
    parser.add_argument("path", type=Path, help="an email .txt file, or a folder with --batch")
    parser.add_argument(
        "--batch", action="store_true", help="process every .txt in the folder and print a table"
    )
    parser.add_argument(
        "--today",
        type=_parse_date,
        default=None,
        help="reference date for 'amanhã', 'sexta', ... (default: today)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=LOW_CONFIDENCE,
        help=f"flag fields below this confidence in --batch (default: {LOW_CONFIDENCE})",
    )
    return parser


def main(argv: list[str] | None = None, *, client: LLMClient | None = None) -> int:
    _utf8_console()
    args = build_parser().parse_args(argv)

    if args.batch and not args.path.is_dir():
        print(f"error: --batch needs a folder: {args.path}", file=sys.stderr)
        return 2
    if not args.batch and not args.path.is_file():
        print(f"error: file not found: {args.path} (use --batch for a folder)", file=sys.stderr)
        return 2

    try:
        client = client or LLMClient()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.batch:
        rows = run_batch(args.path, client, today=args.today)
        print(format_table(rows, args.threshold))
        return 1 if any(row.order is None for row in rows) else 0

    try:
        order = extract_order(read_email(args.path), client=client, today=args.today)
    except LLMKitError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(order.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
