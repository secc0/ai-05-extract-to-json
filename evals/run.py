"""Run the extractor on the labelled examples with a real model and report accuracy.

The model comes from EVAL_MODEL, falling back to MODEL_PRIMARY. No fallback models
are used, so the numbers describe exactly one model.

    python -m evals.run            # all 20 emails
    python -m evals.run --smoke    # the 5-email CI subset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from llm_kit import (
    CallResult,
    ConfigError,
    JSONValidationError,
    LLMClient,
    LLMKitError,
    load_settings,
)

from evals.scoring import FIELDS, EmailScore, Summary, score_email, summarize
from order_extractor.extract import MAX_ATTEMPTS, extract_order
from order_extractor.schema import Order, OrderData

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = Path(__file__).with_name("cases.yaml")
EXAMPLES = ROOT / "examples"


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    score: EmailScore
    attempts: int
    order: Order | None
    error: str | None = None


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ref = data.get("reference_date")
    data["reference_date"] = ref if isinstance(ref, date) else date.fromisoformat(str(ref))
    data["cases"] = [str(c) for c in data.get("cases") or []]
    data["smoke"] = [str(c) for c in data.get("smoke") or []]
    unknown = set(data["smoke"]) - set(data["cases"])
    if unknown:
        raise ValueError(f"smoke ids not in cases: {sorted(unknown)}")
    return data


def load_case(case_id: str, examples: Path = EXAMPLES) -> tuple[str, OrderData]:
    email = (examples / f"{case_id}.txt").read_text(encoding="utf-8")
    expected_json = (examples / "expected" / f"{case_id}.json").read_text(encoding="utf-8")
    return email, OrderData.model_validate_json(expected_json)


def eval_client(**kwargs: Any) -> LLMClient:
    """An LLMClient on EVAL_MODEL (or MODEL_PRIMARY) with no fallbacks."""
    # The placeholder model only lets load_settings read .env and check the key;
    # the real model is chosen below, once .env has been loaded.
    base = load_settings(model_primary="placeholder", model_fallbacks=())
    model = os.getenv("EVAL_MODEL", "").strip() or os.getenv("MODEL_PRIMARY", "").strip()
    if not model:
        raise ConfigError("Missing required setting: EVAL_MODEL (or MODEL_PRIMARY).")
    return LLMClient(replace(base, model_primary=model), **kwargs)


def run_case(case_id: str, client: LLMClient, today: date, examples: Path = EXAMPLES) -> CaseResult:
    email, expected = load_case(case_id, examples)
    calls: list[CallResult] = []
    try:
        order = extract_order(email, client=client, today=today, on_call=calls.append)
    except JSONValidationError as exc:
        return CaseResult(case_id, score_email(expected, None), exc.attempts, None, str(exc))
    except LLMKitError as exc:
        error = f"{type(exc).__name__}: {exc}"
        return CaseResult(case_id, score_email(expected, None), max(len(calls), 1), None, error)
    return CaseResult(case_id, score_email(expected, order), len(calls), order)


def summarize_results(results: list[CaseResult]) -> Summary:
    return summarize(
        [r.score for r in results],
        [r.attempts for r in results],
        failed=sum(r.order is None for r in results),
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def format_report(model: str, results: list[CaseResult], summary: Summary) -> str:
    lines = [f"model: {model}  emails: {summary.n_emails}", ""]
    for r in results:
        status = "EXACT" if r.score.exact else ("ERROR" if r.error else "DIFF ")
        detail = r.error or ", ".join(r.score.wrong_fields)
        lines.append(f"{status}  {r.case_id}  attempts={r.attempts}  {detail}".rstrip())
    lines += ["", "per-field accuracy:"]
    for name in FIELDS:
        correct = sum(r.score.counts[name][0] for r in results)
        total = sum(r.score.counts[name][1] for r in results)
        lines.append(f"  {name:<16} {summary.field_accuracy[name]:6.1%}  ({correct}/{total})")
    exact = sum(r.score.exact for r in results)
    lines += [
        "",
        f"exact-match emails:      {summary.exact_match_rate:.1%} ({exact}/{summary.n_emails})",
        f"failed extractions:      {summary.n_failed}",
        f"mean attempts per email: {summary.mean_attempts:.2f} (max {MAX_ATTEMPTS})",
        f"mean confidence, correct fields:   {_fmt(summary.mean_conf_correct)}",
        f"mean confidence, incorrect fields: {_fmt(summary.mean_conf_incorrect)}",
    ]
    return "\n".join(lines)


def results_json(model: str, today: date, results: list[CaseResult], summary: Summary) -> str:
    payload = {
        "model": model,
        "reference_date": today.isoformat(),
        "summary": asdict(summary),
        "cases": [
            {
                "id": r.case_id,
                "attempts": r.attempts,
                "exact": r.score.exact,
                "wrong_fields": r.score.wrong_fields,
                "error": r.error,
                "order": r.order.model_dump(mode="json") if r.order else None,
            }
            for r in results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="python -m evals.run")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--smoke", action="store_true", help="only the CI smoke subset")
    parser.add_argument("--out", type=Path, help="also write detailed results as JSON")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.8,
        help="exit 1 if the mean per-field accuracy is below this (0-1)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.cases)
    try:
        client = eval_client()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ids = config["smoke"] if args.smoke else config["cases"]
    today = config["reference_date"]
    results = []
    for case_id in ids:
        result = run_case(case_id, client, today)
        print(f"  {case_id}: {'ok' if result.score.exact else 'diff'}", file=sys.stderr)
        results.append(result)

    summary = summarize_results(results)
    model = client.settings.model_primary
    print(format_report(model, results, summary))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(results_json(model, today, results, summary), encoding="utf-8")

    mean_accuracy = sum(summary.field_accuracy.values()) / len(FIELDS)
    return 0 if mean_accuracy >= args.min_accuracy else 1


if __name__ == "__main__":
    raise SystemExit(main())
