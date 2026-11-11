"""Run the extractor over a folder of emails and render a plain-text summary table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from llm_kit import LLMClient, LLMKitError

from order_extractor.extract import extract_order
from order_extractor.schema import Order

LOW_CONFIDENCE = 0.7

# (header, Order field used for the confidence lookup, max width)
COLUMNS = (
    ("FILE", None, 12),
    ("CUSTOMER", "customer_name", 24),
    ("ITEMS", "items", 30),
    ("DELIVERY", "delivery_date", 11),
    ("ADDRESS", "address", 30),
    ("CONF", None, 5),
    ("LOW-CONFIDENCE FIELDS", None, 40),
)


@dataclass(frozen=True)
class BatchRow:
    file: str
    order: Order | None = None
    error: str | None = None


def email_files(folder: Path) -> list[Path]:
    """Every .txt file directly inside `folder`, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def run_batch(folder: Path, client: LLMClient, *, today: date | None = None) -> list[BatchRow]:
    rows = []
    for path in email_files(folder):
        try:
            order = extract_order(read_email(path), client=client, today=today)
            rows.append(BatchRow(path.name, order=order))
        except LLMKitError as exc:  # keep going; the row shows what failed
            rows.append(BatchRow(path.name, error=f"{type(exc).__name__}: {exc}"))
    return rows


def read_email(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def low_confidence_fields(order: Order, threshold: float = LOW_CONFIDENCE) -> list[str]:
    """Fields whose confidence is below `threshold`. A missing confidence counts as low."""
    fields = [name for header, name, _ in COLUMNS if name]
    return [name for name in fields if order.confianca.get(name, 0.0) < threshold]


def _items_text(order: Order) -> str:
    parts = []
    for item in order.items:
        price = f" @{item.unit_price:g}" if item.unit_price is not None else ""
        parts.append(f"{item.quantity}x {item.name}{price}")
    return "; ".join(parts)


def _cell(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "~"


def _row_cells(row: BatchRow, threshold: float) -> list[str]:
    if row.order is None:
        return [row.file, "ERROR", "", "", "", "", _cell(row.error or "", 60)]
    order = row.order
    low = low_confidence_fields(order, threshold)
    values = {
        "customer_name": order.customer_name,
        "items": _items_text(order),
        "delivery_date": order.delivery_date.isoformat() if order.delivery_date else "-",
        "address": order.address or "-",
    }
    cells = [row.file]
    for _, field, width in COLUMNS[1:5]:
        mark = "!" if field in low else ""
        cells.append(mark + _cell(values[field], width - len(mark)))
    cells.append(f"{order.confianca_geral:.2f}")
    cells.append(", ".join(low) or "-")
    return cells


def format_table(rows: list[BatchRow], threshold: float = LOW_CONFIDENCE) -> str:
    """A fixed-width table. Values prefixed with '!' have confidence below `threshold`."""
    body = [_row_cells(row, threshold) for row in rows]
    headers = [header for header, _, _ in COLUMNS]
    widths = [max([len(h)] + [len(cells[i]) for cells in body]) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))

    out = [line(headers), line(["-" * w for w in widths])]
    out += [line(cells).rstrip() for cells in body]
    flagged = sum(1 for row in rows if row.order and low_confidence_fields(row.order, threshold))
    failed = sum(1 for row in rows if row.order is None)
    out.append("")
    out.append(
        f"{len(rows)} email(s), {flagged} with low-confidence fields (< {threshold:g}), "
        f"{failed} failed. '!' marks a value below the threshold."
    )
    return "\n".join(out)
