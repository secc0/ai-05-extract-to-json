"""Pure scoring functions: compare an extracted order with the hand-labelled one.

Text fields are compared leniently (accents, case, punctuation and small typos are
ignored) because the model may tidy "rua das palmeras" into "Rua das Palmeiras".
Numbers and dates must match exactly.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from order_extractor.schema import Item, Order, OrderData

FIELDS = (
    "customer_name",
    "item_name",
    "item_quantity",
    "item_unit_price",
    "delivery_date",
    "address",
)
# Order.confianca key that covers each scored field.
CONFIDENCE_KEY = {
    "customer_name": "customer_name",
    "items": "items",
    "delivery_date": "delivery_date",
    "address": "address",
}
STOPWORDS = {"a", "o", "e", "de", "da", "do", "das", "dos", "com", "para", "pra", "em", "no", "na"}

NAME_THRESHOLD = 0.85
ITEM_THRESHOLD = 0.6
ADDRESS_THRESHOLD = 0.8
TOKEN_THRESHOLD = 0.8


def normalize(text: str) -> str:
    """Lowercase, drop accents and punctuation, collapse spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", plain).split())


def _tokens(text: str) -> list[str]:
    return [t for t in normalize(text).split() if t not in STOPWORDS]


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def similarity(expected: str, got: str) -> float:
    """How well `got` covers `expected`, from 0 to 1.

    The max of whole-string similarity and token recall: the share of expected
    words found in `got` (a word counts if some word in `got` is >= 0.8 similar).
    Recall makes "Duda (Maria Eduarda Lins)" match "Maria Eduarda Lins".
    """
    a, b = normalize(expected), normalize(got)
    if a == b:
        return 1.0
    whole = _ratio(a, b)
    exp_tokens, got_tokens = _tokens(expected), _tokens(got)
    if not exp_tokens or not got_tokens:
        return whole
    found = sum(1 for t in exp_tokens if max(_ratio(t, g) for g in got_tokens) >= TOKEN_THRESHOLD)
    return max(whole, found / len(exp_tokens))


def text_matches(expected: str | None, got: str | None, threshold: float) -> bool:
    if expected is None or got is None:
        return expected is None and got is None
    return similarity(expected, got) >= threshold


def price_matches(expected: float | None, got: float | None) -> bool:
    if expected is None or got is None:
        return expected is None and got is None
    return abs(expected - got) < 0.005


def align_items(expected: list[Item], got: list[Item]) -> list[tuple[int, int]]:
    """Pair expected and extracted items by name, best matches first, one-to-one."""
    candidates = sorted(
        (
            (similarity(e.name, g.name), i, j)
            for i, e in enumerate(expected)
            for j, g in enumerate(got)
        ),
        reverse=True,
    )
    used_e: set[int] = set()
    used_g: set[int] = set()
    pairs = []
    for score, i, j in candidates:
        if score < ITEM_THRESHOLD or i in used_e or j in used_g:
            continue
        used_e.add(i)
        used_g.add(j)
        pairs.append((i, j))
    return sorted(pairs)


@dataclass
class EmailScore:
    """Per-field (correct, total) counts for one email, plus confidence pairs."""

    counts: dict[str, tuple[int, int]]
    # (Order.confianca key, confidence reported, was that field correct)
    confidence: list[tuple[str, float, bool]] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return all(correct == total for correct, total in self.counts.values())

    @property
    def wrong_fields(self) -> list[str]:
        return [name for name, (correct, total) in self.counts.items() if correct != total]


def score_items(expected: list[Item], got: list[Item]) -> dict[str, tuple[int, int]]:
    """Item-level counts.

    item_name: aligned items over max(expected, extracted), so extra or missing items
    both cost. item_quantity / item_unit_price: aligned items with the right value
    over the number of expected items.
    """
    pairs = align_items(expected, got)
    qty = sum(1 for i, j in pairs if expected[i].quantity == got[j].quantity)
    price = sum(1 for i, j in pairs if price_matches(expected[i].unit_price, got[j].unit_price))
    n = len(expected)
    return {
        "item_name": (len(pairs), max(n, len(got))),
        "item_quantity": (qty, n),
        "item_unit_price": (price, n),
    }


def score_email(expected: OrderData, got: Order | None) -> EmailScore:
    """Score one extraction. `got=None` (the extraction failed) scores zero everywhere."""
    if got is None:
        n = len(expected.items)
        zero = {"customer_name": (0, 1), "delivery_date": (0, 1), "address": (0, 1)}
        zero.update({"item_name": (0, n), "item_quantity": (0, n), "item_unit_price": (0, n)})
        return EmailScore({name: zero[name] for name in FIELDS})

    item_counts = score_items(expected.items, got.items)
    single = {
        "customer_name": text_matches(expected.customer_name, got.customer_name, NAME_THRESHOLD),
        "delivery_date": expected.delivery_date == got.delivery_date,
        "address": text_matches(expected.address, got.address, ADDRESS_THRESHOLD),
    }
    counts = {name: (int(ok), 1) for name, ok in single.items()} | item_counts
    items_ok = all(c == t for c, t in item_counts.values())

    correctness = {**single, "items": items_ok}
    confidence = [
        (key, got.confianca[key], correctness[key])
        for key in CONFIDENCE_KEY
        if key in got.confianca
    ]
    return EmailScore({name: counts[name] for name in FIELDS}, confidence)


@dataclass(frozen=True)
class Summary:
    field_accuracy: dict[str, float]
    exact_match_rate: float
    mean_attempts: float
    mean_conf_correct: float | None
    mean_conf_incorrect: float | None
    n_emails: int
    n_failed: int


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(scores: list[EmailScore], attempts: list[int], failed: int = 0) -> Summary:
    """Micro-averaged accuracy per field, exact-match rate, attempts and confidence."""
    accuracy = {}
    for name in FIELDS:
        correct = sum(s.counts[name][0] for s in scores)
        total = sum(s.counts[name][1] for s in scores)
        accuracy[name] = correct / total if total else 0.0
    pairs = [pair for s in scores for pair in s.confidence]
    return Summary(
        field_accuracy=accuracy,
        exact_match_rate=sum(s.exact for s in scores) / len(scores) if scores else 0.0,
        mean_attempts=_mean([float(a) for a in attempts]) or 0.0,
        mean_conf_correct=_mean([conf for _, conf, ok in pairs if ok]),
        mean_conf_incorrect=_mean([conf for _, conf, ok in pairs if not ok]),
        n_emails=len(scores),
        n_failed=failed,
    )
