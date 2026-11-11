import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from order_extractor.schema import Item, Order, OrderData
from tests.conftest import order_json

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_valid_order_parses_dates_and_confidence():
    order = Order.model_validate_json(order_json())
    assert order.delivery_date == date(2026, 11, 13)
    assert order.items[0].quantity == 12
    assert order.confianca["items"] == 0.9


def test_optional_fields_default_to_none():
    order = Order.model_validate_json(
        order_json(delivery_date=None, address=None, items=[{"name": "bolo", "quantity": 1}])
    )
    assert order.delivery_date is None
    assert order.address is None
    assert order.items[0].unit_price is None


@pytest.mark.parametrize(
    "item",
    [
        {"name": "bolo", "quantity": 0},
        {"name": "bolo", "quantity": -2},
        {"name": "bolo", "quantity": 1.5},
        {"name": "bolo", "quantity": 1, "unit_price": -1},
        {"name": "   ", "quantity": 1},
        {"name": "bolo", "quantity": 1, "color": "blue"},
    ],
)
def test_invalid_items_are_rejected(item):
    with pytest.raises(ValidationError):
        Item.model_validate(item)


def test_item_name_is_stripped():
    assert Item(name="  bolo ", quantity=1).name == "bolo"


@pytest.mark.parametrize(
    "overrides",
    [
        {"items": []},
        {"delivery_date": "sexta"},
        {"confianca_geral": 1.2},
        {"confianca": {"items": -0.1}},
        {"confianca": {"phone": 0.5}},
    ],
)
def test_invalid_orders_are_rejected(overrides):
    with pytest.raises(ValidationError):
        Order.model_validate_json(order_json(**overrides))


def test_order_requires_confidence():
    raw = json.loads(order_json())
    del raw["confianca"]
    with pytest.raises(ValidationError):
        Order.model_validate(raw)


def test_data_drops_confidence():
    data = Order.model_validate_json(order_json()).data()
    assert isinstance(data, OrderData)
    assert "confianca" not in data.model_dump()


def test_twenty_examples_each_have_a_valid_expected_file():
    emails = sorted(EXAMPLES.glob("*.txt"))
    assert len(emails) == 20
    for email in emails:
        expected = EXAMPLES / "expected" / f"{email.stem}.json"
        OrderData.model_validate_json(expected.read_text(encoding="utf-8"))
