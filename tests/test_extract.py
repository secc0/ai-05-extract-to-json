from datetime import date

import pytest
from llm_kit import JSONValidationError

from order_extractor.extract import build_prompt, extract_order
from tests.conftest import fake_client, order_json

EMAIL = "Oi, quero 12 pães de queijo pra sexta. Ana Souza"


def test_prompt_contains_today_weekday_and_email():
    prompt = build_prompt(EMAIL, date(2026, 11, 11))
    assert "2026-11-11" in prompt
    assert "quarta-feira" in prompt
    assert EMAIL in prompt
    assert "null" in prompt
    assert "confianca" in prompt


def test_extract_defaults_to_the_real_today():
    client, sdk = fake_client(order_json())
    extract_order(EMAIL, client=client)
    assert date.today().isoformat() in sdk.calls[0]["messages"][0]["content"]


def test_extract_valid_reply_in_one_call():
    client, sdk = fake_client(order_json())
    seen = []
    order = extract_order(EMAIL, client=client, today=date(2026, 11, 11), on_call=seen.append)
    assert order.customer_name == "Ana Souza"
    assert order.delivery_date == date(2026, 11, 13)
    assert len(sdk.calls) == 1 and len(seen) == 1
    # llm-kit appends the JSON Schema instruction to our prompt.
    assert "JSON Schema" in sdk.calls[0]["messages"][0]["content"]


def test_extract_retries_after_invalid_reply():
    invalid = order_json(items=[{"name": "pão de queijo", "quantity": 0}])
    client, sdk = fake_client("not json at all", invalid, order_json())
    order = extract_order(EMAIL, client=client, today=date(2026, 11, 11))
    assert order.items[0].quantity == 12
    assert len(sdk.calls) == 3
    feedback = sdk.calls[2]["messages"][-1]["content"]
    assert "not valid" in feedback and "quantity" in feedback


def test_extract_gives_up_after_three_invalid_replies():
    client, sdk = fake_client("nope", "{}", "still nope", order_json())
    with pytest.raises(JSONValidationError) as info:
        extract_order(EMAIL, client=client, today=date(2026, 11, 11))
    assert info.value.attempts == 3
    assert len(sdk.calls) == 3
