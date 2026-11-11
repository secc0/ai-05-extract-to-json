import json

import pytest

from evals import run as evals_run
from evals.scoring import (
    align_items,
    normalize,
    price_matches,
    score_email,
    similarity,
    summarize,
    text_matches,
)
from order_extractor.schema import Item, Order, OrderData
from tests.conftest import fake_client, order_json


def expected(**overrides):
    base = json.loads(order_json())
    base.pop("confianca")
    base.pop("confianca_geral")
    base.update(overrides)
    return OrderData.model_validate(base)


def got(**overrides):
    return Order.model_validate_json(order_json(**overrides))


def test_normalize_drops_accents_case_and_punctuation():
    assert normalize("  Pão-de-Queijo, Açúcar! ") == "pao de queijo acucar"


@pytest.mark.parametrize(
    ("a", "b", "threshold", "match"),
    [
        ("Maria Eduarda Lins", "Duda (Maria Eduarda Lins)", 0.85, True),
        ("Marcelo Antunes", "Marcelo", 0.85, False),
        ("rua das palmeras 23", "Rua das Palmeiras, 23", 0.8, True),
        ("Rua Harmonia, 300", "Av. Paulista, 2200", 0.8, False),
        (None, None, 0.8, True),
        ("Rua A, 1", None, 0.8, False),
        (None, "Rua A, 1", 0.8, False),
    ],
)
def test_text_matches(a, b, threshold, match):
    assert text_matches(a, b, threshold) is match


def test_similarity_is_one_for_equal_text():
    assert similarity("Bolo de Cenoura", "bolo de cenoura") == 1.0


def test_price_matches():
    assert price_matches(None, None)
    assert price_matches(4.5, 4.50001)
    assert not price_matches(4.5, None)
    assert not price_matches(4.5, 5.5)


def test_align_items_pairs_by_name_one_to_one():
    exp = [Item(name="cupcake de baunilha", quantity=12), Item(name="red velvet", quantity=12)]
    out = [Item(name="Cupcakes red velvet", quantity=12), Item(name="cupcake baunilha", quantity=1)]
    assert align_items(exp, out) == [(0, 1), (1, 0)]
    assert align_items(exp, [Item(name="refrigerante", quantity=1)]) == []


def test_score_email_exact():
    score = score_email(expected(), got())
    assert score.exact
    assert score.wrong_fields == []
    assert all(ok for _, _, ok in score.confidence)


def test_score_email_counts_each_wrong_field():
    out = got(
        customer_name="Ana",
        delivery_date="2026-11-14",
        items=[
            {"name": "pão de queijo", "quantity": 6, "unit_price": None},
            {"name": "refrigerante", "quantity": 1},
        ],
    )
    score = score_email(expected(customer_name="Ana Souza"), out)
    assert score.counts["customer_name"] == (0, 1)
    assert score.counts["delivery_date"] == (0, 1)
    assert score.counts["address"] == (1, 1)
    assert score.counts["item_name"] == (1, 2)  # one extra item
    assert score.counts["item_quantity"] == (0, 1)
    assert score.counts["item_unit_price"] == (0, 1)
    assert not score.exact
    confidence = {key: ok for key, _, ok in score.confidence}
    assert confidence == {
        "customer_name": False,
        "items": False,
        "delivery_date": False,
        "address": True,
    }


def test_score_email_null_must_stay_null():
    score = score_email(expected(address=None), got(address="Rua Inventada, 1"))
    assert score.counts["address"] == (0, 1)


def test_score_email_failed_extraction_scores_zero():
    score = score_email(expected(), None)
    assert score.counts["item_name"] == (0, 1)
    assert not score.exact and score.confidence == []


def test_summarize_micro_averages_and_splits_confidence():
    perfect = score_email(expected(), got())
    wrong = score_email(
        expected(), got(customer_name="Outra Pessoa", confianca={"customer_name": 0.4})
    )
    summary = summarize([perfect, wrong], attempts=[1, 2])
    assert summary.field_accuracy["customer_name"] == 0.5
    assert summary.field_accuracy["address"] == 1.0
    assert summary.exact_match_rate == 0.5
    assert summary.mean_attempts == 1.5
    assert summary.mean_conf_incorrect == pytest.approx(0.4)
    assert summary.mean_conf_correct == pytest.approx((0.95 + 0.9 + 0.8 + 0.9) / 4)


def test_bundled_config_and_cases_load():
    config = evals_run.load_config(evals_run.DEFAULT_CASES)
    assert len(config["cases"]) == 20
    assert len(config["smoke"]) == 5
    assert config["reference_date"].isoformat() == "2026-11-11"
    for case_id in config["cases"]:
        evals_run.load_case(case_id)


def test_run_case_with_fake_client_counts_attempts():
    config = evals_run.load_config(evals_run.DEFAULT_CASES)
    _, exp = evals_run.load_case("01")
    reply = exp.model_dump(mode="json") | {"confianca": {"items": 0.9}, "confianca_geral": 0.9}
    client, _ = fake_client("garbage", json.dumps(reply, ensure_ascii=False))
    result = evals_run.run_case("01", client, config["reference_date"])
    assert result.score.exact
    assert result.attempts == 2

    client, _ = fake_client("a", "b", "c")
    failed = evals_run.run_case("01", client, config["reference_date"])
    assert failed.order is None and failed.attempts == 3
    summary = evals_run.summarize_results([result, failed])
    assert summary.n_failed == 1
    report = evals_run.format_report("fake/model", [result, failed], summary)
    assert "exact-match emails:      50.0% (1/2)" in report


def test_eval_model_falls_back_to_primary(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("MODEL_PRIMARY", "p/primary")
    assert evals_run.eval_client(enable_log=False).settings.model_primary == "p/primary"
    monkeypatch.setenv("EVAL_MODEL", "p/eval")
    client = evals_run.eval_client(enable_log=False)
    assert client.settings.model_primary == "p/eval"
    assert client.settings.model_fallbacks == ()


def test_evals_main_fails_clearly_without_key(capsys):
    assert evals_run.main(["--smoke"]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
