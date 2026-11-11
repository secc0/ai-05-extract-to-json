import json

from order_extractor import cli
from order_extractor.batch import BatchRow, format_table, low_confidence_fields, run_batch
from order_extractor.schema import Order
from tests.conftest import fake_client, order_json

LOW = {"customer_name": 0.95, "items": 0.5, "delivery_date": 0.69, "address": 0.9}


def test_low_confidence_fields_uses_threshold_and_missing_keys():
    order = Order.model_validate_json(order_json(confianca=LOW))
    assert low_confidence_fields(order) == ["items", "delivery_date"]
    assert low_confidence_fields(order, threshold=0.4) == []
    missing = Order.model_validate_json(order_json(confianca={"customer_name": 0.9}))
    assert low_confidence_fields(missing) == ["items", "delivery_date", "address"]


def test_format_table_flags_low_confidence_values():
    flagged = Order.model_validate_json(order_json(confianca=LOW))
    fine = Order.model_validate_json(order_json())
    rows = [
        BatchRow("01.txt", order=flagged),
        BatchRow("02.txt", order=fine),
        BatchRow("03.txt", error="JSONValidationError: no valid Order"),
    ]
    table = format_table(rows)
    lines = table.splitlines()
    row1 = next(line for line in lines if line.startswith("01.txt"))
    row2 = next(line for line in lines if line.startswith("02.txt"))
    assert "!12x pão de queijo @1.5" in row1
    assert "!2026-11-13" in row1
    assert row1.rstrip().endswith("items, delivery_date")
    assert "!" not in row2
    assert "ERROR" in next(line for line in lines if line.startswith("03.txt"))
    assert "3 email(s), 1 with low-confidence fields (< 0.7), 1 failed" in table


def test_run_batch_processes_only_txt_files_in_order(tmp_path):
    (tmp_path / "b.txt").write_text("pedido b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("pedido a", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignore me", encoding="utf-8")
    client, sdk = fake_client(order_json(customer_name="A"), "x", "y", "z")
    rows = run_batch(tmp_path, client)
    assert [row.file for row in rows] == ["a.txt", "b.txt"]
    assert rows[0].order.customer_name == "A"
    assert rows[1].order is None and "JSONValidationError" in rows[1].error
    assert "pedido a" in sdk.calls[0]["messages"][0]["content"]


def test_cli_prints_validated_json(tmp_path, capsys):
    email = tmp_path / "pedido.txt"
    email.write_text("Quero pão de queijo pra sexta. Ana", encoding="utf-8")
    client, sdk = fake_client(order_json())
    assert cli.main([str(email), "--today", "2026-11-11"], client=client) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["customer_name"] == "Ana Souza"
    assert out["delivery_date"] == "2026-11-13"
    assert "2026-11-11" in sdk.calls[0]["messages"][0]["content"]


def test_cli_batch_prints_table(tmp_path, capsys):
    (tmp_path / "01.txt").write_text("pedido", encoding="utf-8")
    client, _ = fake_client(order_json(confianca=LOW))
    assert cli.main([str(tmp_path), "--batch"], client=client) == 0
    out = capsys.readouterr().out
    assert "LOW-CONFIDENCE FIELDS" in out and "items, delivery_date" in out


def test_cli_returns_1_when_extraction_fails(tmp_path, capsys):
    email = tmp_path / "pedido.txt"
    email.write_text("pedido", encoding="utf-8")
    client, _ = fake_client("a", "b", "c")
    assert cli.main([str(email)], client=client) == 1
    assert "JSONValidationError" in capsys.readouterr().err


def test_cli_fails_early_without_key(tmp_path, capsys):
    email = tmp_path / "pedido.txt"
    email.write_text("pedido", encoding="utf-8")
    assert cli.main([str(email)]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_cli_rejects_missing_file_and_wrong_mode(tmp_path, capsys):
    assert cli.main([str(tmp_path / "nope.txt")]) == 2
    assert cli.main([str(tmp_path)]) == 2  # a folder without --batch
    email = tmp_path / "pedido.txt"
    email.write_text("pedido", encoding="utf-8")
    assert cli.main([str(email), "--batch"]) == 2
    assert capsys.readouterr().err.count("error:") == 3
