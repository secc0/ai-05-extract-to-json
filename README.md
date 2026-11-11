# ai-05-extract-to-json

Turns free-text order emails (Portuguese) into JSON validated by Pydantic, retrying
automatically when the model returns invalid output.

Day 5 of my 56-day series of small AI projects on the OpenRouter API. It is built on
[llm-kit](https://github.com/secc0/ai-03-llm-kit) (day 3), which handles retries, fallbacks
and JSON validation.

## Problem

Small businesses receive orders as free-form emails: "oii, queria 2 pizzas de calabresa pra
amanhã...". A person has to retype each one into a system. The text is messy: typos,
abbreviations, relative dates ("sexta", "dia 5"), corrections in the middle of the email
("na verdade troca: são 5"), and fields that are simply missing.

This project turns each email into a typed `Order`, or fails loudly. Every reply is checked
against a Pydantic schema. When a reply doesn't validate, the validation error goes back to
the model, which gets up to three attempts. The model also reports its confidence per field,
so low-confidence values can be routed to a human.

## Demo

A real run (`google/gemma-4-31b-it:free`):

```text
> type examples\02.txt
oii tudo bem??
queria pedir 2 pizzas de calabresa (R$ 42 cada) e 1 de quatro queijos que é R$ 48
pode mandar amanhã pra Av. Brasil 1500, casa 2, Centro, Campinas
valeu!! bjs
Duda (Maria Eduarda Lins)

> extract-order examples\02.txt --today 2026-11-11
{
  "customer_name": "Maria Eduarda Lins",
  "items": [
    {"name": "pizzas de calabresa", "quantity": 2, "unit_price": 42.0},
    {"name": "de quatro queijos", "quantity": 1, "unit_price": 48.0}
  ],
  "delivery_date": "2026-11-12",
  "address": "Av. Brasil 1500, casa 2, Centro, Campinas",
  "notes": null,
  "confianca": {"customer_name": 0.95, "items": 0.98, "delivery_date": 0.99,
                "address": 0.95, "notes": 0.95},
  "confianca_geral": 0.96
}
```

(The JSON is compacted here. The CLI prints it with indent 2.) Every value is right except
one item name: "de quatro queijos" should be "pizza de quatro queijos". The model was still
98% sure about `items`.

Batch mode prints one row per email and marks any value below the confidence threshold
with `!`:

```text
> extract-order mail --batch --today 2026-11-11 --threshold 0.9
FILE    CUSTOMER               ITEMS                           DELIVERY    ADDRESS                         CONF  LOW-CONFIDENCE FIELDS
------  ---------------------  ------------------------------  ----------  ------------------------------  ----  ---------------------
05.txt  Caio Mendonça          4x brigadeiro gourmet           !-          Travessa dos Ipês, 9 - Boa Vi~  0.60  delivery_date
12.txt  Sônia Martins Barreto  50x kits de lanche @16          2026-12-05  Entrega na Escola Municipal R~  0.95  -
16.txt  Larissa Duarte         5x geleia de morango; 2x gele~  2026-11-18  Rua Alfredo Lopes, 55, Santa ~  0.95  -

3 email(s), 1 with low-confidence fields (< 0.9), 0 failed. '!' marks a value below the threshold.
```

(A real run on three of the examples with `nvidia/nemotron-3.5-lightning:free`. Long cells are
cut with `~`. Email 05 only says "semana que vem", so the date is correctly null and flagged.)

## Quickstart

Requires Python 3.11+ and git (llm-kit is installed from GitHub). Commands for Windows
PowerShell. On macOS/Linux, activate with `source .venv/bin/activate`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
copy .env.example .env        # then set OPENROUTER_API_KEY and MODEL_PRIMARY

extract-order examples\01.txt                    # one email -> JSON
extract-order examples --batch --today 2026-11-11  # every .txt -> table
python -m evals.run --out evals/results/latest.json  # accuracy on the 20 labelled emails (EVAL_MODEL, else MODEL_PRIMARY)
python -m pytest                                 # offline tests, no key needed
python -m ruff check .
```

`--today YYYY-MM-DD` sets the date used to resolve "amanhã" or "sexta" (default: today).
`--threshold 0.7` changes the confidence cut-off in batch mode. Exit codes: 0 ok,
1 extraction failed, 2 bad usage or missing config.

Docker:

```powershell
docker build -t ai-05-extract-to-json .
docker run --rm --env-file .env ai-05-extract-to-json examples/01.txt
```

## How it works

```
email.txt ──> build_prompt(email, today) ──> LLMClient.complete_json(prompt, Order, max_attempts=3)
                                                 │
                                                 ├─ reply validates as Order ──> JSON out
                                                 └─ invalid: error fed back, try again (≤3)
                                                     └─ still invalid ──> JSONValidationError
```

- `src/order_extractor/schema.py`: Pydantic v2 models. `Item` has `name`, `quantity` (int > 0)
  and `unit_price` (optional, ≥ 0). `Order` has `customer_name`, `items` (at least one),
  `delivery_date` (ISO date or null), `address`, `notes`, `confianca` (field name → 0..1,
  unknown keys rejected) and `confianca_geral`. `OrderData` is the same order without the
  confidence fields. The hand-labelled expected files use it.
- `src/order_extractor/extract.py`: the prompt. It gives today's date and weekday in Portuguese,
  defines how to resolve "sexta", "amanhã" and "dia 5", says vague dates become null, and says
  to use null for anything not stated and never invent values, nor derive unit prices from
  totals. Then it calls `complete_json`. Retry, feedback and parsing all live in llm-kit and
  are not reimplemented here.
- `src/order_extractor/batch.py` and `cli.py`: `--batch` mode and the table.
- `examples/`: 20 fake emails (invented people and addresses) in varied styles: formal,
  WhatsApp-like, typos, quoted reply threads, mid-email corrections, pickup instead of
  delivery, totals without unit prices, vague dates. `examples/expected/NN.json` holds the
  correct order for each, labelled against a reference date of Wednesday 2026-11-11.
- `evals/`: `run.py` runs the extractor on the examples and `scoring.py` compares fields.
  Names, item names and addresses are matched leniently: accents, case, punctuation and small
  typos are ignored, and every expected word must appear. Quantities, prices and dates must
  match exactly. Items are paired by name, so extra or missing items both count against
  `item_name`. CI runs a 5-email smoke eval, but only when the `OPENROUTER_API_KEY` secret is
  set.

## Results

Measured on 2026-11-11 with `python -m evals.run`: all 20 labelled emails, model
`nvidia/nemotron-3.5-lightning:free`, no fallbacks, reference date 2026-11-11. The full output is
in [`evals/results/latest.json`](evals/results/latest.json).

| Field | Accuracy |
|---|---|
| customer_name | 90.0% (18/20) |
| items: name | 91.9% (34/37) |
| items: quantity | 91.9% (34/37) |
| items: unit_price | 91.9% (34/37) |
| delivery_date | 100.0% (20/20) |
| address | 95.0% (19/20) |

- **Exact-match emails:** 75% (15/20). Every scored field was right.
- **Failed extractions:** 0. **Mean attempts per email:** 1.00 of 3. Every first reply validated,
  so the retry path was never needed in this run. The offline tests cover it.
- **Mean confidence:** 0.92 on correct fields, 0.92 on incorrect fields.

That last line is the most important result: **the model's self-reported confidence did not
separate right answers from wrong ones**. Every wrong field below was reported at 0.85–0.98.
The `< 0.7` flag only catches cases where the model knows the email is vague, such as a missing
date. It does not catch the model being wrong.

`google/gemma-4-31b-it:free` (the original `MODEL_PRIMARY`) could not be measured. Every call
returned HTTP 429 from the provider, even with six retries and 10–60 s backoff.

## Limits

What it got wrong in the eval run above:

- **Who is the customer.** It returned the nickname "Duda" instead of "Maria Eduarda Lins" (02),
  and the company "Café Aurora Ltda." instead of the buyer who signed (06). The prompt allows
  "person (or business)", so 06 is partly a prompt ambiguity.
- **Shortened item names.** "bolo de cenora" for "bolo de cenoura com cobertura de chocolate"
  (11) and "kits de lanche" for "kit de lanche (sanduíche natural + suco)" (12). An item only
  counts as found when it shares at least 60% of the expected words. When a name falls below
  that, its quantity and price also count as wrong, even if the numbers are right.
- **Dropped items.** In 16 ("3 de morango… na verdade são 5"), it applied the correction but
  dropped the 2 potes de damasco. The same email came out right in another run: the output
  is not deterministic, and one run of 20 emails is a small sample.
- **Dropped address parts.** It dropped "Escola Municipal Recanto Verde" from the address (12).

Other limits:

- Confidence is self-reported and, as measured above, uncalibrated. Don't auto-accept orders
  because of it.
- The 20 emails are fake and written by me. Real inboxes have attachments, HTML, signatures with
  several names and long threads.
- Relative dates depend on `--today`. An email read days after it was sent needs its sent date,
  not today's.
- Prices are assumed to be in reais. Only per-unit prices are extracted: a total for several
  units gives `unit_price: null` by design.
- Free models on OpenRouter are slow (about 1 min per email here) and heavily rate-limited. The
  eval waits 6 s between emails (`--delay`).

## License

MIT
