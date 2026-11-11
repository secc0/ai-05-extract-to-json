"""Build the extraction prompt and call llm-kit's complete_json."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from llm_kit import CallResult, LLMClient

from order_extractor.schema import CONFIDENCE_FIELDS, Order

MAX_ATTEMPTS = 3

WEEKDAYS_PT = (
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
)

PROMPT_TEMPLATE = """\
You extract purchase orders from customer emails written in Brazilian Portuguese.

Today is {today_iso} ({weekday_pt}). Resolve relative dates against today:
- "hoje" = today, "amanhã" = tomorrow, "depois de amanhã" = today + 2 days.
- A weekday name ("sexta", "na segunda") means the NEXT occurrence of that weekday after today.
- "dia 5" means the next date with that day of the month that is not in the past.
- If the date is vague ("semana que vem", "o quanto antes", "fim do mês") use null.
Write delivery_date as YYYY-MM-DD.

Rules:
- Use null for anything the email does not state. Never invent or guess values.
- customer_name: the person (or business) placing the order, as they sign or introduce themselves.
- items: one entry per product. quantity is an integer number of units ("uma dúzia" = 12).
  unit_price is the price of ONE unit in reais as a number (R$ 12,50 -> 12.5), or null if no
  per-unit price is stated. Do not divide totals to derive a unit price.
- address: the delivery address as written (street, number, complement, neighbourhood, city).
  Use null if it is not given or the customer will pick the order up.
- notes: other delivery or preparation instructions, or null.
- confianca: your confidence from 0 to 1 for each of these fields: {fields}.
  Use low values (below 0.7) when the email is ambiguous, misspelled or you had to interpret.
- confianca_geral: your overall confidence from 0 to 1.

Email:
<<<
{email}
>>>"""


def build_prompt(text: str, today: date) -> str:
    """Return the extraction prompt for one email."""
    return PROMPT_TEMPLATE.format(
        today_iso=today.isoformat(),
        weekday_pt=WEEKDAYS_PT[today.weekday()],
        fields=", ".join(CONFIDENCE_FIELDS),
        email=text.strip(),
    )


def extract_order(
    text: str,
    *,
    client: LLMClient | None = None,
    today: date | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    on_call: Callable[[CallResult], None] | None = None,
) -> Order:
    """Extract a validated Order from an email.

    Retries with validation feedback come from llm-kit's complete_json. Raises
    llm_kit.JSONValidationError when no reply validates, LLMCallError when calls fail,
    and ConfigError when no client is passed and settings are missing.
    """
    client = client or LLMClient()
    prompt = build_prompt(text, today or date.today())
    return client.complete_json(prompt, Order, max_attempts=max_attempts, on_call=on_call)
