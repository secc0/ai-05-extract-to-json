# ai-05-extract-to-json

Turns free-text order emails (Portuguese) into JSON validated by Pydantic, retrying automatically when the model returns invalid output.

**Status:** in progress (day 5 of my 56-day AI projects series, built on the OpenRouter API).

## Plan
- Pydantic `Order` schema: customer, items (name, quantity, unit price), delivery date, address, notes
- Extraction through `complete_json()` from [llm-kit](https://github.com/secc0/ai-03-llm-kit), with retry when validation fails
- 20 fake example emails, per-field accuracy and a `confianca` field
- Tests with mocked responses and CI

## License
MIT
