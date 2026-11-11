"""Pydantic v2 models for an order extracted from an email."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]

# Fields the model may report a confidence for (keys of Order.confianca).
CONFIDENCE_FIELDS = ("customer_name", "items", "delivery_date", "address", "notes")


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Product name as written in the email.")
    quantity: int = Field(gt=0, description="How many units were ordered.")
    unit_price: float | None = Field(
        default=None, ge=0, description="Price of ONE unit, only if the email states it."
    )

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value


class OrderData(BaseModel):
    """The order itself. Used on its own for the hand-labelled expected files."""

    model_config = ConfigDict(extra="forbid")

    customer_name: str = Field(min_length=1, description="Who is placing the order.")
    items: list[Item] = Field(min_length=1)
    delivery_date: date | None = Field(default=None, description="ISO date (YYYY-MM-DD).")
    address: str | None = Field(default=None, description="Delivery address, if stated.")
    notes: str | None = Field(default=None, description="Other instructions, if any.")


class Order(OrderData):
    """An extracted order plus the model's self-reported confidence."""

    confianca: dict[str, Confidence] = Field(
        description=(
            "Confidence from 0 to 1 for each field, keyed by field name: "
            + ", ".join(CONFIDENCE_FIELDS)
        )
    )
    confianca_geral: Confidence = Field(description="Overall confidence from 0 to 1.")

    @field_validator("confianca")
    @classmethod
    def _known_keys(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(value) - set(CONFIDENCE_FIELDS))
        if unknown:
            raise ValueError(
                f"unknown confidence keys {unknown}; allowed: {list(CONFIDENCE_FIELDS)}"
            )
        return value

    def data(self) -> OrderData:
        """The order without the confidence fields."""
        return OrderData.model_validate(self.model_dump(exclude={"confianca", "confianca_geral"}))
