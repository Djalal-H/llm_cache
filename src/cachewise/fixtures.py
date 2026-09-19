import json
import re
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    tier: str
    region: str


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    customer_id: str
    item: str
    status: str
    tracking: str | None
    delivery_estimate: str | None
    refund_status: str


class FixtureSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_version: int = Field(ge=1)
    catalogue_version: int = Field(ge=1)
    customers: list[Customer]
    orders: list[Order]
    policies: dict

    @model_validator(mode="after")
    def unique_ownership(self) -> "FixtureSnapshot":
        ids = [customer.id for customer in self.customers]
        order_ids = [order.id for order in self.orders]
        if len(ids) != len(set(ids)) or len(order_ids) != len(set(order_ids)):
            raise ValueError("fixture identifiers must be unique")
        if any(order.customer_id not in ids for order in self.orders):
            raise ValueError("orders must belong to known customers")
        return self

    def customer(self, customer_id: str) -> Customer:
        for customer in self.customers:
            if customer.id == customer_id:
                return customer
        raise KeyError(customer_id)

    def order_context(self, customer_id: str, question: str) -> dict:
        owned = [order for order in self.orders if order.customer_id == customer_id]
        requested = {value.upper() for value in re.findall(r"\bCW-\d+\b", question, re.I)}
        selected = [order for order in owned if order.id in requested] if requested else owned
        return {
            "orders": [order.model_dump() for order in selected],
            "requested_order_missing_or_not_owned": bool(requested - {o.id for o in owned}),
            "source": "fictional_fixture_lookup",
        }

    def policy_context(self, customer: Customer) -> dict:
        returns = self.policies["returns"]
        shipping = self.policies["shipping"]
        regional_shipping = shipping[customer.region]
        duration = regional_shipping["duration_business_days"]
        return {
            "return_window": (
                f"{returns[customer.tier + '_window_days']} {returns['window_unit']} "
                f"from {returns['window_starts_at']}"
            ),
            "return_condition": returns["condition"],
            "return_label_fee": returns[customer.region],
            "outbound_shipping_fee": {
                "amount": 0 if customer.tier == "premium" else regional_shipping["fee"],
                "currency": regional_shipping["currency"],
                "free_shipping_minimum": 0
                if customer.tier == "premium"
                else regional_shipping["free_shipping_minimum"],
            },
            "shipping_duration": f"{duration[0]}–{duration[1]} business days",
            "payment_methods": self.policies["payment"],
            "live_stock_availability": "UNKNOWN: no live inventory data is available",
            "order_total_and_paid_shipping": "UNKNOWN: these values are not in the order lookup",
        }

    def prompt_policies(self, customer: Customer) -> dict:
        resolved = self.policy_context(customer)
        shipping = self.policies["shipping"]
        regional_rules = {}
        for region in shipping["destinations"]:
            rules = dict(shipping[region])
            if customer.tier == "premium":
                rules["fee"] = 0
                rules["free_shipping_minimum"] = 0
            regional_rules[region] = rules
        return {
            "returns": {
                "window": resolved["return_window"],
                "condition": resolved["return_condition"],
                "return_label_fee": resolved["return_label_fee"],
            },
            "shipping": {
                "customer_region": customer.region,
                "rules_by_destination": regional_rules,
                "outside_supported_destinations": shipping["international"],
            },
            "payment": self.policies["payment"],
            "stock": self.policies["stock"],
        }


class FixtureService:
    def __init__(self, path: Path | None = None):
        self.path = path

    def snapshot(self) -> FixtureSnapshot:
        # Load once per request, so updates are current and versions remain consistent in flight.
        source = self.path or files("cachewise").joinpath("data/fixtures.json")
        return FixtureSnapshot.model_validate(json.loads(source.read_text(encoding="utf-8")))
