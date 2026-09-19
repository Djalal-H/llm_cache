import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from cachewise.assistant import canonical, digest
from cachewise.fixtures import FixtureService
from cachewise.models import ChatRequest


class DatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    family: str
    split: str
    expected_intent: str
    context: dict
    answer_requirements: str
    request: ChatRequest
    seed: int


# Each row is one paraphrase family, kept entirely within one split.
# Difficult conditions are separate intents with their own answer requirements.
FAMILIES = [
    (
        "return_window",
        "returns",
        "Use 30 days for standard and 60 days for premium.",
        [
            "How long do I have to return an item?",
            "What is my return window?",
            "How many days are returns accepted?",
            "Tell me the deadline for returning a purchase.",
        ],
    ),
    (
        "return_condition",
        "returns",
        "Unused and original packaging are required.",
        [
            "Can I return a used item?",
            "Are opened and used items returnable?",
            "Does a return need its original packaging?",
            "What condition must a return be in?",
        ],
    ),
    (
        "personalized_return",
        "returns",
        "Personalized items cannot be returned unless defective.",
        [
            "Can I return a personalized item?",
            "Is an engraved item returnable if not defective?",
            "Can I send back a custom item just because I dislike it?",
            "What is the return rule for personalized products?",
        ],
    ),
    (
        "defective_return",
        "returns",
        "Defective personalized items are returnable; US label is free.",
        [
            "Can I return a defective personalized item?",
            "My engraved item is faulty. Can I return it?",
            "Is there an exception for defective custom products?",
            "Do I pay for a return label for a defective item?",
        ],
    ),
    (
        "return_label",
        "returns",
        "US labels cost $5 unless defective; EU labels are free.",
        [
            "How much is a return label?",
            "Do I pay return postage?",
            "Are return labels free in my region?",
            "What is the return shipping fee?",
        ],
    ),
    (
        "shipping_time",
        "shipping",
        "US 3–5 business days; EU 5–8 business days.",
        [
            "How long does standard shipping take?",
            "What is the standard delivery time?",
            "How many business days does shipping take?",
            "Tell me the usual shipping duration.",
        ],
    ),
    (
        "shipping_fee",
        "shipping",
        "Apply regional fee, free-shipping minimum and premium waiver.",
        [
            "How much does shipping cost?",
            "What is my shipping fee?",
            "Do you charge for standard shipping?",
            "Explain standard delivery charges.",
        ],
    ),
    (
        "free_shipping",
        "shipping",
        "US minimum $50, EU €70; premium has no minimum.",
        [
            "When is standard shipping free?",
            "What is the free shipping minimum?",
            "How much must I spend for free delivery?",
            "Do premium customers need a minimum spend?",
        ],
    ),
    (
        "shipping_us",
        "shipping",
        "Apply US destination: 3–5 days, $5, $50 minimum; premium free.",
        [
            "What are the shipping rules to the US?",
            "How long is delivery to the United States?",
            "What does standard shipping to the US cost?",
            "Can you explain US delivery?",
        ],
    ),
    (
        "shipping_eu",
        "shipping",
        "Apply EU destination: 5–8 days, €7, €70 minimum; premium free.",
        [
            "What are the shipping rules to the EU?",
            "How long is delivery to the European Union?",
            "What does standard shipping to the EU cost?",
            "Can you explain EU delivery?",
        ],
    ),
    (
        "unsupported_destination",
        "shipping",
        "Shipping outside US/EU is unavailable.",
        [
            "Do you ship to Algeria?",
            "Can I have an order shipped to Canada?",
            "Is standard delivery available in Japan?",
            "What is shipping to Brazil?",
        ],
    ),
    (
        "payment_methods",
        "payment",
        "Visa, Mastercard, PayPal are supported.",
        [
            "Which payment methods do you accept?",
            "How can I pay for a purchase?",
            "What payment options are available?",
            "List the supported payment methods.",
        ],
    ),
    (
        "cash_payment",
        "payment",
        "Cash on delivery is not supported.",
        [
            "Can I pay cash on delivery?",
            "Do you accept cash at delivery?",
            "Is cash payment an option?",
            "Can I pay the courier with cash?",
        ],
    ),
    (
        "crypto_payment",
        "payment",
        "Cryptocurrency is not supported.",
        [
            "Can I pay with cryptocurrency?",
            "Do you accept Bitcoin?",
            "Is payment with crypto supported?",
            "May I use Ethereum at checkout?",
        ],
    ),
    (
        "bank_payment",
        "payment",
        "Bank transfers are not supported.",
        [
            "Can I pay by bank transfer?",
            "Do you support wire transfers?",
            "May I transfer payment from my bank?",
            "Is a bank transfer accepted at checkout?",
        ],
    ),
    (
        "order_status",
        "personalized",
        "Use only this customer's current fixture order status.",
        [
            "Where is my order?",
            "What is my order status?",
            "Has my purchase shipped?",
            "Tell me the current status of my order.",
        ],
    ),
    (
        "order_tracking",
        "personalized",
        "Use owned tracking value, or disclose it is unavailable.",
        [
            "What is my tracking number?",
            "How can I track my shipment?",
            "Give me the tracking reference for my order.",
            "Is tracking available for my purchase?",
        ],
    ),
    (
        "order_refund",
        "personalized",
        "Use only this customer's fixture refund status.",
        [
            "What is my refund status?",
            "Has my refund completed?",
            "Is a refund pending on my order?",
            "Tell me whether my purchase was refunded.",
        ],
    ),
    (
        "order_delivery",
        "personalized",
        "Use owned delivery estimate, or say unavailable.",
        [
            "When will my order arrive?",
            "What is my delivery estimate?",
            "Which date is my purchase expected?",
            "Tell me my estimated delivery date.",
        ],
    ),
    (
        "live_stock",
        "stock",
        "No stock data is connected; do not assert availability.",
        [
            "Is the blue tote in stock right now?",
            "How many desk lamps are available?",
            "Can I buy the travel mug today?",
            "Is the notebook currently sold out?",
        ],
    ),
]


def make_dataset(seed: int = 42) -> list[DatasetRequest]:
    rng = random.Random(seed)
    families = list(FAMILIES)
    rng.shuffle(families)
    snapshot = FixtureService().snapshot()
    customers = snapshot.customers
    rows: list[DatasetRequest] = []
    # Exactly 40% unique requests and 60% exact repeats within each partition.
    for split, subset, unique_count, repeat_count in (
        ("tuning", families[:6], 120, 180),
        ("evaluation", families[6:], 280, 420),
    ):
        candidates = []
        for family, intent, requirements, questions in subset:
            for customer in customers:
                for language in ("en", "fr"):
                    for question in questions:
                        owned_orders = snapshot.order_context(customer.id, question)["orders"]
                        resolved_requirements = requirements
                        if intent == "personalized":
                            resolved_requirements += " Current expected owned orders: " + canonical(
                                owned_orders
                            )
                        candidates.append(
                            DatasetRequest(
                                id="",
                                family=family,
                                split=split,
                                expected_intent=intent,
                                context={
                                    "customer_id": customer.id,
                                    "tier": customer.tier,
                                    "region": customer.region,
                                    "language": language,
                                },
                                answer_requirements=resolved_requirements,
                                request=ChatRequest(
                                    customer_id=customer.id, question=question, language=language
                                ),
                                seed=seed,
                            )
                        )
        rng.shuffle(candidates)
        unique = candidates[:unique_count]
        partition = unique + [rng.choice(unique).model_copy(deep=True) for _ in range(repeat_count)]
        rng.shuffle(partition)
        for row in partition:
            rows.append(row.model_copy(update={"id": f"request-{len(rows) + 1:04d}"}))
    return rows


def write_dataset(path: Path, seed: int = 42) -> list[DatasetRequest]:
    rows = make_dataset(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    return rows


def read_dataset(path: Path) -> list[DatasetRequest]:
    return [
        DatasetRequest.model_validate_json(line) for line in path.read_text().splitlines() if line
    ]


def dataset_hash(rows: list[DatasetRequest]) -> str:
    return digest(canonical([row.model_dump(mode="json") for row in rows]))
