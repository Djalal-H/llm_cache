import json

import pytest

from cachewise.assistant import Assistant
from cachewise.fixtures import FixtureService
from cachewise.models import ChatRequest, HistoryMessage


def test_profiles_and_owned_orders(generation):
    assistant = Assistant(FixtureService(), generation)
    alice = assistant.prepare(ChatRequest(customer_id="demo-alice", question="Where is my order?"))
    bob = assistant.prepare(ChatRequest(customer_id="demo-bob", question="Where is my order?"))
    assert "CW-1001" in alice.messages[-2].content
    assert "CW-1002" not in alice.messages[-2].content
    assert "CW-1002" in bob.messages[-2].content
    assert '"tier":"standard"' in alice.messages[-2].content
    assert '"tier":"premium"' in bob.messages[-2].content
    assert "30 calendar days" in alice.messages[0].content
    assert "60 calendar days" in bob.messages[0].content
    assert alice.system_prompt_hash != bob.system_prompt_hash
    assert alice.serialized_prompt_hash != bob.serialized_prompt_hash


def test_order_reference_cannot_lookup_another_customer():
    context = FixtureService().snapshot().order_context("demo-alice", "Track CW-1002")
    assert context["orders"] == []
    assert context["requested_order_missing_or_not_owned"]
    assert "Desk lamp" not in json.dumps(context)


def test_unknown_customer_rejected(generation):
    with pytest.raises(KeyError):
        Assistant(FixtureService(), generation).prepare(
            ChatRequest(customer_id="missing", question="What is shipping?")
        )


def test_static_prefix_and_history_order(generation):
    assistant = Assistant(FixtureService(), generation)
    first = assistant.prepare(ChatRequest(customer_id="demo-claire", question="Returns?"))
    second = assistant.prepare(
        ChatRequest(
            customer_id="demo-claire",
            question="Shipping?",
            history=[
                HistoryMessage(role="assistant", content="Your order used to be processing."),
            ],
        )
    )
    assert first.messages[0] == second.messages[0]
    assert second.messages[1].role == "assistant"
    assert "Current server-resolved" in second.messages[-2].content


def test_fixture_updates_and_prompt_identity(tmp_path, generation, monkeypatch):
    import cachewise.assistant as module

    snapshot = FixtureService().snapshot().model_dump()
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(snapshot))
    assistant = Assistant(FixtureService(path), generation)
    request = ChatRequest(customer_id="demo-alice", question="Order status?")
    before = assistant.prepare(request)
    snapshot["orders"][0]["status"] = "delivered"
    path.write_text(json.dumps(snapshot))
    updated_order = assistant.prepare(request)
    assert '"status":"delivered"' in updated_order.messages[-2].content
    assert before.system_prompt_hash == updated_order.system_prompt_hash
    snapshot["policy_version"] += 1
    snapshot["policies"]["returns"]["standard_window_days"] = 14
    path.write_text(json.dumps(snapshot))
    updated_policy = assistant.prepare(request)
    assert updated_policy.system_prompt_hash != before.system_prompt_hash
    assert updated_policy.policy_version == 2
    monkeypatch.setattr(module, "INSTRUCTIONS", module.INSTRUCTIONS + " Be kind.\n")
    assert assistant.prepare(request).system_prompt_hash != updated_policy.system_prompt_hash


async def test_uncached_requests_always_generate(generation):
    assistant = Assistant(FixtureService(), generation)
    request = ChatRequest(customer_id="demo-alice", question="Returns?")
    first, second = await assistant.answer(request), await assistant.answer(request)
    assert len(generation.messages) == 2
    assert first.cache_outcome == second.cache_outcome == "bypass"
    assert first.request_id != second.request_id
