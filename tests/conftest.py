import pytest

from cachewise.models import GenerationResult, Usage


class TestGeneration:
    """Test-only provider, never used by production code or live CLI commands."""

    __test__ = False

    def __init__(self):
        self.messages = []
        self.error = None
        self.available = True

    async def generate(self, messages):
        self.messages.append(messages)
        if self.error:
            raise self.error
        return GenerationResult(
            answer="Fixture-grounded test answer",
            model="cachewise-model",
            finish_reason="stop",
            usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        )

    async def ready(self):
        return self.available


@pytest.fixture
def generation():
    return TestGeneration()
