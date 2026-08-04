import pytest

from app.config import Settings
from app.services.llm_client import LLMClient, LLMError


def _make_client(monkeypatch, raw_response: str):
    settings = Settings(anthropic_api_key="test")
    client = LLMClient(settings)

    async def fake_complete(**kwargs):
        return raw_response

    monkeypatch.setattr(client, "complete", fake_complete)
    return client


@pytest.mark.asyncio
async def test_complete_json_parses_clean_json(monkeypatch):
    client = _make_client(monkeypatch, '{"a": 1}')
    result = await client.complete_json(system="s", user="u")
    assert result == {"a": 1}


@pytest.mark.asyncio
async def test_complete_json_strips_markdown_fences(monkeypatch):
    client = _make_client(monkeypatch, '```json\n{"a": 1}\n```')
    result = await client.complete_json(system="s", user="u")
    assert result == {"a": 1}


@pytest.mark.asyncio
async def test_complete_json_extracts_json_from_preamble(monkeypatch):
    client = _make_client(monkeypatch, 'Sure, here you go: {"a": 1} Hope that helps!')
    result = await client.complete_json(system="s", user="u")
    assert result == {"a": 1}


@pytest.mark.asyncio
async def test_complete_json_raises_on_unparseable_response(monkeypatch):
    client = _make_client(monkeypatch, "no json anywhere in sight")
    with pytest.raises(LLMError):
        await client.complete_json(system="s", user="u")
