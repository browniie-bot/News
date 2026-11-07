import asyncio
import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("BROWNIIE_RECORDER"), reason="recorder only"
)


def test_complete_skips_thinking_block(monkeypatch):
    from src.ai.client import AnthropicClient
    from src.models import AIConfig, AIProvider

    body = {
        "id": "msg_1", "type": "message", "role": "assistant",
        "model": "claude-sonnet-5-5", "stop_reason": "end_turn",
        "stop_sequence": None,
        "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "text", "text": "hello"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    with open("/tmp/browniie-replies.json", "w") as f:
        json.dump({"/v1/messages": body}, f)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    cfg = AIConfig(provider=AIProvider.ANTHROPIC, model="claude-sonnet-5-5",
                   api_key_env="ANTHROPIC_API_KEY")
    out = asyncio.run(AnthropicClient(cfg).complete("sys", "user"))
    assert out == "hello"
