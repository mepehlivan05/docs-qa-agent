"""Runs the real Anthropic SDK against a fake HTTP transport: checks that the
request is accepted and sent as expected, and that real SDK response objects
(including citations) are parsed correctly. No network, no API key."""

import json

import anthropic
import httpx2 as httpx

from docs_qa import DocsAgent
from docs_qa.agent import FALLBACK_BETA
from docs_qa.corpus import parse_article

ARTICLES = [parse_article("refunds", "# Refunding a payment\n## Fees on refunds\nStripe keeps its fee.\n")]

API_RESPONSE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [
        {"type": "text", "text": "No, "},
        {
            "type": "text",
            "text": "Stripe does not return its processing fee.",
            "citations": [{
                "type": "content_block_location",
                "cited_text": "Stripe keeps its fee.",
                "document_index": 0,
                "document_title": "Refunding a payment",
                "start_block_index": 0,
                "end_block_index": 1,
            }],
        },
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 20,
        "output_tokens": 15,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 500,
    },
}


def test_real_sdk_request_and_response():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=API_RESPONSE)

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    answer = DocsAgent(ARTICLES, client=client).ask("Do I get the Stripe fee back?")

    body = json.loads(sent[0].content)
    assert sent[0].url.path == "/v1/messages"
    assert FALLBACK_BETA in sent[0].headers["anthropic-beta"]
    assert body["fallbacks"] == "default"
    assert body["messages"][0]["content"][0]["citations"] == {"enabled": True}
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    assert answer.text == "No, Stripe does not return its processing fee."
    assert [(s.article, s.section) for s in answer.sources] == [("refunds", "Fees on refunds")]
    assert answer.cost_usd > 0
