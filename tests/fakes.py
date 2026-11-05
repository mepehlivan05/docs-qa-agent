"""Stand-ins for Anthropic SDK responses, so the agent is tested without API calls."""

from __future__ import annotations

from types import SimpleNamespace


def citation(document_index: int, start: int, end: int | None = None, text: str = "..."):
    return SimpleNamespace(
        type="content_block_location",
        cited_text=text,
        document_index=document_index,
        document_title="",
        start_block_index=start,
        end_block_index=start + 1 if end is None else end,
    )


def text_block(text: str, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def usage(input_tokens=50, output_tokens=40, cache_read=8000, cache_write=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def response(*blocks, stop_reason="end_turn", model="claude-opus-5-5", usage_=None):
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop_reason, model=model, usage=usage_ or usage()
    )


class FakeClient:
    """Records the request and returns a canned response."""

    def __init__(self, canned):
        self.canned = canned
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.canned
