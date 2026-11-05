"""The Q&A agent: one Claude request per question, grounded in the articles."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import anthropic

from .corpus import Article, to_document_blocks
from .pricing import cost_usd

MODEL = "claude-opus-5-5"
IDK = "I don't know."

# On a safety refusal the API re-runs the request on a suitable fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are a support assistant. Answer the user's question using ONLY the help-center articles provided as documents.

Rules:
- Base every statement on the articles and cite the passages you used.
- If the articles do not contain the answer, reply with exactly: I don't know.
- Do not use outside knowledge and do not guess, even when an answer seems obvious.
- If the articles answer only part of the question, answer that part and say plainly which part they don't cover.
- Keep answers short: a few sentences or a short list."""

_IDK_PATTERN = re.compile(r"^\W*i\s+don'?t\s+know\b", re.IGNORECASE)


@dataclass(frozen=True)
class Source:
    article: str  # slug, e.g. "refunds"
    title: str
    section: str


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    is_idk: bool = False
    cost_usd: float = 0.0
    model: str = MODEL
    stop_reason: str | None = None

    @property
    def articles(self) -> list[str]:
        return list(dict.fromkeys(s.article for s in self.sources))


class DocsAgent:
    def __init__(
        self,
        articles: list[Article],
        client: anthropic.Anthropic | None = None,
        model: str = MODEL,
        effort: str = "medium",
    ):
        self.articles = articles
        self.documents = to_document_blocks(articles)
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def ask(self, question: str) -> Answer:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=[
                {
                    "role": "user",
                    # Documents first (cached prefix), the varying question last.
                    "content": [*self.documents, {"type": "text", "text": question}],
                }
            ],
        )
        return self._to_answer(response)

    def _to_answer(self, response) -> Answer:
        cost = cost_usd(response.model, response.usage)

        if response.stop_reason == "refusal":
            return Answer(IDK, [], True, cost, response.model, response.stop_reason)

        parts: list[str] = []
        sources: dict[Source, None] = {}
        for block in response.content:
            if block.type != "text":
                continue  # thinking / fallback blocks carry no answer text
            parts.append(block.text)
            for citation in block.citations or []:
                for source in self._sources_for(citation):
                    sources[source] = None

        text = "".join(parts).strip()

        # Hard rule enforced in code, not just in the prompt: an answer without
        # a single citation is not grounded in the articles, so it is replaced
        # with "I don't know" instead of being shown to the user.
        if not sources or _IDK_PATTERN.match(text):
            return Answer(IDK, [], True, cost, response.model, response.stop_reason)

        return Answer(text, list(sources), False, cost, response.model, response.stop_reason)

    def _sources_for(self, citation) -> list[Source]:
        if citation.type != "content_block_location":
            return []
        if not 0 <= citation.document_index < len(self.articles):
            return []
        article = self.articles[citation.document_index]
        last = min(citation.end_block_index, len(article.sections))
        return [
            Source(article.slug, article.title, article.sections[i].heading)
            for i in range(max(citation.start_block_index, 0), last)
        ]
