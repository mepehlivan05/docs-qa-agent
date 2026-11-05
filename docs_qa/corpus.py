"""Loads Markdown articles and turns them into citable Claude document blocks.

Each article becomes one document; each `## ` section becomes one content
block inside it. Claude's citations then point at (document, block), which maps
straight back to (article, section) without any fuzzy text matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OVERVIEW = "Overview"


@dataclass(frozen=True)
class Section:
    heading: str
    text: str


@dataclass(frozen=True)
class Article:
    slug: str
    title: str
    sections: tuple[Section, ...]


def parse_article(slug: str, markdown: str) -> Article:
    title = slug
    sections: list[Section] = []
    heading = OVERVIEW
    lines: list[str] = []

    def flush() -> None:
        text = "\n".join(lines).strip()
        if text:
            sections.append(Section(heading, text))

    for line in markdown.splitlines():
        if line.startswith("# ") and title == slug and not sections and not lines:
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    flush()

    if not sections:
        raise ValueError(f"article {slug!r} has no content")
    return Article(slug, title, tuple(sections))


def load_corpus(folder: str | Path) -> list[Article]:
    paths = sorted(Path(folder).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no .md files in {folder}")
    return [parse_article(p.stem, p.read_text(encoding="utf-8")) for p in paths]


def to_document_blocks(articles: list[Article]) -> list[dict]:
    """Document blocks for the Messages API, in the same order as `articles`.

    The last block carries cache_control, so the whole corpus is cached once
    and every later question reads it at the cache-read price.
    """
    blocks = [
        {
            "type": "document",
            "source": {
                "type": "content",
                "content": [
                    {"type": "text", "text": f"## {s.heading}\n{s.text}"} for s in article.sections
                ],
            },
            "title": article.title,
            "context": f"Article id: {article.slug}",
            "citations": {"enabled": True},
        }
        for article in articles
    ]
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks
