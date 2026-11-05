from pathlib import Path

import pytest

from docs_qa.corpus import OVERVIEW, load_corpus, parse_article, to_document_blocks

DOCS = Path(__file__).parent.parent / "docs"


def test_parses_title_intro_and_sections():
    article = parse_article("demo", "# Demo title\n\nIntro text.\n\n## First\nOne.\n\n## Second\nTwo.\n")

    assert article.title == "Demo title"
    assert [s.heading for s in article.sections] == [OVERVIEW, "First", "Second"]
    assert article.sections[1].text == "One."


def test_article_without_intro_has_no_overview_section():
    article = parse_article("demo", "# T\n## Only\nBody\n")

    assert [s.heading for s in article.sections] == ["Only"]


def test_empty_article_is_rejected():
    with pytest.raises(ValueError):
        parse_article("empty", "# Title only\n")


def test_loads_sample_corpus_in_stable_order():
    articles = load_corpus(DOCS)

    assert len(articles) == 24
    assert [a.slug for a in articles] == sorted(a.slug for a in articles)


def test_document_blocks_are_citable_and_cache_the_whole_corpus():
    articles = load_corpus(DOCS)
    blocks = to_document_blocks(articles)

    assert len(blocks) == len(articles)
    assert all(b["citations"] == {"enabled": True} for b in blocks)
    # One block per section, so a citation's block index maps to a section.
    assert all(len(b["source"]["content"]) == len(a.sections) for b, a in zip(blocks, articles))
    # Only the last document carries the cache breakpoint.
    assert [("cache_control" in b) for b in blocks] == [False] * (len(blocks) - 1) + [True]
