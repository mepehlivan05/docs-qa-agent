from docs_qa import IDK, DocsAgent
from docs_qa.agent import FALLBACK_BETA, MODEL
from docs_qa.corpus import parse_article

from .fakes import FakeClient, citation, response, text_block, usage

ARTICLES = [
    parse_article("refunds", "# Refunding a payment\n## Refund a card payment\nWithin 180 days.\n## Fees on refunds\nStripe keeps its fee.\n"),
    parse_article("taxes", "# Taxes\n## Add tax rates\nUp to 3 per line.\n"),
]


def ask(canned, question="q"):
    client = FakeClient(canned)
    return DocsAgent(ARTICLES, client=client).ask(question), client


def test_maps_citations_to_article_and_section():
    answer, _ = ask(response(
        text_block("You can refund within 180 days", [citation(0, 0)]),
        text_block(", but Stripe keeps its fee.", [citation(0, 1)]),
    ))

    assert not answer.is_idk
    assert answer.text == "You can refund within 180 days, but Stripe keeps its fee."
    assert [(s.article, s.section) for s in answer.sources] == [
        ("refunds", "Refund a card payment"),
        ("refunds", "Fees on refunds"),
    ]
    assert answer.articles == ["refunds"]


def test_citation_spanning_several_sections_lists_each_once():
    answer, _ = ask(response(
        text_block("A", [citation(0, 0, 2)]),
        text_block("B", [citation(0, 1)]),
    ))

    assert [s.section for s in answer.sources] == ["Refund a card payment", "Fees on refunds"]


def test_uncited_answer_is_replaced_with_idk():
    # The model answered from general knowledge: no citation -> not shown.
    answer, _ = ask(response(text_block("Yes, nonprofits get 50% off.")))

    assert answer.is_idk
    assert answer.text == IDK
    assert answer.sources == []


def test_model_idk_is_reported_as_idk():
    answer, _ = ask(response(text_block("I don't know.")))

    assert answer.is_idk and answer.text == IDK


def test_idk_with_stray_citation_is_still_idk():
    answer, _ = ask(response(text_block("I don't know", [citation(1, 0)])))

    assert answer.is_idk


def test_refusal_is_reported_as_idk():
    answer, _ = ask(response(stop_reason="refusal"))

    assert answer.is_idk and answer.stop_reason == "refusal"


def test_out_of_range_citations_are_ignored():
    answer, _ = ask(response(text_block("Odd", [citation(9, 0), citation(1, 5)])))

    assert answer.is_idk


def test_non_text_blocks_are_skipped():
    from types import SimpleNamespace

    answer, _ = ask(response(
        SimpleNamespace(type="thinking", thinking=""),
        text_block("Up to 3 tax rates per line.", [citation(1, 0)]),
    ))

    assert answer.text == "Up to 3 tax rates per line."


def test_request_shape():
    _, client = ask(response(text_block("I don't know.")), question="Is there a nonprofit discount?")
    req = client.requests[0]

    assert req["model"] == MODEL
    assert "ONLY the help-center articles" in req["system"]
    assert req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    content = req["messages"][0]["content"]
    assert [b["type"] for b in content] == ["document", "document", "text"]
    assert content[-1]["text"] == "Is there a nonprofit discount?"


def test_cost_is_computed_from_usage():
    answer, _ = ask(response(
        text_block("I don't know."),
        usage_=usage(input_tokens=100, output_tokens=50, cache_read=10_000, cache_write=0),
    ))

    # 100*$4 + 50*$20 + 10,000*$0.20 per million tokens
    assert abs(answer.cost_usd - 0.0034) < 1e-9
