from pathlib import Path

from docs_qa import load_corpus
from evals.run_eval import load_questions
from evals.scoring import Question, Result, compute_metrics, grade

ROOT = Path(__file__).parent.parent


def test_sample_questions_are_consistent_with_the_corpus():
    questions = load_questions(ROOT / "evals" / "questions.jsonl")
    slugs = {a.slug for a in load_corpus(ROOT / "docs")}

    assert len(questions) == 30
    assert len({q.id for q in questions}) == 30
    for q in questions:
        if q.answerable:
            assert q.expected_article in slugs, q.id
            assert q.reference_answer, q.id
        else:
            assert q.expected_article is None, q.id


def test_grade_unanswerable_needs_idk():
    q = Question("x", "?", answerable=False)

    assert grade(q, "I don't know.", [], True, None) == (True, False)
    assert grade(q, "Yes.", ["taxes"], False, None) == (False, False)


def test_grade_answerable_needs_answer_citation_and_judge():
    q = Question("x", "?", answerable=True, expected_article="taxes", reference_answer="3")

    assert grade(q, "3", ["taxes"], False, True) == (True, True)
    assert grade(q, "3", ["refunds"], False, True) == (False, False)  # wrong source
    assert grade(q, "4", ["taxes"], False, False) == (False, True)    # wrong fact
    assert grade(q, "I don't know.", [], True, None) == (False, False)  # gave up


def _result(answerable, said_idk, correct, cited_expected=False, cost=0.01):
    return Result("id", "q", answerable, None, "", [], said_idk, cited_expected,
                  None, "", correct, cost)


def test_metrics():
    results = [
        _result(True, False, True, cited_expected=True),   # answered correctly
        _result(True, True, False),                        # wrong "I don't know"
        _result(False, True, True),                        # correct "I don't know"
        _result(False, False, False),                      # guessed
    ]
    m = compute_metrics(results)

    assert m["accuracy"] == 0.5
    assert m["answerable_accuracy"] == 0.5
    assert m["idk_when_not_covered"] == 0.5
    assert m["idk_precision"] == 0.5
    assert m["false_idk"] == 1
    assert m["citation_hit_rate"] == 1.0
    assert m["avg_cost_per_answer_usd"] == 0.01
