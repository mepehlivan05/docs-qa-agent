"""Grading rules and metrics. Pure functions, so they are unit-tested offline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Question:
    id: str
    question: str
    answerable: bool
    expected_article: str | None = None
    reference_answer: str | None = None


@dataclass
class Result:
    id: str
    question: str
    answerable: bool
    expected_article: str | None
    answer: str
    cited_articles: list[str]
    said_idk: bool
    cited_expected: bool
    judge_correct: bool | None  # None when no judge call was needed
    judge_reason: str
    correct: bool
    cost_usd: float

    def to_dict(self) -> dict:
        return asdict(self)


def grade(q: Question, answer_text: str, cited: list[str], said_idk: bool,
          judge_correct: bool | None) -> tuple[bool, bool]:
    """Returns (correct, cited_expected).

    - Not covered by the articles: correct only if the agent said "I don't know".
    - Covered: correct only if the agent answered (no "I don't know"), cited the
      expected article, and the judge found the answer consistent with the
      reference answer.
    """
    cited_expected = q.expected_article in cited if q.expected_article else False
    if not q.answerable:
        return said_idk, cited_expected
    return (not said_idk and cited_expected and bool(judge_correct)), cited_expected


def compute_metrics(results: list[Result]) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    idk = [r for r in results if r.said_idk]
    answered = [r for r in answerable if not r.said_idk]

    def ratio(n: int, d: int) -> float | None:
        return round(n / d, 4) if d else None

    return {
        "questions": len(results),
        "accuracy": ratio(sum(r.correct for r in results), len(results)),
        "answerable_accuracy": ratio(sum(r.correct for r in answerable), len(answerable)),
        # How often it correctly says "I don't know" when the articles don't cover it.
        "idk_when_not_covered": ratio(len([r for r in unanswerable if r.said_idk]), len(unanswerable)),
        # How often an "I don't know" was right (low = it gives up on covered questions).
        "idk_precision": ratio(len([r for r in idk if not r.answerable]), len(idk)),
        "false_idk": len([r for r in answerable if r.said_idk]),
        "citation_hit_rate": ratio(sum(r.cited_expected for r in answered), len(answered)),
        "avg_cost_per_answer_usd": round(sum(r.cost_usd for r in results) / len(results), 5)
        if results else None,
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
    }
