"""Runs the test questions through the agent and writes a report.

    python -m evals.run_eval
    python -m evals.run_eval --docs path/to/articles --questions path/to/questions.jsonl

Outputs results/eval_results.json (per question) and results/eval_report.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from pydantic import BaseModel

from docs_qa import DocsAgent, load_corpus
from docs_qa.agent import FALLBACK_BETA, MODEL
from docs_qa.pricing import cost_usd

from .scoring import Question, Result, compute_metrics, grade

JUDGE_PROMPT = """You grade a support assistant's answer against a reference answer.

Question: {question}
Reference answer: {reference}
Assistant's answer: {answer}

The answer is correct if it states the same facts as the reference answer and contradicts nothing in it. Extra correct detail is fine. Missing the key fact, or stating a different fact, is incorrect."""


class Verdict(BaseModel):
    correct: bool
    reason: str


def load_questions(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def judge(client: anthropic.Anthropic, q: Question, answer: str) -> tuple[bool, str, float]:
    response = client.beta.messages.parse(
        model=MODEL,
        max_tokens=2000,
        output_config={"effort": "low"},
        betas=[FALLBACK_BETA],
        fallbacks="default",
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=q.question, reference=q.reference_answer, answer=answer
            ),
        }],
        output_format=Verdict,
    )
    cost = cost_usd(response.model, response.usage)
    verdict = response.parsed_output
    if verdict is None:  # e.g. refusal: count as incorrect rather than crash the run
        return False, f"judge returned no verdict (stop_reason={response.stop_reason})", cost
    return verdict.correct, verdict.reason, cost


def write_report(path: Path, metrics: dict, results: list[Result], judge_cost: float) -> None:
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    lines = [
        "# Eval report",
        "",
        f"Run: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · model `{MODEL}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Questions | {metrics['questions']} |",
        f"| **Accuracy (all questions)** | **{pct(metrics['accuracy'])}** |",
        f"| Accuracy on covered questions | {pct(metrics['answerable_accuracy'])} |",
        f"| **Says \"I don't know\" when not covered** | **{pct(metrics['idk_when_not_covered'])}** |",
        f"| \"I don't know\" precision | {pct(metrics['idk_precision'])} |",
        f"| Wrong \"I don't know\" on covered questions | {metrics['false_idk']} |",
        f"| Cited the expected article | {pct(metrics['citation_hit_rate'])} |",
        f"| **Average cost per answer** | **${metrics['avg_cost_per_answer_usd']:.4f}** |",
        f"| Total agent cost | ${metrics['total_cost_usd']:.4f} |",
        f"| Grading (judge) cost, not included above | ${judge_cost:.4f} |",
        "",
        "## Per question",
        "",
        "| | ID | Question | Cited | Note |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        mark = "✅" if r.correct else "❌"
        note = "I don't know" if r.said_idk else (r.judge_reason or "")
        cited = ", ".join(r.cited_articles) or "–"
        lines.append(f"| {mark} | {r.id} | {r.question} | {cited} | {note.replace('|', '/')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--questions", default="evals/questions.jsonl")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    agent = DocsAgent(load_corpus(args.docs), client=client)
    questions = load_questions(Path(args.questions))

    results: list[Result] = []
    judge_cost = 0.0
    for q in questions:
        answer = agent.ask(q.question)
        judge_correct, reason = None, ""
        if q.answerable and not answer.is_idk:
            judge_correct, reason, cost = judge(client, q, answer.text)
            judge_cost += cost
        correct, cited_expected = grade(q, answer.text, answer.articles, answer.is_idk, judge_correct)
        results.append(Result(
            id=q.id, question=q.question, answerable=q.answerable,
            expected_article=q.expected_article, answer=answer.text,
            cited_articles=answer.articles, said_idk=answer.is_idk,
            cited_expected=cited_expected, judge_correct=judge_correct,
            judge_reason=reason, correct=correct, cost_usd=answer.cost_usd,
        ))
        print(f"{'PASS' if correct else 'FAIL'}  {q.id}  ${answer.cost_usd:.4f}  {q.question}")

    metrics = compute_metrics(results)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "eval_results.json").write_text(
        json.dumps({"metrics": metrics, "judge_cost_usd": round(judge_cost, 4),
                    "results": [r.to_dict() for r in results]}, indent=2),
        encoding="utf-8",
    )
    write_report(out / "eval_report.md", metrics, results, judge_cost)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
