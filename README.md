# Docs Q&A agent with source citations and an eval script

A small Python agent that answers questions **only** from a folder of Markdown help-center articles, using the Claude API.

- Every answer cites the **article and section** it came from.
- When the articles don't cover the question, it says **"I don't know."** instead of guessing.
- An eval script runs a set of test questions and reports **accuracy**, how often it **correctly says "I don't know"**, and the **average cost per answer**.

The repo ships with a sample help center (24 articles for a fictional invoicing app, "Ledgerly") and 30 sample questions so you can run everything end to end. To use your own content, put your `.md` files in `docs/` and your questions in `evals/questions.jsonl`.

## Quick start

Requires Python 3.10+ and an Anthropic API key.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

set ANTHROPIC_API_KEY=sk-ant-...   # macOS/Linux: export ANTHROPIC_API_KEY=sk-ant-...
```

Ask a question:

```bash
python -m docs_qa "Do I get the Stripe fee back when I refund a customer?"
```

```
No, Stripe does not return its processing fee when you issue a refund.

Sources:
  - Refunding a payment > Fees on refunds  (refunds.md)

Cost: $0.0xxx  ·  model: claude-opus-5-5
```

Run the eval:

```bash
python -m evals.run_eval
```

This writes `results/eval_report.md` (summary + per-question table) and `results/eval_results.json` (full answers, citations and grades).

Run the unit tests (no API key needed):

```bash
pytest
```

## Retrieval setup, and why

**The whole help center goes into every request, and is cached.** There is no vector database and no chunk search.

For a help center of ~60 articles this fits easily in Claude's context window. Compared with a classic retrieve-top-k pipeline:

- **Nothing is missed by retrieval.** The most common RAG failure is the search step returning the wrong chunks, after which the model either guesses or gives up. Here the model always sees every article, so "I don't know" really means "it isn't in the articles".
- **Cheap, thanks to prompt caching.** The articles are a fixed prefix of the request with a cache breakpoint on the last one. After the first question, the articles are read from the cache at a fraction of the normal input price; only the question and the answer are billed at full price.
- **Exact citations without fuzzy matching.** Each article is sent as a document and each `##` section as one block inside it. Claude's built-in citations return *(document, block)*, which maps straight to *(article, section)*.
- **Less to run and maintain.** No embedding model, no index to rebuild when an article changes.

**When to change it:** if the help center grows to several hundred articles, or articles change constantly, add a retrieval step (BM25 or embeddings) that picks the top ~20 articles per question and keep everything else as is. `DocsAgent` only needs the list of articles it should send.

## How it avoids answering when the articles don't cover the question

Two layers:

1. **Instruction.** The system prompt allows answers only from the articles and requires exactly "I don't know." otherwise, with no outside knowledge and no guessing (`docs_qa/agent.py`).
2. **Enforced in code.** An answer is shown only if it carries at least one citation to the articles. An uncited answer is not grounded in the articles, so it is replaced with "I don't know." before it reaches the user. Partial answers are allowed, but must say which part the articles don't cover.

The eval measures both directions: how often it says "I don't know" on questions the articles don't cover, and how often it wrongly gives up on questions they do cover.

## Eval metrics

| Metric | Meaning |
|---|---|
| Accuracy (all questions) | Share of questions handled correctly (right answer, or a correct "I don't know") |
| Accuracy on covered questions | Correct answer **and** the expected article was cited |
| Says "I don't know" when not covered | Of the questions the articles don't cover, how often it said "I don't know" |
| "I don't know" precision | Of all "I don't know" replies, how many were justified |
| Cited the expected article | For answered, covered questions |
| Average cost per answer | From the API's token usage, including cache reads/writes (`docs_qa/pricing.py`) |

**Grading:** a covered question counts as correct only if the agent answered, cited the article listed in `expected_article`, and an LLM judge finds the answer consistent with `reference_answer`. A not-covered question counts as correct only if the agent said "I don't know". The judge's cost is reported separately and is not part of the cost per answer.

### Question file format

One JSON object per line in `evals/questions.jsonl`:

```json
{"id": "q17", "question": "Do I get the Stripe fee back when I refund a customer?", "answerable": true, "expected_article": "refunds", "reference_answer": "No, Stripe does not return its processing fee on refunds."}
{"id": "q24", "question": "Is there a discount for nonprofits?", "answerable": false, "expected_article": null, "reference_answer": null}
```

`expected_article` is the file name without `.md`.

## Model and settings

- Model: `claude-opus-5-5`, effort `medium` (configurable in `DocsAgent`).
- Server-side refusal fallback is enabled (`fallbacks="default"`), so a rare safety refusal is retried on a suitable model instead of failing; a refusal that still happens is reported as "I don't know".
- Prices used for the cost report are in `docs_qa/pricing.py`; update them if Anthropic's prices change.

## Project layout

```
docs/                    help-center articles (.md), one file per article
docs_qa/corpus.py        loads articles, splits sections, builds citable document blocks
docs_qa/agent.py         the agent: prompt, request, citations -> sources, "I don't know" rule
docs_qa/pricing.py       cost per request from token usage
docs_qa/__main__.py      command line: python -m docs_qa "question"
evals/questions.jsonl    test questions
evals/scoring.py         grading rules and metrics
evals/run_eval.py        runs the eval, writes results/
tests/                   unit tests (no API calls)
```
