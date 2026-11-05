"""Ask one question from the command line:

    python -m docs_qa "How do I refund a card payment?"
"""

import argparse

from . import DocsAgent, load_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the help-center articles a question.")
    parser.add_argument("question")
    parser.add_argument("--docs", default="docs", help="folder of .md articles")
    args = parser.parse_args()

    agent = DocsAgent(load_corpus(args.docs))
    answer = agent.ask(args.question)

    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for s in answer.sources:
            print(f"  - {s.title} > {s.section}  ({s.article}.md)")
    print(f"\nCost: ${answer.cost_usd:.4f}  ·  model: {answer.model}")


if __name__ == "__main__":
    main()
