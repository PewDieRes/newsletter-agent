"""Terminal front end: runs the agent and asks the human for decisions when needed."""

import argparse
from collections.abc import Callable, Sequence
from typing import Any

from newsletter_agent.agent.interrupts import (
    ARTICLE_REVIEW_INTERRUPT,
    ArticleReviewResponse,
    DraftReviewResponse,
)
from newsletter_agent.agent.runner import NewsletterAgentRunner, get_default_runner, new_thread_id
from newsletter_agent.config import DEFAULT_GOAL
from newsletter_agent.errors import NewsletterAgentError

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], Any]


def parse_article_numbers(answer: str, candidate_count: int) -> list[int]:
    """Parse '2, 1, 5' into valid, unique 1-based article numbers, ignoring anything else."""
    article_numbers: list[int] = []
    for token in answer.replace(" ", "").split(","):
        if token.isdigit() and 1 <= int(token) <= candidate_count:
            article_number = int(token)
            if article_number not in article_numbers:
                article_numbers.append(article_number)
    return article_numbers


def ask_article_review(
    payload: dict[str, Any],
    input_function: InputFunction = input,
    output_function: OutputFunction = print,
) -> ArticleReviewResponse:
    """Show candidates (agent picks marked with *) and read the human's selection."""
    candidates = payload["candidates"]
    agent_selection = set(payload["selected_urls"])
    output_function("\nArticles chosen by the agent are marked with *:")
    for position, candidate in enumerate(candidates, start=1):
        marker = "*" if candidate["url"] in agent_selection else " "
        output_function(f" {marker} {position}. {candidate['title']} ({candidate['source_name']})")
    answer = input_function(
        "Article numbers to include (comma separated), or Enter to accept: "
    ).strip()
    if not answer:
        return {"selected_urls": list(payload["selected_urls"])}
    chosen_numbers = parse_article_numbers(answer, len(candidates))
    return {"selected_urls": [candidates[number - 1]["url"] for number in chosen_numbers]}


def ask_draft_review(
    payload: dict[str, Any],
    input_function: InputFunction = input,
    output_function: OutputFunction = print,
) -> DraftReviewResponse:
    """Show the draft and critique, then read approve / revise / reject."""
    draft = payload["draft"]
    critique = payload["critique"]
    output_function(f"\n{draft['markdown_body']}\n")
    output_function(f"Critique score: {critique['score']}/10")
    for issue in critique.get("issues", []):
        output_function(f"  - {issue}")
    while True:
        choice = input_function("[a]pprove and send, [r]equest revision, or [x] reject? ")
        normalized_choice = choice.strip().lower()
        if normalized_choice in ("a", "approve"):
            return {"decision": "approve"}
        if normalized_choice in ("r", "revise"):
            feedback = input_function("What should change? ").strip()
            return {"decision": "revise", "feedback": feedback}
        if normalized_choice in ("x", "reject"):
            return {"decision": "reject"}
        output_function("Please enter a, r, or x.")


def ask_human_decision(
    payload: dict[str, Any], input_function: InputFunction, output_function: OutputFunction
) -> dict[str, Any]:
    """Dispatch to the right prompt for the pending interrupt."""
    if payload["type"] == ARTICLE_REVIEW_INTERRUPT:
        return dict(ask_article_review(payload, input_function, output_function))
    return dict(ask_draft_review(payload, input_function, output_function))


def run_cli(
    argv: Sequence[str] | None = None,
    runner: NewsletterAgentRunner | None = None,
    input_function: InputFunction = input,
    output_function: OutputFunction = print,
) -> int:
    """Run the agent from the terminal. Returns a process exit code."""
    argument_parser = argparse.ArgumentParser(description="Run the Newsletter Agent.")
    argument_parser.add_argument("--goal", default=DEFAULT_GOAL)
    argument_parser.add_argument("--mode", choices=["auto", "hitl"], default="auto")
    parsed_arguments = argument_parser.parse_args(argv)

    active_runner = runner or get_default_runner()
    thread_id = new_thread_id()
    trace_stream = active_runner.stream_run(parsed_arguments.goal, parsed_arguments.mode, thread_id)
    while True:
        try:
            for trace_entry in trace_stream:
                output_function(f"[{trace_entry.step}] {trace_entry.message}")
        except NewsletterAgentError as agent_error:
            output_function(f"Run failed: {agent_error}")
            return 1
        result = active_runner.get_result(thread_id)
        if result.status != "awaiting_human":
            break
        human_decision = ask_human_decision(
            result.pending_interrupt, input_function, output_function
        )
        trace_stream = active_runner.stream_resume(thread_id, human_decision)

    if result.status == "completed":
        delivery_report = result.final_state["delivery_report"]
        output_function(
            f"\nSent to {delivery_report.recipient_count} subscribers: "
            f"'{delivery_report.subject}'\nHTML: {delivery_report.html_path}"
        )
        return 0
    if result.status == "rejected":
        output_function("\nDraft rejected. Nothing was sent.")
        return 0
    output_function(f"\nRun failed: {result.error_message}")
    return 1
