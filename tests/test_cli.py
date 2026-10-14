from pathlib import Path

from newsletter_agent.cli import (
    ask_article_review,
    ask_draft_review,
    parse_article_numbers,
    run_cli,
)
from tests.fakes import (
    FakeLanguageModel,
    build_test_dependencies,
    build_test_runner,
    default_structured_responses,
    search_both_tools_message,
)

ARTICLE_PAYLOAD = {
    "type": "article_review",
    "selected_urls": ["https://a.example.com"],
    "candidates": [
        {"title": "A", "url": "https://a.example.com", "source_name": "Alpha"},
        {"title": "B", "url": "https://b.example.com", "source_name": "Beta"},
    ],
}
DRAFT_PAYLOAD = {
    "type": "draft_review",
    "draft": {"subject": "Hello", "markdown_body": "# Hello"},
    "critique": {"score": 9, "issues": []},
}


def scripted_input(*answers: str):
    remaining_answers = list(answers)
    return lambda prompt_text: remaining_answers.pop(0)


def test_parse_article_numbers_ignores_invalid_tokens() -> None:
    assert parse_article_numbers("2, 1, 2, 9, x", candidate_count=3) == [2, 1]


def test_article_review_enter_keeps_agent_selection() -> None:
    response = ask_article_review(ARTICLE_PAYLOAD, scripted_input(""), lambda text: None)

    assert response == {"selected_urls": ["https://a.example.com"]}


def test_article_review_numbers_select_candidates() -> None:
    response = ask_article_review(ARTICLE_PAYLOAD, scripted_input("2"), lambda text: None)

    assert response == {"selected_urls": ["https://b.example.com"]}


def test_draft_review_reprompts_until_valid_then_revises() -> None:
    printed_lines: list[str] = []

    response = ask_draft_review(
        DRAFT_PAYLOAD, scripted_input("maybe", "r", "Shorter intro"), printed_lines.append
    )

    assert response == {"decision": "revise", "feedback": "Shorter intro"}
    assert "Please enter a, r, or x." in printed_lines


def test_draft_review_approve_and_reject() -> None:
    assert ask_draft_review(DRAFT_PAYLOAD, scripted_input("a"), lambda text: None) == {
        "decision": "approve"
    }
    assert ask_draft_review(DRAFT_PAYLOAD, scripted_input("x"), lambda text: None) == {
        "decision": "reject"
    }


def test_run_cli_auto_mode_succeeds(tmp_path: Path) -> None:
    language_model = FakeLanguageModel(
        default_structured_responses(), [search_both_tools_message()]
    )
    runner = build_test_runner(build_test_dependencies(tmp_path, language_model))
    printed_lines: list[str] = []

    exit_code = run_cli(["--mode", "auto"], runner=runner, output_function=printed_lines.append)

    assert exit_code == 0
    assert any(line.startswith("[planner]") for line in printed_lines)
    assert any("Sent to 3 subscribers" in line for line in printed_lines)


def test_run_cli_hitl_mode_uses_prompts(tmp_path: Path) -> None:
    language_model = FakeLanguageModel(
        default_structured_responses(), [search_both_tools_message()]
    )
    runner = build_test_runner(build_test_dependencies(tmp_path, language_model))

    exit_code = run_cli(
        ["--mode", "hitl"],
        runner=runner,
        input_function=scripted_input("", "a"),
        output_function=lambda text: None,
    )

    assert exit_code == 0
