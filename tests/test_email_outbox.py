import json
from email import message_from_bytes, policy
from pathlib import Path

from newsletter_agent.services.email_outbox import EmlFileOutbox, safe_filename
from newsletter_agent.services.subscribers import load_subscribers
from tests.fakes import TEST_SUBSCRIBERS, make_draft

SENDER_ADDRESS = "AI Agents Weekly <newsletter@example.com>"


def test_send_writes_html_and_one_eml_per_recipient(tmp_path: Path) -> None:
    outbox = EmlFileOutbox(tmp_path / "outbox", SENDER_ADDRESS)
    draft = make_draft()

    delivery_report = outbox.send(draft, TEST_SUBSCRIBERS, batch_name="run-1")

    batch_directory = tmp_path / "outbox" / "run-1"
    assert delivery_report.recipient_count == len(TEST_SUBSCRIBERS)
    assert delivery_report.subject == draft.subject
    assert Path(delivery_report.html_path) == batch_directory / "newsletter.html"
    assert Path(delivery_report.html_path).read_text(encoding="utf-8") == draft.html_body
    assert len(delivery_report.outbox_paths) == len(TEST_SUBSCRIBERS)
    assert sorted(eml_path.name for eml_path in batch_directory.glob("*.eml")) == [
        "ada_example.com.eml",
        "grace_example.com.eml",
        "linus_example.com.eml",
    ]


def test_eml_file_is_a_valid_multipart_email(tmp_path: Path) -> None:
    outbox = EmlFileOutbox(tmp_path, SENDER_ADDRESS)
    draft = make_draft()

    delivery_report = outbox.send(draft, TEST_SUBSCRIBERS[:1], batch_name="run-2")

    parsed_message = message_from_bytes(
        Path(delivery_report.outbox_paths[0]).read_bytes(), policy=policy.default
    )
    assert parsed_message["Subject"] == draft.subject
    assert parsed_message["From"] == SENDER_ADDRESS
    assert parsed_message["To"] == "Ada Example <ada@example.com>"
    assert parsed_message.get_body(preferencelist=("plain",)).get_content().strip() == (
        draft.markdown_body
    )
    assert "Draft" in parsed_message.get_body(preferencelist=("html",)).get_content()


def test_send_preserves_non_ascii_subject(tmp_path: Path) -> None:
    outbox = EmlFileOutbox(tmp_path, SENDER_ADDRESS)
    draft = make_draft().model_copy(update={"subject": "Agents weekly ✨ — édition"})

    delivery_report = outbox.send(draft, TEST_SUBSCRIBERS[:1], batch_name="run-3")

    parsed_message = message_from_bytes(
        Path(delivery_report.outbox_paths[0]).read_bytes(), policy=policy.default
    )
    assert parsed_message["Subject"] == "Agents weekly ✨ — édition"


def test_safe_filename_replaces_unsafe_characters() -> None:
    assert safe_filename("a+b@example.com") == "a_b_example.com"
    assert safe_filename("../../etc") == ".._.._etc"


def test_load_subscribers_reads_json_list(tmp_path: Path) -> None:
    subscribers_path = tmp_path / "subscribers.json"
    subscribers_path.write_text(
        json.dumps([{"name": "Ada", "email": "ada@example.com"}]), encoding="utf-8"
    )

    subscribers = load_subscribers(subscribers_path)

    assert [subscriber.email for subscriber in subscribers] == ["ada@example.com"]


def test_project_subscriber_file_loads() -> None:
    from newsletter_agent.config import Settings

    subscribers = load_subscribers(Settings(_env_file=None).subscribers_path)

    assert len(subscribers) == 5
