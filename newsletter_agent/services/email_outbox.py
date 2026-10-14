"""Simulated email delivery: writes real .eml files to an outbox directory."""

import re
from collections.abc import Sequence
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Protocol

from newsletter_agent.domain.models import DeliveryReport, NewsletterDraft, Subscriber

UNSAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")
HTML_FILENAME = "newsletter.html"


def safe_filename(raw_name: str) -> str:
    """Replace every character that is unsafe in a file name with an underscore."""
    return UNSAFE_FILENAME_PATTERN.sub("_", raw_name)


class EmailSender(Protocol):
    """Anything that can deliver a newsletter to recipients."""

    def send(
        self, draft: NewsletterDraft, recipients: Sequence[Subscriber], batch_name: str
    ) -> DeliveryReport:
        """Deliver the draft and report what was sent."""
        ...


class EmlFileOutbox:
    """Writes one .eml file per recipient plus the HTML issue into a per-run folder."""

    def __init__(self, outbox_directory: Path, sender_address: str) -> None:
        self._outbox_directory = outbox_directory
        self._sender_address = sender_address

    def send(
        self, draft: NewsletterDraft, recipients: Sequence[Subscriber], batch_name: str
    ) -> DeliveryReport:
        """Write the newsletter files and return a delivery report."""
        batch_directory = self._outbox_directory / safe_filename(batch_name)
        batch_directory.mkdir(parents=True, exist_ok=True)
        html_path = batch_directory / HTML_FILENAME
        html_path.write_text(draft.html_body, encoding="utf-8")
        outbox_paths: list[str] = []
        for recipient in recipients:
            eml_path = batch_directory / f"{safe_filename(recipient.email)}.eml"
            eml_path.write_bytes(self._build_message(draft, recipient).as_bytes())
            outbox_paths.append(str(eml_path))
        return DeliveryReport(
            recipient_count=len(recipients),
            subject=draft.subject,
            outbox_paths=outbox_paths,
            html_path=str(html_path),
        )

    def _build_message(self, draft: NewsletterDraft, recipient: Subscriber) -> EmailMessage:
        message = EmailMessage()
        message["Subject"] = draft.subject
        message["From"] = self._sender_address
        message["To"] = formataddr((recipient.name, recipient.email))
        message.set_content(draft.markdown_body)
        message.add_alternative(draft.html_body, subtype="html")
        return message
