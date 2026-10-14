"""Tool for (simulated) newsletter delivery."""

from langchain_core.tools import BaseTool, tool

from newsletter_agent.domain.models import DeliveryReport, NewsletterDraft, Subscriber
from newsletter_agent.services.email_outbox import EmailSender


def build_send_email_tool(email_sender: EmailSender) -> BaseTool:
    """Create the send_email tool bound to an email sender."""

    @tool("send_email")
    def send_email(
        draft: NewsletterDraft, recipients: list[Subscriber], batch_name: str
    ) -> DeliveryReport:
        """Send the newsletter to every recipient. Simulated: writes .eml files to the outbox."""
        return email_sender.send(draft, recipients, batch_name)

    return send_email
