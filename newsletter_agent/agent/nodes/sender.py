"""Sender node: delivers the approved newsletter through the send_email tool."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from newsletter_agent.agent.node_names import SENDER
from newsletter_agent.agent.state import NewsletterState
from newsletter_agent.domain.models import DeliveryReport, Subscriber, TraceEntry


def build_sender_node(
    send_email_tool: BaseTool, subscribers: Sequence[Subscriber]
) -> Callable[[NewsletterState, RunnableConfig], dict[str, Any]]:
    """Create the sender node."""

    def sender_node(state: NewsletterState, config: RunnableConfig) -> dict[str, Any]:
        batch_name = str(config["configurable"]["thread_id"])
        delivery_report: DeliveryReport = send_email_tool.invoke(
            {"draft": state["draft"], "recipients": list(subscribers), "batch_name": batch_name}
        )
        trace_message = (
            f"Sent '{delivery_report.subject}' to {delivery_report.recipient_count} subscribers "
            "(simulated: .eml files written to the outbox)"
        )
        return {
            "delivery_report": delivery_report,
            "trace_log": [TraceEntry(step=SENDER, message=trace_message)],
        }

    return sender_node
