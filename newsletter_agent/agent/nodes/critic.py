"""Critic node: the agent's self-reflection step."""

from collections.abc import Callable
from typing import Any

from newsletter_agent.agent.node_names import CRITIC
from newsletter_agent.agent.nodes.formatting import format_summary_list
from newsletter_agent.agent.prompts import CRITIC_SYSTEM_PROMPT, CRITIC_USER_PROMPT
from newsletter_agent.agent.state import NewsletterState
from newsletter_agent.config import Settings
from newsletter_agent.domain.models import CritiqueReport, TraceEntry
from newsletter_agent.services.llm_gateway import LanguageModelGateway


def build_critic_node(
    language_model: LanguageModelGateway, settings: Settings
) -> Callable[[NewsletterState], dict[str, Any]]:
    """Create the critic node."""

    def critic_node(state: NewsletterState) -> dict[str, Any]:
        user_prompt = CRITIC_USER_PROMPT.format(
            numbered_summaries=format_summary_list(state["article_summaries"]),
            draft_markdown=state["draft"].markdown_body,
            quality_threshold=settings.quality_threshold,
        )
        raw_report = language_model.generate_structured(
            CRITIC_SYSTEM_PROMPT, user_prompt, CritiqueReport
        )
        report = raw_report.model_copy(
            update={"passes": raw_report.score >= settings.quality_threshold}
        )
        verdict = "ready to send" if report.passes else "needs revision"
        issue_summary = f"; issues: {'; '.join(report.issues)}" if report.issues else ""
        return {
            "critique_history": [report],
            "trace_log": [
                TraceEntry(
                    step=CRITIC, message=f"Score {report.score}/10, {verdict}{issue_summary}"
                )
            ],
        }

    return critic_node
