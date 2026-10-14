"""Wires the newsletter agent's nodes and edges into a LangGraph StateGraph."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from newsletter_agent.agent.dependencies import AgentDependencies
from newsletter_agent.agent.node_names import (
    ARTICLE_REVIEW,
    CRITIC,
    DRAFT_REVIEW,
    PLANNER,
    RESEARCHER,
    SELECTOR,
    SENDER,
    SUMMARIZER,
    WRITER,
)
from newsletter_agent.agent.nodes.article_review import build_article_review_node
from newsletter_agent.agent.nodes.critic import build_critic_node
from newsletter_agent.agent.nodes.draft_review import build_draft_review_node
from newsletter_agent.agent.nodes.guard import guard_node
from newsletter_agent.agent.nodes.planner import build_planner_node
from newsletter_agent.agent.nodes.researcher import build_researcher_node
from newsletter_agent.agent.nodes.selector import build_selector_node
from newsletter_agent.agent.nodes.sender import build_sender_node
from newsletter_agent.agent.nodes.summarizer import build_summarizer_node
from newsletter_agent.agent.nodes.writer import build_writer_node
from newsletter_agent.agent.routing import build_route_after_critique, route_after_draft_review
from newsletter_agent.agent.state import NewsletterState
from newsletter_agent.services.news_aggregator import NewsAggregator
from newsletter_agent.tools.content_tools import build_content_tools
from newsletter_agent.tools.delivery_tools import build_send_email_tool
from newsletter_agent.tools.search_tools import build_search_tools


def build_newsletter_graph(
    dependencies: AgentDependencies, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    """Build and compile the newsletter agent graph."""
    settings = dependencies.settings
    language_model = dependencies.language_model
    search_tools = build_search_tools(dependencies.web_search_source, dependencies.news_feed_source)
    content_tools = build_content_tools(dependencies.article_fetcher, dependencies.renderer)
    send_email_tool = build_send_email_tool(dependencies.email_sender)
    aggregator = NewsAggregator([dependencies.web_search_source, dependencies.news_feed_source])

    node_functions = {
        PLANNER: build_planner_node(language_model, settings),
        RESEARCHER: build_researcher_node(language_model, search_tools, aggregator, settings),
        SELECTOR: build_selector_node(language_model, settings),
        ARTICLE_REVIEW: build_article_review_node(settings),
        SUMMARIZER: build_summarizer_node(language_model, content_tools.fetch_article, settings),
        WRITER: build_writer_node(
            language_model, content_tools.render_html, dependencies.renderer, settings
        ),
        CRITIC: build_critic_node(language_model, settings),
        DRAFT_REVIEW: build_draft_review_node(dependencies.renderer),
        SENDER: build_sender_node(send_email_tool, dependencies.subscribers),
    }

    graph_builder = StateGraph(NewsletterState)
    for node_name, node_function in node_functions.items():
        graph_builder.add_node(node_name, guard_node(node_name, node_function))

    graph_builder.add_edge(START, PLANNER)
    graph_builder.add_edge(PLANNER, RESEARCHER)
    graph_builder.add_edge(RESEARCHER, SELECTOR)
    graph_builder.add_edge(SELECTOR, ARTICLE_REVIEW)
    graph_builder.add_edge(ARTICLE_REVIEW, SUMMARIZER)
    graph_builder.add_edge(SUMMARIZER, WRITER)
    graph_builder.add_edge(WRITER, CRITIC)
    graph_builder.add_conditional_edges(
        CRITIC, build_route_after_critique(settings), [WRITER, DRAFT_REVIEW]
    )
    graph_builder.add_conditional_edges(
        DRAFT_REVIEW, route_after_draft_review, [WRITER, SENDER, END]
    )
    graph_builder.add_edge(SENDER, END)
    return graph_builder.compile(checkpointer=checkpointer)
