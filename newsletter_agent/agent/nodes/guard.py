"""Wraps graph nodes so any failure names the step that failed."""

import functools
from collections.abc import Callable
from typing import Any

from langgraph.errors import GraphBubbleUp

from newsletter_agent.errors import NodeExecutionError

NodeFunction = Callable[..., dict[str, Any]]


def guard_node(step_name: str, node_function: NodeFunction) -> NodeFunction:
    """Return a node that re-raises failures as NodeExecutionError(step_name, error).

    LangGraph control-flow exceptions (such as interrupts) pass through untouched, and
    functools.wraps keeps the signature so LangGraph still injects `config` when requested.
    """

    @functools.wraps(node_function)
    def guarded_node(*node_arguments: Any, **node_keyword_arguments: Any) -> dict[str, Any]:
        try:
            return node_function(*node_arguments, **node_keyword_arguments)
        except (GraphBubbleUp, NodeExecutionError):
            raise
        except Exception as node_error:
            raise NodeExecutionError(step_name, node_error) from node_error

    return guarded_node
