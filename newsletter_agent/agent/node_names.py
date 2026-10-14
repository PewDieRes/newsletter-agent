"""Graph node names, used for wiring, routing, and trace entries."""

PLANNER = "planner"
RESEARCHER = "researcher"
SELECTOR = "selector"
ARTICLE_REVIEW = "article_review"
SUMMARIZER = "summarizer"
WRITER = "writer"
CRITIC = "critic"
DRAFT_REVIEW = "draft_review"
SENDER = "sender"

PIPELINE_ORDER = (
    PLANNER,
    RESEARCHER,
    SELECTOR,
    ARTICLE_REVIEW,
    SUMMARIZER,
    WRITER,
    CRITIC,
    DRAFT_REVIEW,
    SENDER,
)
