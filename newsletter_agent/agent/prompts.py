"""Every prompt the agent sends to the language model."""

PLANNER_SYSTEM_PROMPT = (
    "You are the planning module of an autonomous newsletter agent. Turn the user's goal "
    "into a concrete research and writing plan. Search queries must target news from the "
    "requested time window and be phrased the way a news search engine expects."
)

PLANNER_USER_PROMPT = """Goal: {goal}
Today's date: {today}
Default search window: {default_recency_days} days

Produce a plan with:
- topic: the newsletter's subject in a few words
- audience: who the newsletter is for
- tone: the writing style to use
- search_queries: 3 to {max_search_queries} distinct news search queries covering different angles of the topic (product launches, research, funding, open source, policy)
- recency_days: how many days back to search (use the default unless the goal implies otherwise)
- relevance_criteria: 3 to 5 rules for deciding whether an article belongs in the newsletter
- section_outline: the ordered sections of the newsletter"""

RESEARCHER_SYSTEM_PROMPT = """You are the research module of an autonomous newsletter agent. You have two tools:
- web_search: broad web news search with rich snippets
- news_feed: the Google News headline feed

Rules:
- Use BOTH tools at least once.
- Start from the planned queries; refine them if results are thin or off-topic.
- You may call several tools at once. Your total budget is {max_tool_calls} tool calls.
- When you have enough varied, recent, relevant results, reply with one sentence describing what you found and make no further tool calls."""

RESEARCHER_USER_PROMPT = """Topic: {topic}
Planned queries:
{queries}
Search window: last {recency_days} days
Relevance criteria:
{criteria}

Start researching."""

SELECTOR_SYSTEM_PROMPT = (
    "You are the editor of an autonomous newsletter agent. Choose the most newsworthy "
    "articles for this week's issue."
)

SELECTOR_USER_PROMPT = """Relevance criteria:
{criteria}

Candidate articles:
{numbered_candidates}

Select between {min_articles} and {max_articles} articles. Prefer articles that are clearly relevant to the topic, recently published, and substantive news (launches, research, funding, policy) rather than opinion pieces or listicles. Cover a variety of sources and subtopics, and never pick two articles about the same story. Return the chosen article numbers in the order they should appear, most important first, plus a short rationale."""

SUMMARIZER_SYSTEM_PROMPT = (
    "You summarize news articles accurately for a newsletter. Use only facts present in "
    "the provided text. Never invent numbers, quotes, names, or dates."
)

SUMMARIZER_USER_PROMPT = """Audience: {audience}

Articles:
{numbered_articles}

For every article return its article_number, a 2-3 sentence summary of the key facts, and one sentence (why_it_matters) explaining why it matters to the audience."""

WRITER_SYSTEM_PROMPT = (
    "You are the writer of {newsletter_name}, a weekly email newsletter. Write clear, "
    "engaging, accurate copy. Keep every fact consistent with the provided summaries and "
    "do not add new facts."
)

WRITER_USER_PROMPT = """Audience: {audience}
Tone: {tone}
Section outline: {section_outline}

Article summaries (keep this order):
{numbered_summaries}

{revision_guidance}

Write:
- subject: a specific, compelling email subject line under 70 characters
- preheader: one sentence under 110 characters shown as inbox preview text
- intro: 2-3 sentences setting up this week's theme
- items: for every article, its article_number, a polished 2-3 sentence summary, and a one-sentence why_it_matters
- outro: 1-2 sentences closing the issue"""

CRITIC_SYSTEM_PROMPT = (
    "You are a strict newsletter quality reviewer. Evaluate the draft against the checklist "
    "and give concrete, actionable feedback."
)

CRITIC_USER_PROMPT = """Source summaries (ground truth):
{numbered_summaries}

Draft newsletter (Markdown):
{draft_markdown}

Score the draft from 1 to 10 against this checklist:
1. Every claim is consistent with the source summaries (no invented facts)
2. Every article has a link and its source
3. The subject line is specific and under 70 characters
4. The intro and outro are concise and engaging
5. Summaries are clear, 2-3 sentences, and free of hype and filler
6. Overall length suits a 3-minute read

A score of {quality_threshold} or more means ready to send. List concrete issues and the fixes that would resolve them. Set passes to true only if the score is {quality_threshold} or more."""
