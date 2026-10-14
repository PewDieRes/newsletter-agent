"""Schemas the language model fills in through structured output."""

from pydantic import BaseModel, Field


class ArticleSelection(BaseModel):
    """The editor's choice of articles for the issue."""

    selected_article_numbers: list[int] = Field(
        description="Numbers of the chosen articles, most important first"
    )
    rationale: str = Field(description="One or two sentences explaining the choice")


class ArticleCopy(BaseModel):
    """Newsletter copy for one numbered article."""

    article_number: int = Field(description="The article's number from the provided list")
    summary: str = Field(description="Two or three sentences covering the key facts")
    why_it_matters: str = Field(description="One sentence on why the audience should care")


class SummaryBatch(BaseModel):
    """Summaries for every selected article."""

    items: list[ArticleCopy]


class NewsletterCopy(BaseModel):
    """All the copy needed to assemble one newsletter issue."""

    subject: str = Field(description="Email subject line under 70 characters")
    preheader: str = Field(description="Inbox preview text under 110 characters")
    intro: str = Field(description="Two or three sentences introducing the issue")
    items: list[ArticleCopy]
    outro: str = Field(description="One or two sentences closing the issue")
