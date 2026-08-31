"""
ChunkRelevanceJudge — filters retrieved chunks by semantic relevance.

Runs upstream in execute_graphrag_search_flow() so the LLM only sees
relevant chunks and can only cite documents that are actually relevant.
"""

from graph.judges.base import BaseJudge
from graph.judges.prompts import CHUNK_RELEVANCE_PROMPT
from graph.judges.schemas import ChunkRelevanceEvaluation


class ChunkRelevanceJudge(BaseJudge):
    """Filters retrieved chunks by semantic relevance to the user's query.

    Uses a lightweight LLM call to evaluate whether each chunk actually
    answers the user's question, preventing keyword-only matches from
    being cited as sources.
    """

    def __init__(self, model: str | None = None):
        super().__init__(
            name="ChunkRelevanceJudge",
            prompt_template=CHUNK_RELEVANCE_PROMPT,
            evaluation_schema=ChunkRelevanceEvaluation,
            model=model,
            max_tokens=2048,
            temperature=0.0,
        )
