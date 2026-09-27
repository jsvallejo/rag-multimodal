from fastapi import APIRouter

from app.api.schemas import QueryRequest, QueryResponse, SourceItem
from app.core.container import get_query_use_case

router = APIRouter(prefix="/query", tags=["rag"])


@router.post("", response_model=QueryResponse)
async def query_documents(payload: QueryRequest):
    use_case = get_query_use_case()
    result = await use_case.execute(payload.question)

    sources = [
        SourceItem(
            source_file=ctx.chunk.source_file,
            page_number=ctx.chunk.page_number,
            text_snippet=ctx.chunk.text[:200],
            score=round(ctx.score, 4),
            image_urls=[f"/images/{img_id}" for img_id in ctx.chunk.related_image_ids],
        )
        for ctx in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        insufficient_context=result.insufficient_context,
        sources=sources,
    )
