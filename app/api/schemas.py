from pydantic import BaseModel


class IngestResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    error_message: str | None = None
    document_id: str | None = None


class QueryRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    source_file: str
    page_number: int
    text_snippet: str
    score: float
    image_urls: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    insufficient_context: bool
    sources: list[SourceItem]
