from functools import lru_cache

from app.core.config import get_settings
from app.domain.ports.llm_port import LLMPort
from app.domain.use_cases.ingest_document import IngestDocumentUseCase
from app.domain.use_cases.query_document import QueryDocumentUseCase
from app.infrastructure.extraction.pymupdf_extractor import PyMuPDFExtractor
from app.infrastructure.job_store import JobStore
from app.infrastructure.llm.gemini_adapter import GeminiAdapter
from app.infrastructure.llm.ollama_adapter import OllamaAdapter
from app.infrastructure.llm.openai_adapter import OpenAIAdapter
from app.infrastructure.storage.local_image_loader import LocalImageLoader
from app.infrastructure.vector_store.chroma_adapter import ChromaVectorStore


@lru_cache
def get_llm() -> LLMPort:
    settings = get_settings()
    if settings.LLM_PROVIDER == "openai":
        return OpenAIAdapter(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            embedding_model=settings.OPENAI_EMBEDDING_MODEL,
        )
    if settings.LLM_PROVIDER == "gemini":
        return GeminiAdapter(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            embedding_model=settings.GEMINI_EMBEDDING_MODEL,
        )
    return OllamaAdapter(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        embedding_model=settings.OLLAMA_EMBEDDING_MODEL,
    )


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    settings = get_settings()
    return ChromaVectorStore(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        collection_name=settings.CHROMA_COLLECTION,
    )


@lru_cache
def get_extractor() -> PyMuPDFExtractor:
    settings = get_settings()
    return PyMuPDFExtractor(
        images_output_dir=settings.IMAGES_DIR,
        chunk_max_tokens=settings.CHUNK_MAX_TOKENS,
    )


@lru_cache
def get_job_store() -> JobStore:
    return JobStore()


@lru_cache
def get_image_loader() -> LocalImageLoader:
    settings = get_settings()
    return LocalImageLoader(images_dir=settings.IMAGES_DIR)


def get_ingest_use_case() -> IngestDocumentUseCase:
    return IngestDocumentUseCase(
        extractor=get_extractor(),
        vector_store=get_vector_store(),
        llm=get_llm(),
        job_store=get_job_store(),
    )


def get_query_use_case() -> QueryDocumentUseCase:
    settings = get_settings()
    return QueryDocumentUseCase(
        vector_store=get_vector_store(),
        llm=get_llm(),
        image_loader=get_image_loader(),
        top_k=settings.TOP_K,
    )