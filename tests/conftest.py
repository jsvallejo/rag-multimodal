"""
Fixtures compartidos. Los mocks de LLM y VectorStore viven aquí para
reutilizarse entre test_ingest_use_case.py y test_query_use_case.py.
"""
import pytest

from app.domain.ports.llm_port import LLMPort
from app.domain.ports.vector_store_port import VectorStorePort
from app.domain.ports.document_extractor_port import DocumentExtractorPort
from app.domain.ports.image_loader_port import ImageLoaderPort
from app.domain.entities import DocumentChunk, ExtractedImage
from app.infrastructure.job_store import JobStore


class FakeLLM(LLMPort):
    """Mock de LLM: no llama a ninguna API real, devuelve respuestas fijas y predecibles."""

    def __init__(self, fixed_answer: str = "Respuesta simulada del LLM."):
        self.fixed_answer = fixed_answer
        self.embed_calls: list[str] = []
        self.generate_calls: list[tuple[str, str]] = []
        self.last_images: list[str] | None = None

    async def generate(
        self, system_prompt: str, user_prompt: str, images_base64: list[str] | None = None
    ) -> str:
        self.generate_calls.append((system_prompt, user_prompt))
        self.last_images = images_base64
        return self.fixed_answer

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [float(len(text) % 10)] * 8


class FakeVectorStore(VectorStorePort):
    """Mock de la base vectorial: guarda en memoria, sin depender de Chroma real."""

    def __init__(self, search_results: list[tuple[DocumentChunk, float]] | None = None):
        self.added_chunks: list[DocumentChunk] = []
        self._search_results = search_results or []

    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.added_chunks.extend(chunks)

    async def search(self, query_embedding, query_text, top_k):
        return self._search_results[:top_k]


class FakeExtractor(DocumentExtractorPort):
    """Mock del extractor: devuelve chunks/imagenes fijos sin abrir ningún PDF real."""

    def __init__(self, chunks: list[DocumentChunk], images: list[ExtractedImage]):
        self._chunks = chunks
        self._images = images

    def extract(self, file_path, document_id):
        return self._chunks, self._images


class FakeImageLoader(ImageLoaderPort):
    """Mock del cargador de imágenes: no toca el filesystem real."""

    def __init__(self, contents: dict[str, str] | None = None):
        self._contents = contents or {}

    def load_as_base64(self, image_filename: str) -> str | None:
        return self._contents.get(image_filename)


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="El motor XJ200 requiere revisión cada 5000 km según el manual técnico.",
        page_number=12,
        source_file="Manual_Motor.pdf",
        related_image_ids=["img-1"],
    )


@pytest.fixture
def job_store() -> JobStore:
    return JobStore()