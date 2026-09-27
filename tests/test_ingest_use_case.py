"""
Tests del caso de uso de ingesta, aislando completamente LLM, VectorStore y
Extractor reales mediante los mocks de conftest.py. Esto permite validar la
LÓGICA (orquestación, manejo de estados, manejo de errores) sin depender de
red, GPU, ni archivos PDF reales.
"""
import pytest

from app.domain.entities import DocumentChunk, IngestionJob, JobStatus
from app.domain.use_cases.ingest_document import IngestDocumentUseCase
from tests.conftest import FakeExtractor, FakeLLM, FakeVectorStore


@pytest.mark.asyncio
async def test_ingest_success_marks_job_completed(job_store, sample_chunk):
    job = IngestionJob.new(filename="manual.pdf")
    job_store.create(job)

    extractor = FakeExtractor(chunks=[sample_chunk], images=[])
    llm = FakeLLM()
    vector_store = FakeVectorStore()

    use_case = IngestDocumentUseCase(
        extractor=extractor, vector_store=vector_store, llm=llm, job_store=job_store
    )

    await use_case.execute(job.job_id, "fake/path.pdf")

    updated_job = job_store.get(job.job_id)
    assert updated_job.status == JobStatus.COMPLETED
    assert updated_job.document_id is not None
    assert updated_job.error_message is None


@pytest.mark.asyncio
async def test_ingest_generates_embeddings_for_every_chunk(job_store, sample_chunk):
    job = IngestionJob.new(filename="manual.pdf")
    job_store.create(job)

    extractor = FakeExtractor(chunks=[sample_chunk, sample_chunk], images=[])
    llm = FakeLLM()
    vector_store = FakeVectorStore()

    use_case = IngestDocumentUseCase(
        extractor=extractor, vector_store=vector_store, llm=llm, job_store=job_store
    )
    await use_case.execute(job.job_id, "fake/path.pdf")

    assert len(llm.embed_calls) == 2
    assert len(vector_store.added_chunks) == 2


@pytest.mark.asyncio
async def test_ingest_failure_marks_job_failed_with_message(job_store):
    job = IngestionJob.new(filename="corrupto.pdf")
    job_store.create(job)

    class BrokenExtractor:
        def extract(self, file_path, document_id):
            raise ValueError("PDF corrupto o ilegible")

    use_case = IngestDocumentUseCase(
        extractor=BrokenExtractor(), vector_store=FakeVectorStore(),
        llm=FakeLLM(), job_store=job_store,
    )

    await use_case.execute(job.job_id, "fake/corrupto.pdf")

    updated_job = job_store.get(job.job_id)
    assert updated_job.status == JobStatus.FAILED
    assert "corrupto" in updated_job.error_message.lower()


@pytest.mark.asyncio
async def test_ingest_preserves_related_image_ids(job_store, sample_chunk):
    """Verifica que la correlación texto-imagen sobreviva hasta el almacenamiento final."""
    job = IngestionJob.new(filename="manual.pdf")
    job_store.create(job)

    extractor = FakeExtractor(chunks=[sample_chunk], images=[])
    vector_store = FakeVectorStore()

    use_case = IngestDocumentUseCase(
        extractor=extractor, vector_store=vector_store, llm=FakeLLM(), job_store=job_store
    )
    await use_case.execute(job.job_id, "fake/path.pdf")

    assert vector_store.added_chunks[0].related_image_ids == ["img-1"]
