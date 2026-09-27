"""
Caso de uso: Ingesta de un documento.

Nótese que esta clase NO sabe nada de FastAPI, PyMuPDF o Chroma concretos:
solo conoce los puertos (interfaces). Esto es lo que la hace 100% testeable
con mocks, y lo que permite cambiar cualquier pieza de infraestructura
sin tocar esta lógica.
"""
import logging
import uuid

from app.domain.entities import IngestionJob, JobStatus
from app.domain.ports.document_extractor_port import DocumentExtractorPort
from app.domain.ports.llm_port import LLMPort
from app.domain.ports.vector_store_port import VectorStorePort
from app.infrastructure.job_store import JobStore

logger = logging.getLogger(__name__)


class IngestDocumentUseCase:
    def __init__(
        self,
        extractor: DocumentExtractorPort,
        vector_store: VectorStorePort,
        llm: LLMPort,
        job_store: JobStore,
    ):
        self._extractor = extractor
        self._vector_store = vector_store
        self._llm = llm
        self._job_store = job_store

    async def execute(self, job_id: str, file_path: str) -> None:
        """
        Se ejecuta en background (ver api/routes/ingest.py). Actualiza el
        estado del job en cada etapa para que el cliente pueda hacer polling.
        """
        job = self._job_store.get(job_id)
        try:
            job.status = JobStatus.PROCESSING
            self._job_store.update(job)

            document_id = str(uuid.uuid4())
            chunks, images = self._extractor.extract(file_path, document_id)
            logger.info(
                "Documento %s: %d chunks, %d imágenes extraídas",
                document_id, len(chunks), len(images),
            )

            # Generar embeddings para cada chunk (podría paralelizarse con asyncio.gather)
            for chunk in chunks:
                chunk.embedding = await self._llm.embed(chunk.text)

            await self._vector_store.add_chunks(chunks)

            job.status = JobStatus.COMPLETED
            job.document_id = document_id
            self._job_store.update(job)

        except Exception as exc:  # noqa: BLE001 - queremos capturar cualquier fallo y reflejarlo en el job
            logger.exception("Fallo en la ingesta del job %s", job_id)
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            self._job_store.update(job)
