"""
Entidades de dominio puras. No dependen de FastAPI, Chroma, ni de ningún SDK.
Esto es lo que permite testear la lógica de negocio sin infraestructura real.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BoundingBox:
    """Coordenadas espaciales de un elemento dentro de una página del PDF."""
    x0: float
    y0: float
    x1: float
    y1: float
    page_number: int


@dataclass
class ExtractedImage:
    """Imagen extraída de un PDF, con su ubicación para poder correlacionarla con texto cercano."""
    image_id: str
    file_path: str
    bbox: BoundingBox
    page_number: int


@dataclass
class DocumentChunk:
    """
    Un fragmento de contenido indexable. Puede tener o no una imagen asociada
    (se decide en tiempo de ingesta según cercanía espacial en la página).
    """
    chunk_id: str
    document_id: str
    text: str
    page_number: int
    source_file: str
    related_image_ids: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None


@dataclass
class IngestionJob:
    """Estado de una tarea de ingesta asíncrona, consultable vía job_id."""
    job_id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    document_id: Optional[str] = None

    @staticmethod
    def new(filename: str) -> "IngestionJob":
        return IngestionJob(job_id=str(uuid.uuid4()), filename=filename)


@dataclass
class RetrievedContext:
    """Resultado de una búsqueda: el chunk más su(s) imagen(es) relacionada(s), listo para el LLM."""
    chunk: DocumentChunk
    score: float
    image_paths: list[str] = field(default_factory=list)


@dataclass
class RagAnswer:
    """Respuesta final que el caso de uso de consulta devuelve a la capa de API."""
    answer: str
    sources: list[RetrievedContext]
    insufficient_context: bool = False
