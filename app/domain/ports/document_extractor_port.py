"""
Puerto para la extracción de contenido de documentos (PDF, etc.).
Permite cambiar PyMuPDF por Unstructured.io sin afectar el caso de uso de ingesta.
"""
from abc import ABC, abstractmethod
from app.domain.entities import DocumentChunk, ExtractedImage


class DocumentExtractorPort(ABC):
    @abstractmethod
    def extract(self, file_path: str, document_id: str) -> tuple[list[DocumentChunk], list[ExtractedImage]]:
        """
        Extrae texto (segmentado en chunks con metadata de página) e imágenes
        (con su bounding box) de un documento. La correlación texto-imagen se
        resuelve aquí mediante cercanía espacial en la página.
        """
        raise NotImplementedError
