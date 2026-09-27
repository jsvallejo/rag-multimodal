from abc import ABC, abstractmethod
from app.domain.entities import DocumentChunk


class VectorStorePort(ABC):
    @abstractmethod
    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Persiste chunks junto con su embedding y metadata."""
        raise NotImplementedError

    @abstractmethod
    async def search(
        self, query_embedding: list[float], query_text: str, top_k: int
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Búsqueda híbrida: combina similitud semántica (embedding) con
        coincidencia de palabras clave (query_text) sobre el texto de los chunks.
        Devuelve tuplas (chunk, score) ordenadas por relevancia descendente.
        """
        raise NotImplementedError
