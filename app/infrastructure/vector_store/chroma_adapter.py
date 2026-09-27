"""
Adaptador para ChromaDB.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.domain.entities import DocumentChunk
from app.domain.ports.vector_store_port import VectorStorePort

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3


class ChromaVectorStore(VectorStorePort):
    def __init__(self, host: str, port: int, collection_name: str):
        # HttpClient: Chroma corre como SERVICIO APARTE en docker-compose (no
        # embebido en el proceso del backend). Esto refleja mejor un despliegue
        # real, donde la base vectorial es infraestructura independiente que
        # podría escalarse, respaldarse o compartirse entre múltiples backends
        # sin acoplarla al ciclo de vida de la API.
        self._client = chromadb.HttpClient(
            host=host, port=port, settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Se fuerza espacio "cosine": la distancia queda acotada en [0, 2], lo que
        # permite convertirla a un score de similitud interpretable (1 - distancia).
        # Por defecto Chroma usa L2 al cuadrado, cuya escala es arbitraria y puede
        # dar scores negativos sin sentido al restarla de 1.
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "page_number": c.page_number,
                    "source_file": c.source_file,
                    "related_image_ids": ",".join(c.related_image_ids),
                }
                for c in chunks
            ],
        )

    async def search(
        self, query_embedding: list[float], query_text: str, top_k: int
    ) -> list[tuple[DocumentChunk, float]]:
        # Sobre-recuperamos para tener margen al re-rankear con keyword matching
        raw = self._collection.query(query_embeddings=[query_embedding], n_results=top_k * 3)

        if not raw["ids"] or not raw["ids"][0]:
            return []

        keywords = {w.lower() for w in query_text.split() if len(w) > 3}
        scored: list[tuple[DocumentChunk, float]] = []

        for i, chunk_id in enumerate(raw["ids"][0]):
            document_text = raw["documents"][0][i]
            metadata = raw["metadatas"][0][i]
            distance = raw["distances"][0][i]
            # Con espacio "cosine", distance está en [0, 2]; se acota el score a [0, 1] por seguridad.
            semantic_score = max(0.0, min(1.0, 1.0 - distance))

            keyword_hits = sum(1 for kw in keywords if kw in document_text.lower())
            keyword_score = min(keyword_hits / max(len(keywords), 1), 1.0)

            combined_score = (SEMANTIC_WEIGHT * semantic_score) + (KEYWORD_WEIGHT * keyword_score)

            related_images = metadata.get("related_image_ids", "")
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=metadata["document_id"],
                text=document_text,
                page_number=metadata["page_number"],
                source_file=metadata["source_file"],
                related_image_ids=related_images.split(",") if related_images else [],
            )
            scored.append((chunk, combined_score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]