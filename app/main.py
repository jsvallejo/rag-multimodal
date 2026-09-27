import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import ingest, query
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Sistema RAG Multimodal - procesamiento de PDFs técnicos con texto, tablas e imágenes.",
    version="1.0.0",
)

app.include_router(ingest.router)
app.include_router(query.router)

# Sirve las imágenes extraídas para que el frontend pueda renderizarlas junto a las respuestas
app.mount("/images", StaticFiles(directory=settings.IMAGES_DIR), name="images")


@app.get("/health")
async def health():
    return {"status": "ok", "llm_provider": settings.LLM_PROVIDER}
