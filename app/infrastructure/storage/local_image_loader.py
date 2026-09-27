"""
Adaptador concreto de ImageLoaderPort: lee archivos del filesystem local.
Si en el futuro las imágenes se movieran a S3/GCS, solo este archivo cambia.
"""
import base64
import logging
import os

from app.domain.ports.image_loader_port import ImageLoaderPort

logger = logging.getLogger(__name__)


class LocalImageLoader(ImageLoaderPort):
    def __init__(self, images_dir: str):
        self._images_dir = images_dir

    def load_as_base64(self, image_filename: str) -> str | None:
        path = os.path.join(self._images_dir, image_filename)
        if not os.path.isfile(path):
            logger.warning("Imagen no encontrada para envío al LLM: %s", path)
            return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")