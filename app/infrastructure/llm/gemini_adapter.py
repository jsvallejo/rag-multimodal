"""
Adaptador concreto para Google Gemini. Implementa el mismo puerto que
OllamaAdapter y OpenAIAdapter — intercambiable sin tocar ningún caso de uso
(ver core/container.py). Requiere el paquete `google-genai`.

Ventaja frente a un modelo local en CPU: es una API en la nube, así que las
respuestas tardan segundos en vez de minutos, y su soporte de visión es
mucho más sólido que modelos livianos como moondream.
"""
import base64
import logging

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from app.domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


def _is_transient_error(exc: BaseException) -> bool:
    """Solo reintenta errores de servidor (5xx) o límites de tasa; un error
    4xx de la petición (ej. imagen rechazada) no mejora reintentando igual."""
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError):
        return getattr(exc, "code", None) == 429  # rate limit
    return False


class GeminiAdapter(LLMPort):
    def __init__(self, api_key: str, model: str, embedding_model: str):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._embedding_model = embedding_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_transient_error),
    )
    async def generate(
        self, system_prompt: str, user_prompt: str, images_base64: list[str] | None = None
    ) -> str:
        try:
            return await self._generate_content(system_prompt, user_prompt, images_base64)
        except ClientError as exc:
            # Igual que en los otros adaptadores: si el modelo rechaza la
            # imagen (400), se reintenta UNA vez solo con texto en vez de
            # fallar la consulta entera — mismo comportamiento defensivo
            # en los tres proveedores, ya que todos implementan LLMPort.
            if getattr(exc, "code", None) == 400 and images_base64:
                logger.warning(
                    "El modelo '%s' de Gemini rechazó las imágenes (400) — "
                    "reintentando solo con texto.",
                    self._model,
                )
                return await self._generate_content(system_prompt, user_prompt, images_base64=None)
            raise

    async def _generate_content(
        self, system_prompt: str, user_prompt: str, images_base64: list[str] | None
    ) -> str:
        parts: list = [user_prompt]
        if images_base64:
            for img_b64 in images_base64:
                parts.append(
                    types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/png")
                )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=parts,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_transient_error),
    )
    async def embed(self, text: str) -> list[float]:
        response = await self._client.aio.models.embed_content(
            model=self._embedding_model, contents=text
        )
        return response.embeddings[0].values