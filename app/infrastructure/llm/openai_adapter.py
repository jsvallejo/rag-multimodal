"""
Adaptador concreto para OpenAI. Implementa el mismo puerto que OllamaAdapter,
por eso son intercambiables sin tocar ningún caso de uso (ver core/container.py).
"""
import logging

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, BadRequestError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)

RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)


class OpenAIAdapter(LLMPort):
    def __init__(self, api_key: str, model: str, embedding_model: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._embedding_model = embedding_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    async def generate(
        self, system_prompt: str, user_prompt: str, images_base64: list[str] | None = None
    ) -> str:
        try:
            return await self._chat(system_prompt, user_prompt, images_base64)
        except BadRequestError:
            # Igual que en OllamaAdapter: si el modelo configurado no admite
            # contenido multimodal (o rechaza la imagen por otro motivo),
            # se reintenta UNA vez solo con texto en vez de fallar la consulta
            # entera. Mantiene el mismo comportamiento defensivo entre
            # proveedores, ya que ambos implementan el mismo LLMPort.
            if images_base64:
                logger.warning(
                    "El modelo '%s' de OpenAI rechazó las imágenes (400) — "
                    "reintentando solo con texto.",
                    self._model,
                )
                return await self._chat(system_prompt, user_prompt, images_base64=None)
            raise

    async def _chat(
        self, system_prompt: str, user_prompt: str, images_base64: list[str] | None
    ) -> str:
        if images_base64:
            # Formato de OpenAI (gpt-4o, gpt-4o-mini) para contenido multimodal:
            # el "content" del mensaje se vuelve una lista de bloques de texto/imagen.
            content: list[dict] = [{"type": "text", "text": user_prompt}]
            for img_b64 in images_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            user_message = {"role": "user", "content": content}
        else:
            user_message = {"role": "user", "content": user_prompt}

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                user_message,
            ],
        )
        return response.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(model=self._embedding_model, input=text)
        return response.data[0].embedding