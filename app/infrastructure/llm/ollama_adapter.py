"""
Adaptador concreto para Ollama (modelo local).
Implementa el puerto LLMPort - la lógica de negocio nunca importa este archivo directamente.
"""
import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from app.domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


def _is_transient_error(exc: BaseException) -> bool:
    """
    Solo reintenta errores TRANSITORIOS (red caída, timeout, o error 5xx del
    servidor). Un 4xx (ej. 400 "el modelo no soporta imágenes") es un error
    de la PETICIÓN, no algo que mejore reintentando la misma petición 3 veces
    — eso solo desperdicia minutos en CPU antes de fallar igual.
    """
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


class OllamaAdapter(LLMPort):
    def __init__(self, base_url: str, model: str, embedding_model: str):
        self._base_url = base_url.rstrip("/")
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
            return await self._chat(system_prompt, user_prompt, images_base64)
        except httpx.HTTPStatusError as exc:
            # Si el modelo configurado no soporta visión, Ollama responde 400
            # al recibir el campo "images". En vez de fallar la consulta
            # entera, se reintenta UNA vez solo con texto: la respuesta pierde
            # el aporte visual, pero el usuario igual recibe algo útil.
            if exc.response.status_code == 400 and images_base64:
                logger.warning(
                    "El modelo '%s' rechazó las imágenes (400) — reintentando solo con texto.",
                    self._model,
                )
                return await self._chat(system_prompt, user_prompt, images_base64=None)
            raise

    async def _chat(
        self, system_prompt: str, user_prompt: str, images_base64: list[str] | None
    ) -> str:
        user_message: dict = {"role": "user", "content": user_prompt}
        if images_base64:
            # Formato de Ollama para modelos con visión (ej. llama3.2-vision, llava):
            # el campo "images" del mensaje lleva la lista de imágenes en base64.
            user_message["images"] = images_base64

        # Timeout amplio: en CPU (sin GPU) la generación puede tardar varios minutos,
        # y más aún con imágenes (el encoder visual añade carga extra de cómputo).
        async with httpx.AsyncClient(timeout=420.0) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        user_message,
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_transient_error),
    )
    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._embedding_model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]