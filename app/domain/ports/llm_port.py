from abc import ABC, abstractmethod


class LLMPort(ABC):
    @abstractmethod
    async def generate(
        self, system_prompt: str, user_prompt: str, images_base64: list[str] | None = None
    ) -> str:
        """
        Genera una respuesta de texto dado un prompt de sistema y uno de usuario.
        Si el modelo subyacente soporta visión, `images_base64` (imágenes
        codificadas en base64, sin prefijo data:) se adjunta al mensaje para
        que el LLM interprete su contenido visual, no solo el texto que las rodea.
        Un adaptador sin soporte de visión puede ignorar este parámetro.
        """
        raise NotImplementedError

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Genera el embedding vectorial de un texto."""
        raise NotImplementedError