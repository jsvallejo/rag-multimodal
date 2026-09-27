from abc import ABC, abstractmethod


class ImageLoaderPort(ABC):
    @abstractmethod
    def load_as_base64(self, image_filename: str) -> str | None:
        """
        Devuelve el contenido de la imagen codificado en base64 (sin el
        prefijo data:image/...), o None si el archivo no existe.
        """
        raise NotImplementedError