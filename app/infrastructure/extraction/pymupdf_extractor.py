"""
Extractor multimodal de PDFs usando PyMuPDF (fitz).

Estrategia de chunking: por BLOQUE DE TEXTO detectado por el layout del PDF
(no por número fijo de caracteres), agrupando bloques consecutivos hasta un
máximo de tokens aproximado. Esto respeta mejor la semántica del documento
(párrafos, títulos, listas) que un corte ciego cada N caracteres.

Correlación texto-imagen: se calcula la distancia vertical entre el bloque
de texto y cada imagen de la misma página; si está por debajo de un umbral,
se consideran relacionados (mapeo espacial vía bounding boxes).
"""
import os
import uuid

import pymupdf as fitz  # PyMuPDF (API moderna; el alias evita reescribir todo el módulo)

from app.domain.entities import BoundingBox, DocumentChunk, ExtractedImage
from app.domain.ports.document_extractor_port import DocumentExtractorPort

# Distancia vertical máxima (en puntos PDF) para considerar un texto "cercano" a una imagen
PROXIMITY_THRESHOLD = 150
APPROX_CHARS_PER_TOKEN = 4


class PyMuPDFExtractor(DocumentExtractorPort):
    def __init__(self, images_output_dir: str, chunk_max_tokens: int = 500):
        self._images_dir = images_output_dir
        self._chunk_max_chars = chunk_max_tokens * APPROX_CHARS_PER_TOKEN
        os.makedirs(self._images_dir, exist_ok=True)

    def extract(
        self, file_path: str, document_id: str
    ) -> tuple[list[DocumentChunk], list[ExtractedImage]]:
        doc = fitz.open(file_path)
        source_file = os.path.basename(file_path)

        all_images: list[ExtractedImage] = []
        all_chunks: list[DocumentChunk] = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1

            page_images = self._extract_images(doc, page, page_number, document_id)
            all_images.extend(page_images)

            page_chunks, chunk_bboxes = self._extract_text_chunks(
                page, page_number, document_id, source_file, page_images
            )
            self._ensure_every_image_has_a_chunk(page_chunks, chunk_bboxes, page_images)
            all_chunks.extend(page_chunks)

            table_chunks = self._extract_table_chunks(page, page_number, document_id, source_file)
            all_chunks.extend(table_chunks)

        doc.close()
        return all_chunks, all_images

    def _extract_images(self, doc, page, page_number: int, document_id: str) -> list[ExtractedImage]:
        extracted = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:  # noqa: BLE001 - imagen corrupta o no soportada, se ignora
                continue

            raw_id = str(uuid.uuid4())
            ext = base_image.get("ext", "png")
            # El "image_id" usado en todo el sistema es el NOMBRE DE ARCHIVO real
            # (no solo el uuid), para que la URL /images/<image_id> del static
            # server siempre coincida con el archivo guardado en disco.
            image_id = f"{document_id}_{raw_id}.{ext}"
            image_path = os.path.join(self._images_dir, image_id)
            with open(image_path, "wb") as f:
                f.write(base_image["image"])

            # bbox real de la imagen en la página (para mapeo espacial)
            rects = page.get_image_rects(xref)
            bbox_rect = rects[0] if rects else fitz.Rect(0, 0, 0, 0)

            extracted.append(
                ExtractedImage(
                    image_id=image_id,
                    file_path=image_path,
                    page_number=page_number,
                    bbox=BoundingBox(
                        x0=bbox_rect.x0, y0=bbox_rect.y0, x1=bbox_rect.x1, y1=bbox_rect.y1,
                        page_number=page_number,
                    ),
                )
            )
        return extracted

    def _extract_text_chunks(
        self, page, page_number: int, document_id: str, source_file: str,
        page_images: list[ExtractedImage],
    ) -> tuple[list[DocumentChunk], list]:
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]  # solo bloques de texto

        chunks: list[DocumentChunk] = []
        chunk_bboxes: list = []  # bbox de cada chunk, mismo índice que `chunks` (para el fallback de imágenes huérfanas)
        buffer_text = ""
        buffer_bbox = None

        def flush_buffer():
            if not buffer_text.strip():
                return
            related = self._find_related_images(buffer_bbox, page_images)
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    text=buffer_text.strip(),
                    page_number=page_number,
                    source_file=source_file,
                    related_image_ids=[img.image_id for img in related],
                )
            )
            chunk_bboxes.append(buffer_bbox)

        for block in text_blocks:
            x0, y0, x1, y1, text, *_ = block
            if len(buffer_text) + len(text) > self._chunk_max_chars and buffer_text:
                flush_buffer()
                buffer_text = ""
                buffer_bbox = None

            buffer_text += text + "\n"
            buffer_bbox = fitz.Rect(x0, y0, x1, y1) if buffer_bbox is None else buffer_bbox | fitz.Rect(x0, y0, x1, y1)

        flush_buffer()
        return chunks, chunk_bboxes

    @staticmethod
    def _ensure_every_image_has_a_chunk(
        chunks: list[DocumentChunk], chunk_bboxes: list, page_images: list[ExtractedImage]
    ) -> None:
        """
        Fallback de robustez: PROXIMITY_THRESHOLD puede fallar en layouts donde
        el pie de foto queda un poco más lejos de lo esperado, dejando una
        imagen sin ningún chunk que la referencie ("huérfana"). Sin esto, esa
        imagen jamás se recuperaría en una consulta, aunque exista en el PDF.
        Acá se garantiza que TODA imagen quede asociada al menos al chunk de
        texto más cercano de su misma página, sin importar la distancia.
        """
        if not chunks or not page_images:
            return

        referenced_ids = {img_id for chunk in chunks for img_id in chunk.related_image_ids}

        for img in page_images:
            if img.image_id in referenced_ids:
                continue  # ya tiene al menos un chunk que la referencia

            closest_index, closest_distance = None, None
            for i, bbox in enumerate(chunk_bboxes):
                if bbox is None:
                    continue
                distance = min(abs(bbox.y1 - img.bbox.y0), abs(img.bbox.y1 - bbox.y0))
                if closest_distance is None or distance < closest_distance:
                    closest_index, closest_distance = i, distance

            if closest_index is not None:
                chunks[closest_index].related_image_ids.append(img.image_id)

    def _extract_table_chunks(
        self, page, page_number: int, document_id: str, source_file: str
    ) -> list[DocumentChunk]:
        """
        Extrae tablas como estructura separada (no como texto plano), usando el
        detector nativo de PyMuPDF. Cada tabla se convierte a Markdown para
        preservar filas y columnas — indexarla como texto suelto perdería esa
        estructura y dificultaría que el LLM interprete relaciones entre celdas.
        """
        chunks: list[DocumentChunk] = []
        try:
            found_tables = page.find_tables()
        except Exception:  # noqa: BLE001 - algunas páginas sin tablas lanzan error interno de detección
            return chunks

        for table_index, table in enumerate(found_tables.tables):
            try:
                markdown_table = table.to_markdown()
            except Exception:  # noqa: BLE001 - tabla mal formada, se ignora en vez de romper la ingesta
                continue

            if not markdown_table.strip():
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    text=f"[Tabla {table_index + 1} de la página {page_number}]\n{markdown_table}",
                    page_number=page_number,
                    source_file=source_file,
                    related_image_ids=[],
                )
            )
        return chunks

    @staticmethod
    def _find_related_images(text_bbox, page_images: list[ExtractedImage]) -> list[ExtractedImage]:
        if text_bbox is None:
            return []
        related = []
        for img in page_images:
            vertical_gap = min(
                abs(text_bbox.y1 - img.bbox.y0),
                abs(img.bbox.y1 - text_bbox.y0),
            )
            if vertical_gap <= PROXIMITY_THRESHOLD:
                related.append(img)
        return related