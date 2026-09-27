"""
Caso de uso: Consultar el sistema RAG.

Contiene el "Prompt Engineering defensivo": si no hay contexto suficiente,
se le instruye explícitamente al LLM que lo declare, para mitigar alucinaciones.
"""
from app.domain.entities import RagAnswer, RetrievedContext
from app.domain.ports.image_loader_port import ImageLoaderPort
from app.domain.ports.llm_port import LLMPort
from app.domain.ports.vector_store_port import VectorStorePort

# Cuántas imágenes distintas (de los chunks recuperados) se envían al LLM.
# Se acota para no disparar el tamaño del prompt ni el tiempo de inferencia.
MAX_IMAGES_PER_QUERY = 2

# Muchos modelos de visión livianos (ej. moondream) tienen una ventana de
# contexto muy chica (2048 tokens), y buena parte se consume solo en
# procesar la imagen. Cuando se envían imágenes, el contexto de texto se
# recorta agresivamente para no exceder ese límite.
MAX_CONTEXT_CHARS_WITH_IMAGES = 600

# Palabras que indican que la pregunta realmente depende de contenido visual.
# Sin esta señal, NO se envían imágenes aunque algún chunk recuperado tenga
# una relacionada: adjuntar imágenes de forma indiscriminada dispara el
# recorte de contexto (MAX_CONTEXT_CHARS_WITH_IMAGES) y perjudica preguntas
# puramente textuales (ej. "¿de qué trata el documento?").
VISUAL_INTENT_KEYWORDS = [
    "imagen", "imágenes", "figura", "figuras", "captura", "capturas",
    "diagrama", "diagramas", "muestra", "muestran", "screenshot",
    "pantalla", "pantallazo", "foto", "fotos", "gráfico", "gráficos",
    "ilustración", "ilustra", "visual", "dibujo", "esquema",
]

SYSTEM_PROMPT = """Eres un asistente técnico que responde preguntas basándose \
ÚNICAMENTE en el contexto proporcionado, extraído de documentos técnicos (PDF).

Reglas estrictas:
1. Si el contexto no contiene información suficiente para responder con \
certeza, debes decir explícitamente: "No cuento con información suficiente \
en los documentos para responder esto con certeza." No inventes ni asumas datos.
2. Cuando cites información, menciona el archivo y número de página de donde \
proviene (ya incluidos en el contexto).
3. Si se te adjuntan imágenes extraídas del documento, obsérvalas con atención \
y describe explícitamente lo que muestran (capturas de pantalla, diagramas, \
interfaces, etc.) como parte de tu respuesta, no solo el texto que las rodea.
4. Responde en el mismo idioma de la pregunta del usuario.
"""


class QueryDocumentUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        llm: LLMPort,
        image_loader: ImageLoaderPort | None = None,
        top_k: int = 5,
    ):
        self._vector_store = vector_store
        self._llm = llm
        self._image_loader = image_loader
        self._top_k = top_k

    async def execute(self, question: str) -> RagAnswer:
        query_embedding = await self._llm.embed(question)

        results = await self._vector_store.search(
            query_embedding=query_embedding, query_text=question, top_k=self._top_k
        )

        if not results:
            return RagAnswer(
                answer="No cuento con información suficiente en los documentos "
                       "para responder esto con certeza.",
                sources=[],
                insufficient_context=True,
            )

        contexts = [
            RetrievedContext(
                chunk=chunk,
                score=score,
                image_paths=[],  # se resuelve en la capa de infraestructura/API con las rutas reales
            )
            for chunk, score in results
        ]

        images_base64 = (
            self._load_related_images(contexts) if self._question_has_visual_intent(question) else []
        )

        # Con imágenes, el presupuesto de tokens de texto es mucho menor
        # (el encoder visual ya ocupa buena parte de la ventana de contexto).
        context_text = self._build_context_text(
            contexts, max_chars=MAX_CONTEXT_CHARS_WITH_IMAGES if images_base64 else None
        )
        user_prompt = f"Contexto recuperado:\n{context_text}\n\nPregunta: {question}"

        answer_text = await self._llm.generate(SYSTEM_PROMPT, user_prompt, images_base64=images_base64)

        insufficient = "no cuento con información suficiente" in answer_text.lower()

        return RagAnswer(answer=answer_text, sources=contexts, insufficient_context=insufficient)

    @staticmethod
    def _question_has_visual_intent(question: str) -> bool:
        """
        Heurística simple: solo se activa el envío de imágenes al LLM si la
        pregunta usa palabras que sugieren que depende de contenido visual.
        Evita adjuntar imágenes (y recortar el contexto de texto) en preguntas
        generales que casualmente recuperan un chunk con imagen asociada.
        """
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in VISUAL_INTENT_KEYWORDS)

    def _load_related_images(self, contexts: list[RetrievedContext]) -> list[str]:
        """
        Recolecta, de los chunks recuperados (por orden de relevancia), las
        imágenes relacionadas y las carga en base64 vía ImageLoaderPort, hasta
        el límite MAX_IMAGES_PER_QUERY. Si no hay loader configurado (LLM sin
        visión) o ninguna imagen relacionada, devuelve una lista vacía.
        """
        if self._image_loader is None:
            return []

        images: list[str] = []
        seen_filenames: set[str] = set()

        for ctx in contexts:
            for image_filename in ctx.chunk.related_image_ids:
                if len(images) >= MAX_IMAGES_PER_QUERY:
                    return images
                if image_filename in seen_filenames:
                    continue
                seen_filenames.add(image_filename)

                encoded = self._image_loader.load_as_base64(image_filename)
                if encoded is not None:
                    images.append(encoded)

        return images

    @staticmethod
    def _build_context_text(contexts: list[RetrievedContext], max_chars: int | None = None) -> str:
        parts = []
        remaining = max_chars

        for ctx in contexts:
            text = ctx.chunk.text
            if remaining is not None:
                if remaining <= 0:
                    break
                text = text[:remaining]
                remaining -= len(text)

            parts.append(
                f"[Fuente: {ctx.chunk.source_file}, Página {ctx.chunk.page_number}]\n{text}"
            )

        return "\n\n---\n\n".join(parts)