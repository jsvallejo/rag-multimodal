"""
Tests del caso de uso de consulta. Clave: se verifica el comportamiento
DEFENSIVO ante falta de contexto (mitigación de alucinaciones), que es uno
de los puntos explícitamente evaluados en la prueba.
"""
import pytest

from app.domain.entities import DocumentChunk
from app.domain.use_cases.query_document import QueryDocumentUseCase
from tests.conftest import FakeImageLoader, FakeLLM, FakeVectorStore


@pytest.mark.asyncio
async def test_query_returns_insufficient_context_when_no_results(job_store):
    llm = FakeLLM()
    vector_store = FakeVectorStore(search_results=[])  # sin resultados de búsqueda

    use_case = QueryDocumentUseCase(vector_store=vector_store, llm=llm)
    result = await use_case.execute("¿Cuál es la presión máxima del motor?")

    assert result.insufficient_context is True
    assert "no cuento con información suficiente" in result.answer.lower()
    assert len(llm.generate_calls) == 0


@pytest.mark.asyncio
async def test_query_builds_context_from_retrieved_chunks(sample_chunk):
    llm = FakeLLM(fixed_answer="El motor requiere revisión cada 5000 km.")
    vector_store = FakeVectorStore(search_results=[(sample_chunk, 0.92)])

    use_case = QueryDocumentUseCase(vector_store=vector_store, llm=llm)
    result = await use_case.execute("¿Cada cuánto se revisa el motor?")

    assert result.answer == "El motor requiere revisión cada 5000 km."
    assert len(result.sources) == 1
    assert result.sources[0].chunk.source_file == "Manual_Motor.pdf"

    _, user_prompt = llm.generate_calls[0]
    assert "Manual_Motor.pdf" in user_prompt
    assert "Página 12" in user_prompt


@pytest.mark.asyncio
async def test_query_detects_insufficient_context_flagged_by_llm(sample_chunk):
    """Si el LLM mismo declara falta de contexto (siguiendo el prompt defensivo), debe reflejarse en el flag."""
    llm = FakeLLM(fixed_answer="No cuento con información suficiente en los documentos para responder esto con certeza.")
    vector_store = FakeVectorStore(search_results=[(sample_chunk, 0.4)])

    use_case = QueryDocumentUseCase(vector_store=vector_store, llm=llm)
    result = await use_case.execute("¿Cuál es el color oficial del motor?")

    assert result.insufficient_context is True


@pytest.mark.asyncio
async def test_query_respects_top_k_limit(sample_chunk):
    many_results = [(sample_chunk, 0.9 - i * 0.01) for i in range(10)]
    vector_store = FakeVectorStore(search_results=many_results)

    use_case = QueryDocumentUseCase(vector_store=vector_store, llm=FakeLLM(), top_k=3)
    result = await use_case.execute("pregunta cualquiera")

    assert len(result.sources) <= 3


@pytest.mark.asyncio
async def test_query_sends_related_images_to_vision_llm(sample_chunk):
    """El LLM debe recibir la imagen relacionada al chunk recuperado (visión)."""
    llm = FakeLLM()
    vector_store = FakeVectorStore(search_results=[(sample_chunk, 0.9)])
    image_loader = FakeImageLoader(contents={"img-1": "ZmFrZV9pbWFnZV9ieXRlcw=="})

    use_case = QueryDocumentUseCase(vector_store=vector_store, llm=llm, image_loader=image_loader)
    await use_case.execute("¿Qué muestra el diagrama?")

    assert llm.last_images == ["ZmFrZV9pbWFnZV9ieXRlcw=="]


@pytest.mark.asyncio
async def test_query_without_image_loader_sends_no_images(sample_chunk):
    """Sin ImageLoaderPort configurado (ej. modelo sin visión), no debe intentar cargar imágenes."""
    llm = FakeLLM()
    vector_store = FakeVectorStore(search_results=[(sample_chunk, 0.9)])

    use_case = QueryDocumentUseCase(vector_store=vector_store, llm=llm, image_loader=None)
    await use_case.execute("¿Qué muestra el diagrama?")

    assert llm.last_images == []


@pytest.mark.asyncio
async def test_query_respects_max_images_limit(sample_chunk):
    """No debe exceder MAX_IMAGES_PER_QUERY aunque haya más imágenes relacionadas disponibles."""
    chunk_with_many_images = DocumentChunk(
        chunk_id="chunk-2",
        document_id="doc-1",
        text="Texto con varias imágenes relacionadas.",
        page_number=5,
        source_file="Manual.pdf",
        related_image_ids=["img-a", "img-b", "img-c", "img-d"],
    )
    llm = FakeLLM()
    vector_store = FakeVectorStore(search_results=[(chunk_with_many_images, 0.9)])
    image_loader = FakeImageLoader(contents={
        "img-a": "YQ==", "img-b": "Yg==", "img-c": "Yw==", "img-d": "ZA==",
    })

    use_case = QueryDocumentUseCase(vector_store=vector_store, llm=llm, image_loader=image_loader)
    await use_case.execute("¿qué muestran las imágenes?")

    assert len(llm.last_images) == 2  # MAX_IMAGES_PER_QUERY


@pytest.mark.asyncio
async def test_query_does_not_attach_images_for_generic_question(sample_chunk):
    """Una pregunta general (sin intención visual) no debe activar el envío de imágenes,
    aunque el chunk recuperado tenga una imagen relacionada."""
    llm = FakeLLM()
    vector_store = FakeVectorStore(search_results=[(sample_chunk, 0.9)])
    image_loader = FakeImageLoader(contents={"img-1": "ZmFrZQ=="})

    use_case = QueryDocumentUseCase(vector_store=vector_store, llm=llm, image_loader=image_loader)
    await use_case.execute("¿De qué trata el documento?")

    assert llm.last_images == []


@pytest.mark.asyncio
async def test_query_attaches_images_when_question_mentions_visual_content(sample_chunk):
    """Una pregunta que menciona explícitamente contenido visual sí debe activar el envío."""
    llm = FakeLLM()
    vector_store = FakeVectorStore(search_results=[(sample_chunk, 0.9)])
    image_loader = FakeImageLoader(contents={"img-1": "ZmFrZQ=="})

    use_case = QueryDocumentUseCase(vector_store=vector_store, llm=llm, image_loader=image_loader)
    await use_case.execute("¿Qué muestra la figura 2?")

    assert llm.last_images == ["ZmFrZQ=="]


@pytest.mark.asyncio
async def test_query_skips_image_not_found_on_disk(sample_chunk):
    """Si el ImageLoaderPort no encuentra el archivo (devuelve None), se omite sin fallar."""
    llm = FakeLLM()
    vector_store = FakeVectorStore(search_results=[(sample_chunk, 0.9)])
    image_loader = FakeImageLoader(contents={})  # "img-1" no está presente

    use_case = QueryDocumentUseCase(vector_store=vector_store, llm=llm, image_loader=image_loader)
    await use_case.execute("¿qué muestra la imagen?")

    assert llm.last_images == []