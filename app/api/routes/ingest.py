"""
Endpoint de ingesta. Clave: NO procesa el PDF en el request-response cycle
(eso bloquearía el hilo y causaría timeouts en documentos pesados).
En su lugar, dispara un BackgroundTask y responde inmediatamente con un job_id.
"""
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from app.api.schemas import IngestResponse, JobStatusResponse
from app.core.config import get_settings
from app.core.container import get_ingest_use_case, get_job_store
from app.domain.entities import IngestionJob

router = APIRouter(prefix="/documents", tags=["ingestion"])


@router.post("/upload", response_model=IngestResponse, status_code=202)
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se admiten archivos PDF.")

    settings = get_settings()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    job = IngestionJob.new(filename=file.filename)
    job_store = get_job_store()
    job_store.create(job)

    file_path = os.path.join(settings.UPLOAD_DIR, f"{job.job_id}_{file.filename}")
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    use_case = get_ingest_use_case()
    background_tasks.add_task(use_case.execute, job.job_id, file_path)

    return IngestResponse(job_id=job.job_id, status=job.status.value)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job_store = get_job_store()
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado.")

    return JobStatusResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status.value,
        error_message=job.error_message,
        document_id=job.document_id,
    )
