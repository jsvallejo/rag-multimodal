"""
Almacén de estado de los jobs de ingesta.

Implementación en memoria para simplicidad de la prueba. En producción real
esto se cambiaría por Redis (para sobrevivir reinicios y escalar horizontalmente
entre múltiples workers) sin tocar el caso de uso, ya que este es el único
lugar que lo conoce directamente.
"""
import threading

from app.domain.entities import IngestionJob


class JobStore:
    def __init__(self):
        self._jobs: dict[str, IngestionJob] = {}
        self._lock = threading.Lock()

    def create(self, job: IngestionJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job: IngestionJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
