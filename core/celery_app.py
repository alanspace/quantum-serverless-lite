"""
Celery application configuration for Quantum Serverless Lite.

Uses a local Redis instance (redis://localhost:6379/0) as both
the message broker and the result backend.
"""

from celery import Celery

# Redis connection URL — matches the docker-compose service on port 6379.
REDIS_URL = "redis://localhost:6379/0"

celery_app = Celery(
    "quantum_serverless_lite",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# ── Celery configuration ────────────────────────────────────────────
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Autodiscover tasks defined in the top-level tasks module.
    imports=["tasks"],
)
