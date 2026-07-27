"""Celery application. Tasks live next to the code they operate on
(e.g. app/memory/consolidation.py); scheduling gets wired in a later stage."""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "sentinelops",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.memory.consolidation", "app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
