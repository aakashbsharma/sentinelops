# The SentinelOps API (FastAPI). Celery workers reuse this image with a
# different command (see docker-compose.yml).
FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY mcp_servers mcp_servers
COPY alembic.ini .
COPY alembic alembic

EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=5s --retries=10 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
