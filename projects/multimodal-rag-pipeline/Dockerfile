# Path: projects/multimodal-rag-pipeline/Dockerfile
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY main.py ./
RUN python -m pip install --upgrade pip && pip install '.[ai]'

RUN mkdir -p /app/data/input /app/data/output /app/data/index && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
