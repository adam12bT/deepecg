FROM node:24-bookworm-slim AS frontend-build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/src ./src
RUN npm run build

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
COPY pyproject.toml eeg_data.py eeg_model.py ./
COPY sleepops ./sleepops

FROM base AS api
RUN python -m pip install '.[api]'
COPY registry ./registry
COPY artifacts/demo_windows.npz ./artifacts/
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "sleepops.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM nginx:stable-alpine AS demo
COPY --from=frontend-build /web/dist /usr/share/nginx/html
COPY deploy/nginx-demo.conf /etc/nginx/conf.d/default.conf
EXPOSE 8501

FROM base AS space
RUN apt-get update && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 spaceuser \
    && python -m pip install '.[api]'
COPY --chown=spaceuser --from=frontend-build /web/dist /app/frontend/dist
COPY --chown=spaceuser artifacts/demo_windows.npz ./artifacts/
COPY --chown=spaceuser registry ./registry
COPY --chown=spaceuser deploy/huggingface/ ./deploy/huggingface/
ENV HOME=/home/spaceuser
USER spaceuser
EXPOSE 7860
CMD ["python", "deploy/huggingface/serve.py"]
