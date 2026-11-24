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
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "sleepops.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS demo
RUN python -m pip install '.[demo]'
COPY demo.py ./
COPY artifacts/demo_windows.npz ./artifacts/
USER appuser
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "demo.py", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]

FROM base AS space
RUN apt-get update && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 spaceuser \
    && python -m pip install '.[api,demo]'
COPY --chown=spaceuser demo.py ./
COPY --chown=spaceuser artifacts/demo_windows.npz ./artifacts/
COPY --chown=spaceuser registry ./registry
COPY --chown=spaceuser deploy/huggingface/ ./deploy/huggingface/
ENV HOME=/home/spaceuser SLEEPOPS_API_URL=http://127.0.0.1:8000
USER spaceuser
EXPOSE 7860
CMD ["python", "deploy/huggingface/serve.py"]
