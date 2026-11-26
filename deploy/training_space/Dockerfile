FROM node:24-bookworm-slim AS admin-build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/vite.admin.config.js ./
COPY frontend/admin ./admin
ARG PYODIDE_VERSION=314.0.8
RUN npm pack "pyodide@$PYODIDE_VERSION" --silent --pack-destination /tmp \
    && mkdir -p admin/public/python \
    && tar -xzf "/tmp/pyodide-$PYODIDE_VERSION.tgz" --strip-components=1 -C admin/public/python \
       package/pyodide.mjs package/pyodide.asm.mjs package/pyodide.asm.wasm \
       package/python_stdlib.zip package/pyodide-lock.json \
    && node admin/stage-notebook-packages.mjs admin/public/python "$PYODIDE_VERSION"
RUN npm run build:admin

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 GRADIO_ANALYTICS_ENABLED=False \
    MPLBACKEND=Agg HOME=/home/user

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
    && python -m pip install -r requirements.txt gradio==6.32.0
COPY --chown=user:user . .
COPY --chown=user:user --from=admin-build /web/admin/dist ./frontend/admin/dist
USER user
EXPOSE 7860
CMD ["python", "app.py"]
