# IGNIS Multimodal — imagem de distribuição (GHCR).
#
# Estágio 1 compila o frontend React; estágio 2 contém Python, PyTorch, FastAPI e o
# frontend compilado. Pesos de modelos, datasets, checkpoints, caches e credenciais NÃO
# fazem parte da imagem: são montados como volumes ou baixados pelo Model Manager.
#
# TORCH_VARIANT=cu128 (padrão) usa as wheels CUDA 12.8 do PyTorch, que trazem o runtime
# CUDA/cuDNN; o host precisa apenas do driver NVIDIA (>= 570) e do NVIDIA Container Toolkit
# (ou Docker Desktop + WSL2). Sem GPU, a mesma imagem executa em CPU (FP32).
# TORCH_VARIANT=cpu gera uma imagem menor, exclusivamente CPU.

# Imagens-base parametrizáveis (espelhos corporativos ou testes offline).
ARG PYTHON_IMAGE=python:3.13-slim-bookworm
ARG NODE_IMAGE=node:22-bookworm-slim

FROM ${NODE_IMAGE} AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


FROM ${PYTHON_IMAGE} AS runtime

ARG TORCH_VARIANT=cu128
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/${TORCH_VARIANT}
ARG IGNIS_VERSION=0.0.0-dev
ARG VCS_REF=unknown
ARG IGNIS_UID=1000
ARG IGNIS_GID=1000

LABEL org.opencontainers.image.title="IGNIS Multimodal" \
      org.opencontainers.image.description="Plataforma local de IA multimodal: visão-linguagem, geração/edição com Diffusers e treinamento LoRA. Sem pesos de modelos." \
      org.opencontainers.image.source="https://github.com/ViniciusKanh/IGNIS-llm-Multimodal" \
      org.opencontainers.image.url="https://github.com/ViniciusKanh/IGNIS-llm-Multimodal" \
      org.opencontainers.image.authors="Vinicius de Souza Santos" \
      org.opencontainers.image.version="${IGNIS_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      io.ignis.torch.variant="${TORCH_VARIANT}" \
      io.ignis.torch.version="${TORCH_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    IGNIS_VERSION=${IGNIS_VERSION} \
    IGNIS_TORCH_VARIANT=${TORCH_VARIANT} \
    IGNIS_HOST=0.0.0.0 \
    IGNIS_PORT=8000

# PyTorch em camada própria (a maior da imagem), reaproveitada quando só o código muda.
RUN python -m pip install \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}" \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/ignis-pip-freeze.txt

RUN groupadd --gid "${IGNIS_GID}" ignis \
    && useradd --uid "${IGNIS_UID}" --gid "${IGNIS_GID}" --create-home --shell /usr/sbin/nologin ignis

COPY --chown=root:root backend/ backend/
COPY --chown=root:root scripts/ scripts/
COPY --chown=root:root tests/ tests/
COPY --chown=root:root huggingface/ huggingface/
COPY --chown=root:root pytest.ini README.md THIRD_PARTY_NOTICES.md ./
COPY --chown=root:root configs/train.json /opt/ignis/defaults/train.json
COPY --chown=root:root docker/entrypoint.sh /usr/local/bin/ignis-entrypoint
COPY --from=frontend --chown=root:root /src/frontend/dist frontend/dist

# Código somente leitura para o usuário da aplicação; dados persistentes graváveis.
RUN sed -i 's/\r$//' /usr/local/bin/ignis-entrypoint \
    && chmod 0755 /usr/local/bin/ignis-entrypoint \
    && mkdir -p models datasets training outputs logs configs .cache \
    && cp /opt/ignis/defaults/train.json configs/train.json \
    && chown -R ignis:ignis models datasets training outputs logs configs .cache \
    && python -m compileall -q backend scripts

USER ignis

VOLUME ["/app/models", "/app/datasets", "/app/training", "/app/outputs", "/app/logs", "/app/configs", "/app/.cache"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('IGNIS_PORT', '8000'), timeout=8)" || exit 1

ENTRYPOINT ["ignis-entrypoint"]
CMD ["serve"]
