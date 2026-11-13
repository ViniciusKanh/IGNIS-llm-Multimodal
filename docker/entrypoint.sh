#!/bin/sh
# Inicialização do container IGNIS: valida diretórios persistentes e inicia a API.
set -eu
cd /app

for dir in models datasets training outputs logs configs .cache; do
    mkdir -p "$dir" 2>/dev/null || true
    if [ ! -w "$dir" ]; then
        echo "ERRO: /app/$dir não é gravável pelo usuário $(id -u):$(id -g)." >&2
        echo "Em Linux, crie as pastas montadas antes de iniciar e ajuste o dono:" >&2
        echo "  mkdir -p models datasets training outputs logs configs && sudo chown -R $(id -u):$(id -g) <pastas>" >&2
        echo "ou defina IGNIS_UID/IGNIS_GID no .env para o seu usuário." >&2
        exit 70
    fi
done

# Volume de configuração vazio: restaura o padrão versionado, sem sobrescrever o do usuário.
if [ ! -f configs/train.json ]; then
    cp /opt/ignis/defaults/train.json configs/train.json
fi

if [ "$#" -eq 0 ] || [ "$1" = "serve" ]; then
    [ "$#" -gt 0 ] && shift
    echo "IGNIS Multimodal ${IGNIS_VERSION:-dev} (torch ${IGNIS_TORCH_VARIANT:-?}) em ${IGNIS_HOST}:${IGNIS_PORT} dentro do container."
    echo "Publique a porta somente em loopback (127.0.0.1); a API não possui autenticação."
    exec python -m uvicorn backend.app:app --host "${IGNIS_HOST}" --port "${IGNIS_PORT}" "$@"
fi

exec "$@"
