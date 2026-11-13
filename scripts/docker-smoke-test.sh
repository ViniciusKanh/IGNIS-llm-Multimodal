#!/usr/bin/env bash
# Teste de fumaça da imagem IGNIS: API, frontend, permissões, persistência e CUDA.
# Não baixa modelos nem executa inferência; isso é feito por scripts/gpu_check.py.
#
# Uso: scripts/docker-smoke-test.sh <imagem> [--gpu]
set -euo pipefail

IMAGE="${1:?Informe a imagem, ex.: ghcr.io/viniciuskanh/ignis-multimodal:latest}"
GPU_FLAG=()
[[ "${2:-}" == "--gpu" ]] && GPU_FLAG=(--gpus all)
PORT="${SMOKE_PORT:-18000}"
NAME="ignis-smoke-$$"
DATA="$(mktemp -d)"
FAILURES=0

cleanup() {
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    # Arquivos criados pelo uid do container podem exigir o próprio container para remoção.
    docker run --rm --entrypoint sh -v "$DATA:/data" --user 0 "$IMAGE" -c 'rm -rf /data/*' >/dev/null 2>&1 || true
    rm -rf "$DATA"
}
trap cleanup EXIT

pass() { echo "  [OK]    $*"; }
fail() { echo "  [FALHA] $*"; FAILURES=$((FAILURES + 1)); }
check() { local description="$1"; shift; if "$@"; then pass "$description"; else fail "$description"; fi; }

mkdir -p "$DATA"/{models,datasets,training,outputs,logs,configs}
chmod -R 0777 "$DATA"

start() {
    docker run -d --name "$NAME" --init "${GPU_FLAG[@]}" \
        -p "127.0.0.1:${PORT}:8000" \
        -e IGNIS_ALLOWED_ORIGINS="http://127.0.0.1:${PORT}" \
        -v "$DATA/models:/app/models" -v "$DATA/datasets:/app/datasets" \
        -v "$DATA/training:/app/training" -v "$DATA/outputs:/app/outputs" \
        -v "$DATA/logs:/app/logs" -v "$DATA/configs:/app/configs" \
        "$IMAGE" >/dev/null
    for _ in $(seq 1 90); do
        if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then return 0; fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]]; then break; fi
        sleep 2
    done
    echo "Container não respondeu:"; docker logs "$NAME" | tail -50; return 1
}

echo "== Imagem: $IMAGE"
docker image inspect "$IMAGE" --format 'Tamanho: {{.Size}} bytes | Labels: {{json .Config.Labels}}'

echo "== Conteúdo da imagem"
check "sem pesos/checkpoints na imagem" \
    test -z "$(docker run --rm --entrypoint sh "$IMAGE" -c "find / -xdev \( -name '*.safetensors' -o -name '*.ckpt' -o -name '*.gguf' -o -name '*.pt' -o \( -name '*.pth' -not -path '*/site-packages/*' \) \) -not -path '/proc/*' 2>/dev/null | head -5")"
check "sem diretórios de dados preenchidos" \
    test "$(docker run --rm --entrypoint sh "$IMAGE" -c 'ls -A /app/models /app/datasets /app/training /app/outputs | grep -v ":" | grep -c .' || true)" = "0"
check "sem credenciais (.env, tokens HF)" \
    test -z "$(docker run --rm --entrypoint sh "$IMAGE" -c "find /app /home /root -xdev \( -name '.env' -o -name 'token' -o -name '*.pem' \) 2>/dev/null | head -5")"
check "usuário não-root" test "$(docker run --rm --entrypoint id "$IMAGE" -u)" != "0"
check "código da aplicação somente leitura" \
    docker run --rm --entrypoint sh "$IMAGE" -c 'test ! -w /app/backend/app.py'

echo "== Inicialização"
start
check "GET /api/health" curl -fsS "http://127.0.0.1:${PORT}/api/health" -o "$DATA/health.json"
cat "$DATA/health.json"; echo
check "frontend compilado servido em /" sh -c "curl -fsS http://127.0.0.1:${PORT}/ | grep -q '<div id=\"root\"'"
check "OpenAPI disponível" sh -c "curl -fsS http://127.0.0.1:${PORT}/openapi.json | grep -q '/api/images'"
curl -fsS "http://127.0.0.1:${PORT}/api/status" -o "$DATA/status.json"
check "GET /api/status informa CUDA (booleano) e 3 modelos" python3 -c "
import json, sys
s = json.load(open('$DATA/status.json'))
print('  cuda=%s gpu=%s torch=%s' % (s['cuda'], s['gpu'], s['torch']))
sys.exit(0 if isinstance(s['cuda'], bool) and len(s['models']) == 3 else 1)"
check "falha explícita sem pesos (sem resposta fictícia)" sh -c "
code=\$(curl -s -o /dev/null -w '%{http_code}' -X POST -F prompt=teste http://127.0.0.1:${PORT}/api/images); test \"\$code\" = 503"
check "origem externa bloqueada (403)" sh -c "
code=\$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Origin: http://evil.example' -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:${PORT}/api/config); test \"\$code\" = 403"

echo "== Persistência"
CONFIG='{"name":"docker-smoke","rank":8,"resolution":512,"batch_size":1,"gradient_accumulation":4,"learning_rate":0.0001,"max_steps":800,"warmup_steps":50,"checkpoint_every":50,"validation_every":50,"patience":5,"min_steps":300,"seed":42}'
check "POST /api/config com origem permitida" curl -fsS -X POST -H "Origin: http://127.0.0.1:${PORT}" \
    -H 'Content-Type: application/json' -d "$CONFIG" "http://127.0.0.1:${PORT}/api/config" -o /dev/null
check "configuração gravada no volume do host" grep -q docker-smoke "$DATA/configs/train.json"
check "SQLite de histórico criado em outputs" sh -c "curl -fsS http://127.0.0.1:${PORT}/api/chat/smoke >/dev/null && test -f '$DATA/outputs/history.sqlite'"
docker restart "$NAME" >/dev/null
for _ in $(seq 1 60); do curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1 && break; sleep 2; done
check "configuração preservada após reinício" sh -c "curl -fsS http://127.0.0.1:${PORT}/api/config | grep -q docker-smoke"

echo "== Testes automatizados dentro do container"
check "pytest (contratos + HTTP)" docker exec -e IGNIS_TEST_HTTP=1 "$NAME" python -m pytest -q -p no:cacheprovider

echo "== Diagnóstico CUDA (sem inferência)"
docker exec "$NAME" python scripts/gpu_check.py --output /app/outputs/validation/smoke-gpu-check.json | tail -25 || fail "gpu_check"
if [[ ${#GPU_FLAG[@]} -gt 0 ]]; then
    check "CUDA detectada com --gpus all" docker exec "$NAME" python scripts/gpu_check.py --require-cuda --output /app/outputs/validation/smoke-gpu-required.json
fi

echo "== Logs do container"
docker logs "$NAME" 2>&1 | tail -8

if [[ $FAILURES -gt 0 ]]; then echo "RESULTADO: $FAILURES verificação(ões) falharam."; exit 1; fi
echo "RESULTADO: todas as verificações passaram."
