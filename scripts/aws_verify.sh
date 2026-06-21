#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

curl -fsS http://127.0.0.1:6333 >/dev/null
curl -fsS http://127.0.0.1:8000/health >/dev/null
curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null

docker compose -f docker-compose.aws.yml run --rm rag-api python -m app.healthcheck

curl -fsS http://127.0.0.1:8000/api/ask/qwen \
  -H 'content-type: application/json' \
  -d '{"question":"연차 신청은 며칠 전까지 해야 하나요?","top_k":3}' \
  -o /tmp/llmenhance-qwen-response.json

python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/llmenhance-qwen-response.json").read_text(encoding="utf-8"))
if not payload.get("sources"):
    raise SystemExit("expected non-empty sources for grounded sample answer")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
