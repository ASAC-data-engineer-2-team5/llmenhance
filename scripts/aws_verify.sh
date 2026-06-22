#!/usr/bin/env bash
set -euo pipefail

cd /opt/llmenhance/app

docker compose -f docker-compose.aws.yml ps
curl -fsS http://127.0.0.1:8000/health/services
curl -fsS http://127.0.0.1:8501/ >/dev/null

python3 - <<'PY'
import json
import urllib.request

with urllib.request.urlopen(
    "http://127.0.0.1:6333/collections/llmenhance_chunks",
    timeout=5,
) as response:
    payload = json.load(response)

points_count = payload["result"]["points_count"]
if points_count <= 0:
    raise SystemExit(f"Qdrant collection has no points: {points_count}")
print(f"QDRANT_POINTS={points_count}")
PY

echo "AWS_VERIFY_OK"
