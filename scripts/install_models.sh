#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_FILE="$REPO_ROOT/models.txt"

if [ ! -f "$MODELS_FILE" ]; then
    echo "models.txt not found at $MODELS_FILE"
    exit 1
fi

echo "Installing models from $MODELS_FILE"
echo ""

while IFS= read -r model || [ -n "$model" ]; do
    [[ -z "$model" || "$model" == \#* ]] && continue
    echo "==> Pulling $model"
    ollama pull "$model"
    echo ""
done < "$MODELS_FILE"

echo "All models installed:"
ollama list
