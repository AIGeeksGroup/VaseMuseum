#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"

PROMPT="${1:-Hello, please introduce yourself.}"
# If provided, $2 is treated as image path/URL. This makes it easy to call:
#   bash call_webwatcher_api.sh "question" "/abs/path/to/image.jpg"
# Otherwise falls back to IMAGE_URL env var.
IMAGE_URL_ARG="${2:-}"
IMAGE_URL="${IMAGE_URL_ARG:-${IMAGE_URL:-}}"

# Some users may paste Cursor's "@file" references into shells by accident.
# If IMAGE_URL looks like "...@/path/..." keep only the part before '@'.
if [[ "${IMAGE_URL}" == *"@"* ]]; then
  IMAGE_URL="${IMAGE_URL%%@*}"
fi

# If it's a local file path, normalize to absolute path for the server.
if [[ -n "${IMAGE_URL}" ]] && [[ -f "${IMAGE_URL}" ]]; then
  IMAGE_URL="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${IMAGE_URL}")"
fi

API_URL="http://${HOST}:${PORT}/v1/chat/completions"

# Python subprocess reads prompt/model/image from environment variables.
export PROMPT IMAGE_URL MODEL

if [[ -n "${IMAGE_URL}" ]]; then
  python - <<'PY' | curl -sS "${API_URL}" -H "Content-Type: application/json" -d @-
import base64, json, mimetypes, os, sys
prompt = os.environ.get("PROMPT", "")
image = os.environ.get("IMAGE_URL", "")
model = os.environ.get("MODEL", "WebWatcher-7B")

def to_openai_image_url(s: str) -> str:
  s = (s or "").strip()
  if not s:
    return s
  # Already a URL (http/https) or data URL.
  if s.startswith("http://") or s.startswith("https://") or s.startswith("data:"):
    return s
  # Treat as local file path; encode as data URL.
  if os.path.isfile(s):
    mime, _ = mimetypes.guess_type(s)
    if not mime:
      # Reasonable default for many vision models.
      mime = "image/jpeg"
    with open(s, "rb") as f:
      b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"
  # Fall back to sending raw string; server may resolve it.
  return s

image_url = to_openai_image_url(image)
req = {
  "model": model,
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": image_url}},
      {"type": "text", "text": prompt},
    ],
  }],
  "max_tokens": 512,
  "temperature": 0.7,
}
print(json.dumps(req, ensure_ascii=False))
PY
else
  python - <<'PY' | curl -sS "${API_URL}" -H "Content-Type: application/json" -d @-
import json, os
prompt = os.environ.get("PROMPT", "")
model = os.environ.get("MODEL", "WebWatcher-7B")
req = {
  "model": model,
  "messages": [{"role": "user", "content": prompt}],
  "max_tokens": 256,
  "temperature": 0.7,
}
print(json.dumps(req, ensure_ascii=False))
PY
fi

