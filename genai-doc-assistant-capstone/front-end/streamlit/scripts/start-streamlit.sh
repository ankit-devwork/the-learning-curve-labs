#!/bin/sh
set -e

PORT="${PORT:-8501}"

# Align Streamlit's upload widget limit with backend (default 10 MB).
# Streamlit default is 200 MB, which misleads users when backend enforces 10 MB.
MAX_MB="${APP_FILE_UPLOAD__MAX_FILE_SIZE_MB:-${MAX_FILE_SIZE_MB:-10}}"

if [ -n "${BACKEND_URL:-}" ]; then
  LIMITS=$(curl -sf "${BACKEND_URL%/}/upload-limits" 2>/dev/null || true)
  if [ -n "$LIMITS" ]; then
    PARSED=$(printf '%s' "$LIMITS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('max_file_size_mb',''))" 2>/dev/null || true)
    if [ -n "$PARSED" ]; then
      MAX_MB="$PARSED"
    fi
  fi
fi

mkdir -p /app/.streamlit
cat > /app/.streamlit/config.toml <<EOF
[server]
maxUploadSize = ${MAX_MB}
EOF

exec streamlit run chat.py \
  --server.port="$PORT" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.maxUploadSize="$MAX_MB" \
  --browser.gatherUsageStats=false
