#!/bin/bash
set -e

PORT=${1:-8090}
WORKERS=${2:-4}
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"

echo "  RAGFlow Dify Proxy"
echo "  Port:     $PORT"
echo "  Workers:  $WORKERS"

# 安装依赖
if ! python3 -c "import flask" 2>/dev/null; then
    echo "  安装依赖..."
    pip3 install -r "$BASE_DIR/requirements.txt" -q 2>/dev/null || \
    pip install -r "$BASE_DIR/requirements.txt" -q
fi

# 停旧进程（包括孤儿 worker）
fuser -k $PORT/tcp 2>/dev/null || true
if [ -f "$LOG_DIR/proxy.pid" ]; then
    OLD=$(cat "$LOG_DIR/proxy.pid")
    if kill -0 "$OLD" 2>/dev/null; then
        echo "  停止旧进程 PID $OLD..."
        kill "$OLD" && sleep 2
    fi
fi

# 导出环境变量
export RAGFLOW_BASE_URL="${RAGFLOW_BASE_URL:-http://127.0.0.1:8088}"
export RAGFLOW_API_KEY="${RAGFLOW_API_KEY:-ragflow-307044760fae4f548209426ba6191d9e}"

cd "$BASE_DIR"
/usr/bin/python3 -m gunicorn proxy_server:app \
    --bind "0.0.0.0:$PORT" \
    --workers "$WORKERS" \
    --worker-class gevent \
    --worker-connections 1000 \
    --timeout 120 \
    --keep-alive 30 \
    --max-requests 50000 \
    --max-requests-jitter 5000 \
    --access-logfile "$LOG_DIR/access.log" \
    --error-logfile "$LOG_DIR/error.log" \
    --pid "$LOG_DIR/proxy.pid" \
    --daemon

sleep 2

if [ -f "$LOG_DIR/proxy.pid" ] && kill -0 "$(cat "$LOG_DIR/proxy.pid")" 2>/dev/null; then
    echo "  ✓ Proxy 已启动 (PID: $(cat "$LOG_DIR/proxy.pid"))"
else
    echo "  ✗ Proxy 启动失败，查看日志: $LOG_DIR/error.log"
    exit 1
fi
