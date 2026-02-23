#!/usr/bin/env bash
#
# Spiral GPU PIR Map Demo
#
# Usage:
#   ./run_demo.sh                     # Full demo (requires tiles.bin)
#   ./run_demo.sh --synthetic         # Generate synthetic tiles first
#   ./run_demo.sh --ngrok             # Also start ngrok tunnel
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPIRAL_GPU_PORT=8082
PROXY_PORT=8004
USE_SYNTHETIC=false
USE_NGROK=false
TILES_DIR="$ROOT/demo/tiles"
BUILD_DIR="$ROOT/build"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --synthetic)    USE_SYNTHETIC=true; shift ;;
        --ngrok)        USE_NGROK=true; shift ;;
        --port)         SPIRAL_GPU_PORT="$2"; shift 2 ;;
        --proxy-port)   PROXY_PORT="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--synthetic] [--ngrok] [--port N] [--proxy-port N]"
            exit 1 ;;
    esac
done

# ─── Kill stale processes ──────────────────────────────────────────────────────
kill_port() {
    local port=$1
    local pids
    pids=$(ss -tlnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
    if [[ -n "$pids" ]]; then
        echo "==> Killing stale process(es) on port $port (PIDs: $pids)"
        for pid in $pids; do kill "$pid" 2>/dev/null || true; done
        sleep 1
    fi
}

kill_port "$SPIRAL_GPU_PORT"
kill_port "$PROXY_PORT"

PIDS=()
CLEANED_UP=false
cleanup() {
    if $CLEANED_UP; then return; fi
    CLEANED_UP=true
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for i in $(seq 1 6); do
        alive=false
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then alive=true; break; fi
        done
        if ! $alive; then break; fi
        sleep 0.5
    done
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# ─── Step 1: Build spiral-gpu server ──────────────────────────────────────────
SERVER="$BUILD_DIR/server/spiral_gpu_server"
if [[ ! -x "$SERVER" ]]; then
    echo "==> Building spiral-gpu server..."
    cmake -S "$ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=120
    cmake --build "$BUILD_DIR" --target spiral_gpu_server --parallel "$(nproc)"
else
    echo "==> spiral-gpu server already built ($SERVER)"
fi

# ─── Step 2: Set up demo/frontend and demo/proxy symlinks ─────────────────────
mkdir -p "$ROOT/demo"
if [[ ! -e "$ROOT/demo/frontend" ]]; then
    echo "==> Symlinking frontend from spiral-cpu..."
    ln -sf "$ROOT/../spiral-cpu/demo/frontend" "$ROOT/demo/frontend"
fi
if [[ ! -e "$ROOT/demo/proxy" ]]; then
    echo "==> Symlinking proxy from spiral-cpu..."
    ln -sf "$ROOT/../spiral-cpu/demo/proxy" "$ROOT/demo/proxy"
fi

# ─── Step 3: Tile database ─────────────────────────────────────────────────────
if $USE_SYNTHETIC; then
    echo "==> Generating synthetic tiles..."
    mkdir -p "$TILES_DIR"
    python3 "$ROOT/../mulpir-gpu/demo/tiles/prepare_tiles.py" \
        --synthetic --synthetic-count 1000 \
        --output "$TILES_DIR"
    TILES_BIN="$TILES_DIR/tiles.bin"
else
    TILES_BIN="$TILES_DIR/tiles.bin"
    if [[ ! -f "$TILES_BIN" ]]; then
        # Fall back to shared tile data
        for FALLBACK_DIR in \
            "$ROOT/../spiral-cpu/demo/tiles" \
            "$ROOT/../mulpir-gpu/demo/tiles"; do
            if [[ -f "$FALLBACK_DIR/tiles.bin" ]]; then
                echo "==> Using tiles from $FALLBACK_DIR"
                TILES_DIR="$FALLBACK_DIR"
                TILES_BIN="$TILES_DIR/tiles.bin"
                break
            fi
        done
    fi
    if [[ ! -f "$TILES_BIN" ]]; then
        echo "ERROR: $TILES_BIN not found."
        echo "  Run: $0 --synthetic"
        exit 1
    fi
fi

NUM_TILES=$(python3 -c "import json; m=json.load(open('$TILES_DIR/tile_mapping.json')); print(m.get('num_pir_slots', m['num_tiles']))")
TILE_SIZE=$(python3  -c "import json; print(json.load(open('$TILES_DIR/tile_mapping.json'))['tile_size'])")
echo "==> Database: $TILES_BIN ($NUM_TILES PIR slots, ${TILE_SIZE}B each)"

# ─── Step 4: Start spiral-gpu server ─────────────────────────────────────────
echo "==> Starting spiral-gpu server on port $SPIRAL_GPU_PORT..."
"$SERVER" \
    --database     "$TILES_BIN" \
    --tile-mapping "$TILES_DIR/tile_mapping.json" \
    --num-tiles    "$NUM_TILES" \
    --tile-size    "$TILE_SIZE" \
    --port         "$SPIRAL_GPU_PORT" &
SPIRAL_PID=$!
PIDS+=($SPIRAL_PID)

# Wait for server to be ready
echo -n "==> Waiting for spiral-gpu server..."
for i in $(seq 1 60); do
    if ! kill -0 "$SPIRAL_PID" 2>/dev/null; then
        echo " FAILED (process exited)"
        exit 1
    fi
    if (echo > /dev/tcp/localhost/$SPIRAL_GPU_PORT) 2>/dev/null; then
        echo " ready!"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo " TIMEOUT"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# ─── Step 5: Start Flask proxy ────────────────────────────────────────────────
echo "==> Starting Flask proxy on port $PROXY_PORT..."
python3 "$ROOT/demo/proxy/server.py" \
    --spiral-port "$SPIRAL_GPU_PORT" \
    --port        "$PROXY_PORT" \
    --tiles-dir   "$TILES_DIR" &
PROXY_PID=$!
PIDS+=($PROXY_PID)

sleep 2
if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "ERROR: Flask proxy failed to start"
    exit 1
fi

# ─── Step 6: Optional ngrok ───────────────────────────────────────────────────
if $USE_NGROK; then
    if command -v ngrok &>/dev/null; then
        echo "==> Starting ngrok tunnel..."
        ngrok http "$PROXY_PORT" &
        PIDS+=($!)
        sleep 2
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels \
            | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null || echo "unknown")
        echo "==> Public URL: $NGROK_URL"
    else
        echo "WARNING: ngrok not found, skipping tunnel"
    fi
fi

echo ""
echo "========================================"
echo "  Spiral GPU PIR Map Demo ready!"
echo ""
echo "  Open:  http://localhost:$PROXY_PORT"
echo ""
echo "  GPU server: localhost:$SPIRAL_GPU_PORT"
echo "  Proxy:      localhost:$PROXY_PORT"
echo "  Tiles:      $NUM_TILES"
echo ""
echo "  Press Ctrl+C to stop"
echo "========================================"
echo ""

wait
