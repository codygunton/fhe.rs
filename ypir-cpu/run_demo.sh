#!/usr/bin/env bash
#
# PIR Map Demo — YPIR CPU backend
#
# Usage:
#   ./run_demo.sh                     # Full demo (requires tiles.bin)
#   ./run_demo.sh --synthetic         # Generate synthetic tiles first
#
set -euo pipefail

# All paths are relative to this script (ypir-cpu/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

YPIR_PORT=8084
PROXY_PORT=8009
USE_SYNTHETIC=false
TILES_DIR="$ROOT/demo/tiles"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --synthetic)
            USE_SYNTHETIC=true
            shift
            ;;
        --ypir-port)
            YPIR_PORT="$2"
            shift 2
            ;;
        --proxy-port)
            PROXY_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--synthetic] [--ypir-port N] [--proxy-port N]"
            exit 1
            ;;
    esac
done

# ─── Kill stale processes on OUR ports ONLY ─────────────────────────
# SAFETY: Only kill 8084 and 8009 — never 8080/8081/8082/8002
kill_port() {
    local port=$1
    local pids
    pids=$(ss -tlnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
    if [[ -n "$pids" ]]; then
        echo "==> Killing stale process(es) on port $port (PIDs: $pids)"
        for pid in $pids; do
            kill "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
}

kill_port "$YPIR_PORT"
kill_port "$PROXY_PORT"

# Track background PIDs for cleanup
PIDS=()
CLEANED_UP=false
cleanup() {
    if $CLEANED_UP; then return; fi
    CLEANED_UP=true
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for i in $(seq 1 6); do
        alive=false
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then alive=true; break; fi
        done
        if ! $alive; then break; fi
        sleep 0.5
    done
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# ─── Step 1: Build ypir-wasm WASM if needed ──────────────────────────
WASM_PKG="$ROOT/../crates/ypir-wasm/pkg"
if [[ ! -f "$WASM_PKG/ypir_wasm_bg.wasm" ]]; then
    echo "==> Building ypir-wasm (this takes ~30s)..."
    (cd "$ROOT/../crates/ypir-wasm" && wasm-pack build --target web --release)
else
    echo "==> ypir-wasm already built ($WASM_PKG)"
fi

# ─── Step 2: Symlink WASM pkg into frontend ──────────────────────────
FRONTEND_PKG="$ROOT/demo/frontend/pkg"
if [[ ! -e "$FRONTEND_PKG" ]]; then
    echo "==> Symlinking WASM pkg into frontend..."
    ln -sf "$WASM_PKG" "$FRONTEND_PKG"
fi

# ─── Step 3: Build YPIR server if needed ─────────────────────────────
YPIR_SERVER="$ROOT/server/target/release/ypir-cpu-server"
if [[ ! -x "$YPIR_SERVER" ]]; then
    echo "==> Building YPIR CPU server (this may take a few minutes, nightly required)..."
    (cd "$ROOT/server" && cargo build --release)
fi

# ─── Step 4: Determine tile database ─────────────────────────────────
if $USE_SYNTHETIC; then
    echo "==> Generating synthetic tiles..."
    mkdir -p "$TILES_DIR"
    python3 "$ROOT/../mulpir-gpu/demo/tiles/prepare_tiles.py" \
        --synthetic --synthetic-count 1000 \
        --output "$TILES_DIR"
    TILES_BIN="$TILES_DIR/tiles.bin"
    NUM_TILES=$(python3 -c "import json; m=json.load(open('$TILES_DIR/tile_mapping.json')); print(m.get('num_pir_slots', m['num_tiles']))")
    TILE_SIZE=$(python3 -c "import json; print(json.load(open('$TILES_DIR/tile_mapping.json'))['tile_size'])")
else
    # Try local tiles dir first, then fall back to mulpir-gpu tiles
    TILES_BIN="$TILES_DIR/tiles.bin"
    if [[ ! -f "$TILES_BIN" ]]; then
        FALLBACK_DIR="$ROOT/../mulpir-gpu/demo/tiles"
        if [[ -f "$FALLBACK_DIR/tiles.bin" ]]; then
            echo "==> Using shared tiles from mulpir-gpu/demo/tiles/"
            TILES_DIR="$FALLBACK_DIR"
            TILES_BIN="$FALLBACK_DIR/tiles.bin"
        else
            echo "ERROR: No tiles.bin found."
            echo "  Looked in: $TILES_DIR"
            echo "  Fallback:  $FALLBACK_DIR"
            echo "  Run: python3 $ROOT/../mulpir-gpu/demo/tiles/prepare_tiles.py --input <mbtiles> --output $TILES_DIR"
            echo "  Or use: $0 --synthetic"
            exit 1
        fi
    fi
    NUM_TILES=$(python3 -c "import json; m=json.load(open('$TILES_DIR/tile_mapping.json')); print(m.get('num_pir_slots', m['num_tiles']))")
    TILE_SIZE=$(python3 -c "import json; print(json.load(open('$TILES_DIR/tile_mapping.json'))['tile_size'])")
fi

echo "==> Database: $TILES_BIN ($NUM_TILES PIR slots, ${TILE_SIZE}B each)"

# ─── Step 5: Start YPIR server ───────────────────────────────────────
echo "==> Starting YPIR CPU server on port $YPIR_PORT..."
"$YPIR_SERVER" \
    --database "$TILES_BIN" \
    --tile-mapping "$TILES_DIR/tile_mapping.json" \
    --num-tiles "$NUM_TILES" \
    --tile-size "$TILE_SIZE" \
    --port "$YPIR_PORT" &
YPIR_PID=$!
PIDS+=($YPIR_PID)

# Wait for YPIR server to be ready
echo -n "==> Waiting for YPIR server..."
for i in $(seq 1 120); do
    if ! kill -0 "$YPIR_PID" 2>/dev/null; then
        echo " FAILED (process exited)"
        echo "    YPIR server crashed — check logs above."
        exit 1
    fi
    if (echo > /dev/tcp/localhost/$YPIR_PORT) 2>/dev/null; then
        echo " ready!"
        break
    fi
    if [[ $i -eq 120 ]]; then
        echo " TIMEOUT (port $YPIR_PORT not responding after 120s)"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# ─── Step 6: Start Flask proxy ───────────────────────────────────────
echo "==> Starting Flask proxy on port $PROXY_PORT..."
python3 "$ROOT/demo/proxy/server.py" \
    --ypir-port "$YPIR_PORT" \
    --port "$PROXY_PORT" \
    --tiles-dir "$TILES_DIR" &
PROXY_PID=$!
PIDS+=($PROXY_PID)

# Verify proxy started
sleep 2
if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "ERROR: Flask proxy failed to start on port $PROXY_PORT"
    echo "    Check if port $PROXY_PORT is already in use: ss -tlnp sport = :$PROXY_PORT"
    exit 1
fi

# ─── Step 7: Print URL ───────────────────────────────────────────────
echo ""
echo "========================================"
echo "  YPIR CPU PIR Map Demo ready!"
echo ""
echo "  Open:  http://localhost:$PROXY_PORT"
echo ""
echo "  YPIR server:  localhost:$YPIR_PORT"
echo "  Flask proxy:  localhost:$PROXY_PORT"
echo "  Tiles:        $NUM_TILES"
echo ""
echo "  Press Ctrl+C to stop"
echo "========================================"
echo ""

# Wait for any background process to exit
wait
