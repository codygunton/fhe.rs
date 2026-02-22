#!/usr/bin/env bash
#
# PIR Map Demo — MulPIR CPU backend
#
# Usage:
#   ./run_demo.sh                     # Full demo (requires tiles.bin)
#   ./run_demo.sh --synthetic         # Generate synthetic tiles first
#   ./run_demo.sh --ngrok             # Also start ngrok tunnel
#
set -euo pipefail

# All paths are relative to this script (mulpir-cpu/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MULPIR_PORT=8081
PROXY_PORT=8003
USE_SYNTHETIC=false
USE_NGROK=false
TILES_DIR="$ROOT/demo/tiles"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --synthetic)
            USE_SYNTHETIC=true
            shift
            ;;
        --ngrok)
            USE_NGROK=true
            shift
            ;;
        --mulpir-port)
            MULPIR_PORT="$2"
            shift 2
            ;;
        --proxy-port)
            PROXY_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--synthetic] [--ngrok] [--mulpir-port N] [--proxy-port N]"
            exit 1
            ;;
    esac
done

# ─── Kill stale processes on our ports ────────────────────────────────
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

kill_port "$MULPIR_PORT"
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
    # Give processes 3 seconds to exit gracefully
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

# ─── Step 1: Build mulpir-cpu-wasm WASM if needed ─────────────────────
WASM_PKG="$ROOT/../crates/mulpir-cpu-wasm/pkg"
if [[ ! -f "$WASM_PKG/mulpir_cpu_wasm_bg.wasm" ]]; then
    echo "==> Building mulpir-cpu-wasm (this takes ~30s)..."
    (cd "$ROOT/../crates/mulpir-cpu-wasm" && wasm-pack build --target web --release)
else
    echo "==> mulpir-cpu-wasm already built ($WASM_PKG)"
fi

# ─── Step 2: Symlink WASM pkg into frontend ───────────────────────────
FRONTEND_PKG="$ROOT/demo/frontend/pkg"
if [[ ! -e "$FRONTEND_PKG" ]]; then
    echo "==> Symlinking WASM pkg into frontend..."
    ln -sf "$WASM_PKG" "$FRONTEND_PKG"
fi

# ─── Step 3: Build MulPIR CPU server if needed ────────────────────────
SERVER="$ROOT/server/target/release/mulpir_cpu_server"
if [[ ! -x "$SERVER" ]]; then
    echo "==> Building MulPIR CPU server (this may take a few minutes)..."
    (cd "$ROOT/server" && cargo build --release)
fi

# ─── Step 4: Determine tile database ──────────────────────────────────
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
    TILES_BIN="$TILES_DIR/tiles.bin"
    if [[ ! -f "$TILES_BIN" ]]; then
        # Fall back to shared tile data from mulpir-gpu demo
        GPU_TILES_DIR="$ROOT/../mulpir-gpu/demo/tiles"
        if [[ -f "$GPU_TILES_DIR/tiles.bin" ]]; then
            echo "==> Using tiles from mulpir-gpu demo ($GPU_TILES_DIR)"
            TILES_DIR="$GPU_TILES_DIR"
            TILES_BIN="$TILES_DIR/tiles.bin"
        else
            echo "ERROR: $TILES_BIN not found."
            echo "  Run: python3 $ROOT/../mulpir-gpu/demo/tiles/prepare_tiles.py --input <mbtiles> --output $TILES_DIR"
            echo "  Or use: $0 --synthetic"
            exit 1
        fi
    fi
    NUM_TILES=$(python3 -c "import json; m=json.load(open('$TILES_DIR/tile_mapping.json')); print(m.get('num_pir_slots', m['num_tiles']))")
    TILE_SIZE=$(python3 -c "import json; print(json.load(open('$TILES_DIR/tile_mapping.json'))['tile_size'])")
fi

echo "==> Database: $TILES_BIN ($NUM_TILES PIR slots, ${TILE_SIZE}B each)"

# ─── Step 5: Start MulPIR CPU server ──────────────────────────────────
echo "==> Starting MulPIR CPU server on port $MULPIR_PORT..."
"$SERVER" \
    --database "$TILES_BIN" \
    --tile-mapping "$TILES_DIR/tile_mapping.json" \
    --num-tiles "$NUM_TILES" \
    --tile-size "$TILE_SIZE" \
    --port "$MULPIR_PORT" &
MULPIR_PID=$!
PIDS+=($MULPIR_PID)

# Wait for MulPIR server to be ready
echo -n "==> Waiting for MulPIR-CPU server..."
for i in $(seq 1 60); do
    if ! kill -0 "$MULPIR_PID" 2>/dev/null; then
        echo " FAILED (process exited)"
        echo "    MulPIR-CPU server crashed — check logs above."
        exit 1
    fi
    if (echo > /dev/tcp/localhost/$MULPIR_PORT) 2>/dev/null; then
        echo " ready!"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo " TIMEOUT (port $MULPIR_PORT not responding after 60s)"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# ─── Step 6: Start Flask proxy ────────────────────────────────────────
echo "==> Starting Flask proxy on port $PROXY_PORT..."
python3 "$ROOT/demo/proxy/server.py" \
    --spiral-port "$MULPIR_PORT" \
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

# ─── Step 7: Print URL ────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  MulPIR CPU PIR Map Demo ready!"
echo ""
echo "  Open:  http://localhost:$PROXY_PORT"
echo ""
echo "  MulPIR-CPU server:  localhost:$MULPIR_PORT"
echo "  Flask proxy:        localhost:$PROXY_PORT"
echo "  Tiles:              $NUM_TILES"
echo ""
echo "  Press Ctrl+C to stop"
echo "========================================"
echo ""

# ─── Step 8: Optional ngrok tunnel ────────────────────────────────────
if $USE_NGROK; then
    if command -v ngrok &>/dev/null; then
        echo "==> Starting ngrok tunnel..."
        ngrok http "$PROXY_PORT" &
        PIDS+=($!)
        sleep 2
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null || echo "unknown")
        echo "==> Public URL: $NGROK_URL"
    else
        echo "WARNING: ngrok not found, skipping tunnel"
    fi
fi

# Wait for any background process to exit
wait
