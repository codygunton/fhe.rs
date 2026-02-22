#!/usr/bin/env bash
#
# PIR Map Demo — Launch all components
#
# Usage:
#   ./run_demo.sh                     # Full demo (requires tiles.bin)
#   ./run_demo.sh --test-tiles        # Use test_vectors (100 tiles)
#   ./run_demo.sh --synthetic         # Generate synthetic tiles first
#   ./run_demo.sh --ngrok             # Also start ngrok tunnel
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEMO_DIR="$PROJECT_ROOT/pir-map-demo"

GPU_PORT=8080
PROXY_PORT=8000
USE_TEST_TILES=false
USE_SYNTHETIC=false
USE_NGROK=false
TILES_DIR="$DEMO_DIR/tiles"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-tiles)
            USE_TEST_TILES=true
            shift
            ;;
        --synthetic)
            USE_SYNTHETIC=true
            shift
            ;;
        --ngrok)
            USE_NGROK=true
            shift
            ;;
        --gpu-port)
            GPU_PORT="$2"
            shift 2
            ;;
        --proxy-port)
            PROXY_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test-tiles] [--synthetic] [--ngrok] [--gpu-port N] [--proxy-port N]"
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

kill_port "$GPU_PORT"
kill_port "$PROXY_PORT"

# Track background PIDs for cleanup
PIDS=()
cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# ─── Step 1: Build WASM if needed ─────────────────────────────────────
WASM_PKG="$PROJECT_ROOT/crates/fhe-wasm/pkg"
if [[ ! -f "$WASM_PKG/fhe_wasm_bg.wasm" ]]; then
    echo "==> Building WASM (this takes ~10s)..."
    (cd "$PROJECT_ROOT/crates/fhe-wasm" && wasm-pack build --target web --release)
else
    echo "==> WASM already built ($WASM_PKG)"
fi

# ─── Step 2: Symlink WASM pkg into frontend ───────────────────────────
FRONTEND_PKG="$DEMO_DIR/frontend/pkg"
if [[ ! -e "$FRONTEND_PKG" ]]; then
    echo "==> Symlinking WASM pkg into frontend..."
    ln -sf "$WASM_PKG" "$FRONTEND_PKG"
fi

# ─── Step 3: Determine tile database ──────────────────────────────────
if $USE_TEST_TILES; then
    TILES_DIR="$PROJECT_ROOT/mulpir-gpu-server/test_vectors"
    TILES_BIN="$TILES_DIR/tiles.bin"
    # Read num_tiles from params.json
    NUM_TILES=$(python3 -c "import json; print(json.load(open('$TILES_DIR/params.json'))['num_tiles'])")
    TILE_SIZE=$(python3 -c "import json; print(json.load(open('$TILES_DIR/params.json'))['tile_size'])")

    # Create tile_mapping.json from params.json if it doesn't exist
    if [[ ! -f "$TILES_DIR/tile_mapping.json" ]]; then
        echo "==> Generating tile_mapping.json for test vectors..."
        python3 -c "
import json, math
params = json.load(open('$TILES_DIR/params.json'))
n = params['num_tiles']
# Map tiles to a grid at zoom 4 so MapLibre will actually request them
z = 4
cols = int(math.ceil(math.sqrt(n)))
rows = int(math.ceil(n / cols))
tiles = {}
idx = 0
for row in range(rows):
    for col in range(cols):
        if idx >= n:
            break
        tiles[f'{z}/{col}/{2 + row}'] = idx
        idx += 1
def tile_center(z, x0, x1, y0, y1):
    nn = 1 << z
    lon = ((x0 + x1) / 2) / nn * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ((y0 + y1) / 2) / nn))))
    return [round(lon, 1), round(lat, 1)]
mapping = {
    'num_tiles': n,
    'tile_size': params['tile_size'],
    'center': tile_center(z, 0, cols, 2, 2 + rows),
    'min_zoom': z,
    'max_zoom': z,
    'tiles': tiles
}
json.dump(mapping, open('$TILES_DIR/tile_mapping.json', 'w'), indent=2)
print(f'Created tile_mapping.json with {n} tiles at z={z}')
"
    fi
elif $USE_SYNTHETIC; then
    echo "==> Generating synthetic tiles..."
    mkdir -p "$TILES_DIR"
    python3 "$DEMO_DIR/tiles/prepare_tiles.py" \
        --synthetic --synthetic-count 1000 \
        --output "$TILES_DIR"
    TILES_BIN="$TILES_DIR/tiles.bin"
    NUM_TILES=$(python3 -c "import json; m=json.load(open('$TILES_DIR/tile_mapping.json')); print(m.get('num_pir_slots', m['num_tiles']))")
    TILE_SIZE=$(python3 -c "import json; print(json.load(open('$TILES_DIR/tile_mapping.json'))['tile_size'])")
else
    TILES_BIN="$TILES_DIR/tiles.bin"
    if [[ ! -f "$TILES_BIN" ]]; then
        echo "ERROR: $TILES_BIN not found."
        echo "  Run: python3 $DEMO_DIR/tiles/prepare_tiles.py --input <mbtiles> --output $TILES_DIR"
        echo "  Or use: $0 --test-tiles  or  $0 --synthetic"
        exit 1
    fi
    NUM_TILES=$(python3 -c "import json; m=json.load(open('$TILES_DIR/tile_mapping.json')); print(m.get('num_pir_slots', m['num_tiles']))")
    TILE_SIZE=$(python3 -c "import json; print(json.load(open('$TILES_DIR/tile_mapping.json'))['tile_size'])")
fi

echo "==> Database: $TILES_BIN ($NUM_TILES PIR slots, ${TILE_SIZE}B each)"

# ─── Step 4: Start GPU server ─────────────────────────────────────────
GPU_SERVER="$PROJECT_ROOT/mulpir-gpu-server/build/mulpir_server"
if [[ ! -x "$GPU_SERVER" ]]; then
    echo "ERROR: GPU server not found at $GPU_SERVER"
    echo "  Build it first: cd mulpir-gpu-server && mkdir -p build && cd build && cmake .. && make -j"
    exit 1
fi

echo "==> Starting GPU server on port $GPU_PORT..."
GPU_ARGS=(
    --database "$TILES_BIN"
    --num-tiles "$NUM_TILES"
    --tile-size "$TILE_SIZE"
    --port "$GPU_PORT"
)
# Test vectors from generate_test_vectors have a 16-byte header; real tiles don't.
if $USE_TEST_TILES; then
    GPU_ARGS+=(--data-offset 16)
fi
$GPU_SERVER "${GPU_ARGS[@]}" &
GPU_PID=$!
PIDS+=($GPU_PID)

# Wait for GPU server to be ready (check if process is alive AND port is listening)
echo -n "==> Waiting for GPU server..."
for i in $(seq 1 30); do
    if ! kill -0 "$GPU_PID" 2>/dev/null; then
        echo " FAILED (process exited)"
        echo "    GPU server crashed — check for CUDA OOM or database errors."
        echo "    Try: nvidia-smi  to check GPU memory usage."
        exit 1
    fi
    if (echo > /dev/tcp/localhost/$GPU_PORT) 2>/dev/null; then
        echo " ready!"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo " TIMEOUT (port $GPU_PORT not responding after 30s)"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# ─── Step 5: Start Flask proxy ────────────────────────────────────────
echo "==> Starting Flask proxy on port $PROXY_PORT..."
python3 "$DEMO_DIR/proxy/server.py" \
    --gpu-port "$GPU_PORT" \
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

# ─── Step 6: Print URL ────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  PIR Map Demo ready!"
echo ""
echo "  Open:  http://localhost:$PROXY_PORT"
echo ""
echo "  GPU server:  localhost:$GPU_PORT"
echo "  Flask proxy: localhost:$PROXY_PORT"
echo "  Tiles:       $NUM_TILES"
echo ""
echo "  Press Ctrl+C to stop"
echo "========================================"
echo ""

# ─── Step 7: Optional ngrok tunnel ────────────────────────────────────
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
