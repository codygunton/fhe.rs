"""
Flask proxy server for the PIR map demo.

Translates HTTP requests from the browser into TCP messages
for the MulPIR GPU server, and serves static frontend files.
"""

import argparse
import json
import logging
import math
import os
import socket
import struct
import subprocess

from flask import Flask, Response, jsonify, request, send_from_directory

# Wire protocol message types (must match GPU server)
MSG_SET_GALOIS_KEY = 0x01
MSG_SET_RELIN_KEY = 0x02
MSG_QUERY = 0x03

# Wire protocol response status codes
STATUS_OK = 0x00
STATUS_ERROR_INVALID_MESSAGE = 0x01
STATUS_ERROR_INVALID_QUERY = 0x02
STATUS_ERROR_PROCESSING_FAILED = 0x03
STATUS_ERROR_NOT_READY = 0x04

STATUS_NAMES = {
    STATUS_OK: "ok",
    STATUS_ERROR_INVALID_MESSAGE: "invalid_message",
    STATUS_ERROR_INVALID_QUERY: "invalid_query",
    STATUS_ERROR_PROCESSING_FAILED: "processing_failed",
    STATUS_ERROR_NOT_READY: "not_ready",
}

# BFV encryption parameters (fixed for this scheme)
POLY_DEGREE = 8192
BITS_PER_COEFF = 20
BYTES_PER_PLAINTEXT = (BITS_PER_COEFF * POLY_DEGREE) // 8  # 20480

logger = logging.getLogger(__name__)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a socket, looping until complete."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Connection closed after {len(buf)}/{n} bytes"
            )
        buf.extend(chunk)
    return bytes(buf)


def send_to_gpu(
    host: str, port: int, msg_type: int, payload: bytes
) -> tuple[int, bytes]:
    """
    Send a message to the GPU server over TCP and return the response.

    Opens a new TCP connection for each message. The wire protocol is:
      Request:  [msg_type: u32 LE] [payload_len: u32 LE] [payload bytes]
      Response: [status: u32 LE]   [resp_len: u32 LE]    [resp bytes]
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        header = struct.pack("<II", msg_type, len(payload))
        sock.sendall(header + payload)

        resp_header = recv_exact(sock, 8)
        status, resp_len = struct.unpack("<II", resp_header)
        resp_payload = recv_exact(sock, resp_len) if resp_len > 0 else b""

        return status, resp_payload
    finally:
        sock.close()


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def compute_pir_params(num_tiles: int, tile_size: int) -> dict:
    """
    Compute PIR dimension parameters from the tile database size.

    Returns a dict with dim1, dim2, expansion_level, and intermediate values.
    """
    elements_per_plaintext = (BYTES_PER_PLAINTEXT * 8) // (tile_size * 8)
    num_rows = math.ceil(num_tiles / elements_per_plaintext)
    dim1 = math.ceil(math.sqrt(num_rows))
    dim2 = math.ceil(num_rows / dim1)
    expansion_level = math.ceil(
        math.log2(next_power_of_two(dim1 + dim2))
    )

    return {
        "poly_degree": POLY_DEGREE,
        "bits_per_coeff": BITS_PER_COEFF,
        "bytes_per_plaintext": BYTES_PER_PLAINTEXT,
        "elements_per_plaintext": elements_per_plaintext,
        "num_tiles": num_tiles,
        "tile_size": tile_size,
        "num_rows": num_rows,
        "dim1": dim1,
        "dim2": dim2,
        "expansion_level": expansion_level,
    }


def create_app(gpu_host: str, gpu_port: int, tiles_dir: str) -> Flask:
    """Create and configure the Flask application."""
    frontend_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "frontend")
    )

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

    # ------------------------------------------------------------------ #
    # Key setup endpoints
    # ------------------------------------------------------------------ #

    @app.route("/api/setup-galois-key", methods=["POST"])
    def setup_galois_key():
        payload = request.get_data()
        logger.info("Sending galois key (%d bytes) to GPU server", len(payload))
        try:
            status, resp = send_to_gpu(
                gpu_host, gpu_port, MSG_SET_GALOIS_KEY, payload
            )
        except Exception as exc:
            logger.error("GPU connection failed: %s", exc)
            return jsonify({"status": "error", "message": str(exc)}), 502

        if status == STATUS_OK:
            return jsonify({"status": "ok"})
        msg = STATUS_NAMES.get(status, f"unknown_error_{status:#x}")
        logger.warning("GPU returned error for galois key: %s", msg)
        return jsonify({"status": "error", "message": msg}), 400

    @app.route("/api/setup-relin-key", methods=["POST"])
    def setup_relin_key():
        payload = request.get_data()
        logger.info("Sending relin key (%d bytes) to GPU server", len(payload))
        try:
            status, resp = send_to_gpu(
                gpu_host, gpu_port, MSG_SET_RELIN_KEY, payload
            )
        except Exception as exc:
            logger.error("GPU connection failed: %s", exc)
            return jsonify({"status": "error", "message": str(exc)}), 502

        if status == STATUS_OK:
            return jsonify({"status": "ok"})
        msg = STATUS_NAMES.get(status, f"unknown_error_{status:#x}")
        logger.warning("GPU returned error for relin key: %s", msg)
        return jsonify({"status": "error", "message": msg}), 400

    # ------------------------------------------------------------------ #
    # Query endpoint — returns raw binary ciphertext
    # ------------------------------------------------------------------ #

    @app.route("/api/query", methods=["POST"])
    def query():
        payload = request.get_data()
        logger.info("Sending query (%d bytes) to GPU server", len(payload))
        try:
            status, resp = send_to_gpu(
                gpu_host, gpu_port, MSG_QUERY, payload
            )
        except Exception as exc:
            logger.error("GPU connection failed: %s", exc)
            return jsonify({"status": "error", "message": str(exc)}), 502

        if status == STATUS_OK:
            return Response(resp, content_type="application/octet-stream")
        msg = STATUS_NAMES.get(status, f"unknown_error_{status:#x}")
        logger.warning("GPU returned error for query: %s", msg)
        return jsonify({"status": "error", "message": msg}), 400

    # ------------------------------------------------------------------ #
    # GPU metrics
    # ------------------------------------------------------------------ #

    @app.route("/api/metrics", methods=["GET"])
    def metrics():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return jsonify({"error": "nvidia-smi failed"}), 500

            # Parse CSV: "utilization, mem_used, mem_total, temp"
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return jsonify(
                {
                    "gpu_utilization": int(parts[0]),
                    "memory_used_mb": int(parts[1]),
                    "memory_total_mb": int(parts[2]),
                    "temperature_c": int(parts[3]),
                }
            )
        except FileNotFoundError:
            return jsonify({"error": "nvidia-smi not found"}), 500
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ------------------------------------------------------------------ #
    # PIR parameters (derived from tile_mapping.json)
    # ------------------------------------------------------------------ #

    @app.route("/api/params", methods=["GET"])
    def params():
        mapping_path = os.path.join(tiles_dir, "tile_mapping.json")
        try:
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
        except FileNotFoundError:
            return jsonify({"error": "tile_mapping.json not found"}), 404
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"Invalid JSON: {exc}"}), 500

        tile_size = mapping.get("tile_size", 0)

        if tile_size <= 0:
            return (
                jsonify({"error": "tile_size missing or invalid in mapping"}),
                500,
            )

        # Use total PIR slot count for dimension calculation.
        # Split tiles occupy multiple slots, so num_pir_slots >= num_tiles.
        num_pir_slots = mapping.get("num_pir_slots")
        if num_pir_slots is None:
            # Backwards compat: count slots from tile entries
            num_pir_slots = 0
            for value in mapping.get("tiles", {}).values():
                if isinstance(value, list):
                    num_pir_slots += len(value)
                else:
                    num_pir_slots += 1

        return jsonify(compute_pir_params(num_pir_slots, tile_size))

    # ------------------------------------------------------------------ #
    # Tile mapping file
    # ------------------------------------------------------------------ #

    @app.route("/api/tile-mapping", methods=["GET"])
    def tile_mapping():
        mapping_path = os.path.join(tiles_dir, "tile_mapping.json")
        try:
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            return jsonify(mapping)
        except FileNotFoundError:
            return jsonify({"error": "tile_mapping.json not found"}), 404
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"Invalid JSON: {exc}"}), 500

    # ------------------------------------------------------------------ #
    # Static file serving (frontend)
    # ------------------------------------------------------------------ #

    @app.route("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        return send_from_directory(frontend_dir, path)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flask proxy server for the PIR map demo"
    )
    parser.add_argument(
        "--gpu-host",
        default="localhost",
        help="Hostname of the MulPIR GPU server (default: localhost)",
    )
    parser.add_argument(
        "--gpu-port",
        type=int,
        default=8080,
        help="Port of the MulPIR GPU server (default: 8080)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for this HTTP proxy server (default: 8000)",
    )
    parser.add_argument(
        "--tiles-dir",
        required=True,
        help="Path to directory containing tiles.bin and tile_mapping.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    tiles_dir = os.path.abspath(args.tiles_dir)
    if not os.path.isdir(tiles_dir):
        logger.error("Tiles directory does not exist: %s", tiles_dir)
        raise SystemExit(1)

    mapping_path = os.path.join(tiles_dir, "tile_mapping.json")
    if not os.path.isfile(mapping_path):
        logger.warning("tile_mapping.json not found in %s", tiles_dir)

    app = create_app(args.gpu_host, args.gpu_port, tiles_dir)

    logger.info(
        "Starting proxy server on :%d  (GPU server at %s:%d, tiles: %s)",
        args.port,
        args.gpu_host,
        args.gpu_port,
        tiles_dir,
    )
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
