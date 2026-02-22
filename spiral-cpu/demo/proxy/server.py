"""
Flask proxy server for the Spiral-CPU PIR map demo.

Forwards API requests to the Rust spiral-cpu server over HTTP,
and serves the static frontend files.
"""

import argparse
import json
import logging
import os
import sys

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

# Add pir-map-shared/proxy to path for shared helpers
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "pir-map-shared", "proxy")
    ),
)
from common import load_tile_mapping, get_num_pir_slots

logger = logging.getLogger(__name__)

SHARED_FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "pir-map-shared", "frontend")
)


def create_app(spiral_host: str, spiral_port: int, tiles_dir: str) -> Flask:
    frontend_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "frontend")
    )

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

    spiral_base = f"http://{spiral_host}:{spiral_port}"

    # ------------------------------------------------------------------ #
    # Spiral PIR setup endpoint
    # ------------------------------------------------------------------ #

    @app.route("/api/setup", methods=["POST"])
    def setup():
        payload = request.get_data()
        logger.info("Forwarding /api/setup (%d bytes) to spiral server", len(payload))
        try:
            resp = requests.post(
                f"{spiral_base}/api/setup",
                data=payload,
                headers={"Content-Type": "application/octet-stream"},
                timeout=30,
            )
            resp.raise_for_status()
            return Response(resp.content, content_type="text/plain")
        except Exception as exc:
            logger.error("Spiral server setup failed: %s", exc)
            return jsonify({"status": "error", "message": str(exc)}), 502

    # ------------------------------------------------------------------ #
    # Spiral PIR query endpoint
    # ------------------------------------------------------------------ #

    @app.route("/api/private-read", methods=["POST"])
    def private_read():
        payload = request.get_data()
        logger.info("Forwarding /api/private-read (%d bytes) to spiral server", len(payload))
        try:
            resp = requests.post(
                f"{spiral_base}/api/private-read",
                data=payload,
                headers={"Content-Type": "application/octet-stream"},
                timeout=60,
            )
            resp.raise_for_status()
            return Response(resp.content, content_type="application/octet-stream")
        except Exception as exc:
            logger.error("Spiral server query failed: %s", exc)
            return jsonify({"status": "error", "message": str(exc)}), 502

    # ------------------------------------------------------------------ #
    # PIR parameters — forwarded from Spiral server
    # ------------------------------------------------------------------ #

    @app.route("/api/params", methods=["GET"])
    def params():
        try:
            resp = requests.get(f"{spiral_base}/api/params", timeout=5)
            resp.raise_for_status()
            return jsonify(resp.json())
        except Exception as exc:
            logger.error("Failed to fetch params from spiral server: %s", exc)
            return jsonify({"error": str(exc)}), 502

    # ------------------------------------------------------------------ #
    # Tile mapping file
    # ------------------------------------------------------------------ #

    @app.route("/api/tile-mapping", methods=["GET"])
    def tile_mapping():
        try:
            mapping = load_tile_mapping(tiles_dir)
            return jsonify(mapping)
        except FileNotFoundError:
            return jsonify({"error": "tile_mapping.json not found"}), 404
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"Invalid JSON: {exc}"}), 500

    # ------------------------------------------------------------------ #
    # CPU metrics
    # ------------------------------------------------------------------ #

    @app.route("/api/metrics", methods=["GET"])
    def metrics():
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            return jsonify({
                "cpu_percent": round(cpu, 1),
                "memory_used_mb": int(mem.used / 1024 / 1024),
                "memory_total_mb": int(mem.total / 1024 / 1024),
            })
        except ImportError:
            return jsonify({"error": "psutil not installed"}), 500
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ------------------------------------------------------------------ #
    # Shared JS modules (served from pir-map-shared/frontend/)
    # ------------------------------------------------------------------ #

    @app.route("/shared/<path:filename>")
    def shared_files(filename):
        return send_from_directory(SHARED_FRONTEND_DIR, filename)

    # ------------------------------------------------------------------ #
    # Static frontend files
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
        description="Flask proxy server for the Spiral-CPU PIR map demo"
    )
    parser.add_argument(
        "--spiral-host",
        default="localhost",
        help="Hostname of the Spiral-CPU server (default: localhost)",
    )
    parser.add_argument(
        "--spiral-port",
        type=int,
        default=8081,
        help="Port of the Spiral-CPU server (default: 8081)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="Port for this HTTP proxy server (default: 8002)",
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

    app = create_app(args.spiral_host, args.spiral_port, tiles_dir)

    logger.info(
        "Starting Spiral-CPU proxy on :%d  (Spiral server at %s:%d, tiles: %s)",
        args.port,
        args.spiral_host,
        args.spiral_port,
        tiles_dir,
    )
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
