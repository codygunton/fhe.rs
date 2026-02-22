#!/usr/bin/env python3
"""Extract vector tiles from an MBTiles file and prepare them for PIR database.

Oversized tiles (larger than one PIR slot) are split across consecutive slots.

Input: MBTiles file (SQLite database) from Planetiler with gzip-compressed PBF vector tiles.
Output:
    - tiles.bin: concatenated fixed-size PIR slots
    - tile_mapping.json: mapping from z/x/y to PIR slot index (or array of indices)
"""

import argparse
import gzip
import json
import logging
import math
import os
import sqlite3
import struct
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Default bounding box (approximate continental US).
DEFAULT_BBOX = {
    "west": -125.0,
    "east": -66.0,
    "north": 50.0,
    "south": 24.0,
}

# Active bbox — set by CLI args.
ACTIVE_BBOX = dict(DEFAULT_BBOX)


def lon_to_tile_x(lon: float, zoom: int) -> int:
    """Convert longitude to tile X coordinate at a given zoom level."""
    return int((lon + 180.0) / 360.0 * (1 << zoom))


def lat_to_tile_y(lat: float, zoom: int) -> int:
    """Convert latitude to tile Y coordinate (XYZ / slippy map convention)."""
    lat_rad = math.radians(lat)
    n = 1 << zoom
    return int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )


def tile_ranges_for_zoom(zoom: int) -> tuple[int, int, int, int]:
    """Return (x_min, x_max, y_min, y_max) in XYZ convention for the US bbox."""
    x_min = lon_to_tile_x(ACTIVE_BBOX["west"], zoom)
    x_max = lon_to_tile_x(ACTIVE_BBOX["east"], zoom)
    # Note: lat_to_tile_y flips N/S — higher latitude gives smaller Y.
    y_min = lat_to_tile_y(ACTIVE_BBOX["north"], zoom)
    y_max = lat_to_tile_y(ACTIVE_BBOX["south"], zoom)
    return x_min, x_max, y_min, y_max


def is_in_us_bbox(zoom: int, x: int, y: int) -> bool:
    """Check whether tile (z, x, y) in XYZ convention falls within the US bbox."""
    x_min, x_max, y_min, y_max = tile_ranges_for_zoom(zoom)
    return x_min <= x <= x_max and y_min <= y <= y_max


def pack_tile_into_slots(data: bytes, tile_size: int) -> list[bytearray]:
    """Pack tile data into one or more fixed-size PIR slots.

    Format:
      First slot:  [data_len: u32 LE][gzip bytes...][zero padding]
      Extra slots: [continuation gzip bytes...][zero padding]

    data_len covers the total gzip payload across all slots.
    """
    compressed_size = len(data)
    first_capacity = tile_size - 4  # 4 bytes for length header

    # Build the wire payload: [u32 length][data]
    payload = bytearray(struct.pack("<I", compressed_size))
    payload.extend(data)

    slots: list[bytearray] = []
    offset = 0
    while offset < len(payload):
        chunk = bytearray(payload[offset : offset + tile_size])
        # Pad to exactly tile_size
        if len(chunk) < tile_size:
            chunk.extend(b"\x00" * (tile_size - len(chunk)))
        slots.append(chunk)
        offset += tile_size

    return slots


def extract_tiles_from_mbtiles(
    mbtiles_path: str,
    max_zoom: int,
    max_tiles: int,
    tile_size: int,
) -> tuple[bytearray, dict]:
    """Read tiles from an MBTiles file, filter to US bbox, split oversized tiles.

    Returns (slot_data, tile_mapping) where slot_data is a bytearray of
    concatenated fixed-size PIR slots and tile_mapping is the JSON-serialisable
    metadata dict.
    """
    if not os.path.isfile(mbtiles_path):
        log.error("MBTiles file not found: %s", mbtiles_path)
        sys.exit(1)

    conn = sqlite3.connect(mbtiles_path)
    cursor = conn.cursor()

    query = (
        "SELECT zoom_level, tile_column, tile_row, tile_data "
        "FROM tiles "
        "WHERE zoom_level <= ? "
        "ORDER BY zoom_level, tile_column, tile_row"
    )
    cursor.execute(query, (max_zoom,))

    slot_data = bytearray()
    tiles_dict: dict[str, int | list[int]] = {}
    tile_count = 0
    slot_index = 0
    split_count = 0
    skipped_outside = 0

    for zoom_level, tile_column, tile_row, data in cursor:
        # MBTiles uses TMS Y-axis convention — convert to XYZ.
        y_xyz = (1 << zoom_level) - 1 - tile_row

        if not is_in_us_bbox(zoom_level, tile_column, y_xyz):
            skipped_outside += 1
            continue

        slots = pack_tile_into_slots(data, tile_size)
        num_slots = len(slots)

        for s in slots:
            slot_data.extend(s)

        key = f"{zoom_level}/{tile_column}/{y_xyz}"
        if num_slots == 1:
            tiles_dict[key] = slot_index
        else:
            tiles_dict[key] = list(range(slot_index, slot_index + num_slots))
            split_count += 1

        slot_index += num_slots
        tile_count += 1

        if tile_count >= max_tiles:
            log.info("Reached max tiles limit (%d), stopping.", max_tiles)
            break

    conn.close()

    log.info(
        "Extracted %d tiles (%d split across multiple slots) into %d PIR slots",
        tile_count,
        split_count,
        slot_index,
    )
    log.info("Skipped %d tiles outside US bbox", skipped_outside)

    center_lon = (ACTIVE_BBOX["west"] + ACTIVE_BBOX["east"]) / 2
    center_lat = (ACTIVE_BBOX["south"] + ACTIVE_BBOX["north"]) / 2

    tile_mapping = {
        "num_tiles": tile_count,
        "num_pir_slots": slot_index,
        "tile_size": tile_size,
        "center": [round(center_lon, 2), round(center_lat, 2)],
        "min_zoom": 0,
        "max_zoom": max_zoom,
        "tiles": tiles_dict,
    }

    return slot_data, tile_mapping


def generate_synthetic_tiles(
    count: int,
    tile_size: int,
    max_zoom: int,
) -> tuple[bytearray, dict]:
    """Generate numbered synthetic tiles for development/testing."""
    slot_data = bytearray()
    tiles_dict: dict[str, int | list[int]] = {}
    tile_count = 0
    slot_index = 0

    for zoom in range(max_zoom + 1):
        if tile_count >= count:
            break

        x_min, x_max, y_min, y_max = tile_ranges_for_zoom(zoom)
        max_coord = (1 << zoom) - 1
        x_min = max(0, x_min)
        x_max = min(max_coord, x_max)
        y_min = max(0, y_min)
        y_max = min(max_coord, y_max)

        for x in range(x_min, x_max + 1):
            if tile_count >= count:
                break
            for y in range(y_min, y_max + 1):
                if tile_count >= count:
                    break

                content = json.dumps({
                    "z": zoom,
                    "x": x,
                    "y": y,
                    "index": tile_count,
                    "synthetic": True,
                }).encode("utf-8")

                compressed = gzip.compress(content)
                slots = pack_tile_into_slots(compressed, tile_size)

                for s in slots:
                    slot_data.extend(s)

                key = f"{zoom}/{x}/{y}"
                if len(slots) == 1:
                    tiles_dict[key] = slot_index
                else:
                    tiles_dict[key] = list(range(slot_index, slot_index + len(slots)))

                slot_index += len(slots)
                tile_count += 1

    log.info("Generated %d synthetic tiles in %d PIR slots", tile_count, slot_index)

    tile_mapping = {
        "num_tiles": tile_count,
        "num_pir_slots": slot_index,
        "tile_size": tile_size,
        "center": [-98.5, 39.8],
        "min_zoom": 0,
        "max_zoom": max_zoom,
        "tiles": tiles_dict,
    }

    return slot_data, tile_mapping


def write_output(
    output_dir: str,
    slot_data: bytearray,
    tile_mapping: dict,
) -> None:
    """Write tiles.bin and tile_mapping.json to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    tiles_bin_path = os.path.join(output_dir, "tiles.bin")
    mapping_path = os.path.join(output_dir, "tile_mapping.json")

    num_pir_slots = tile_mapping["num_pir_slots"]
    tile_size = tile_mapping["tile_size"]
    expected_size = num_pir_slots * tile_size

    assert len(slot_data) == expected_size, (
        f"slot_data size mismatch: {len(slot_data)} != {expected_size} "
        f"({num_pir_slots} slots * {tile_size} bytes)"
    )

    with open(tiles_bin_path, "wb") as f:
        f.write(slot_data)
    log.info(
        "Wrote %s (%d bytes, %d PIR slots)",
        tiles_bin_path,
        len(slot_data),
        num_pir_slots,
    )

    with open(mapping_path, "w") as f:
        json.dump(tile_mapping, f, indent=2)
    log.info("Wrote %s", mapping_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract vector tiles from MBTiles and prepare for PIR database.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to MBTiles file (required unless --synthetic)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory for tiles.bin and tile_mapping.json (default: current dir)",
    )
    parser.add_argument(
        "--max-zoom",
        type=int,
        default=11,
        help="Maximum zoom level to include (default: 11)",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=100000,
        help="Maximum number of tiles to include (default: 100000)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=20480,
        help="Fixed PIR slot size in bytes (default: 20480)",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Bounding box as west,south,east,north (default: continental US)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic test tiles instead of reading MBTiles",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=1000,
        help="Number of synthetic tiles to generate (default: 1000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        ACTIVE_BBOX["west"], ACTIVE_BBOX["south"], ACTIVE_BBOX["east"], ACTIVE_BBOX["north"] = parts
        log.info("Using bbox: %s", ACTIVE_BBOX)

    if args.synthetic:
        log.info(
            "Generating %d synthetic tiles (tile_size=%d, max_zoom=%d)",
            args.synthetic_count,
            args.tile_size,
            args.max_zoom,
        )
        slot_data, tile_mapping = generate_synthetic_tiles(
            count=args.synthetic_count,
            tile_size=args.tile_size,
            max_zoom=args.max_zoom,
        )
    else:
        if args.input is None:
            log.error("--input is required when not using --synthetic")
            sys.exit(1)
        log.info(
            "Extracting tiles from %s (max_zoom=%d, max_tiles=%d, tile_size=%d)",
            args.input,
            args.max_zoom,
            args.max_tiles,
            args.tile_size,
        )
        slot_data, tile_mapping = extract_tiles_from_mbtiles(
            mbtiles_path=args.input,
            max_zoom=args.max_zoom,
            max_tiles=args.max_tiles,
            tile_size=args.tile_size,
        )

    write_output(args.output, slot_data, tile_mapping)

    num_tiles = tile_mapping["num_tiles"]
    num_pir_slots = tile_mapping["num_pir_slots"]
    total_bytes = num_pir_slots * args.tile_size
    log.info(
        "Done: %d tiles in %d PIR slots, %.2f MB total",
        num_tiles,
        num_pir_slots,
        total_bytes / (1024 * 1024),
    )


if __name__ == "__main__":
    main()
