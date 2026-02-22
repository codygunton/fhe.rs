# pir-map-shared/proxy/common.py
import json
import math
import os

# BFV constants (shared reference for MulPIR param computation)
POLY_DEGREE = 8192
BITS_PER_COEFF = 20
BYTES_PER_PLAINTEXT = (BITS_PER_COEFF * POLY_DEGREE) // 8  # 20480


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def compute_pir_params(num_tiles: int, tile_size: int) -> dict:
    """Compute MulPIR (BFV) dimension parameters from tile database size."""
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


def load_tile_mapping(tiles_dir: str) -> dict:
    """Load and return tile_mapping.json as a dict, or raise FileNotFoundError."""
    mapping_path = os.path.join(tiles_dir, "tile_mapping.json")
    with open(mapping_path, "r") as f:
        return json.load(f)


def get_num_pir_slots(mapping: dict) -> int:
    """Extract num_pir_slots from a tile_mapping dict (with backwards compat)."""
    num_pir_slots = mapping.get("num_pir_slots")
    if num_pir_slots is None:
        num_pir_slots = 0
        for value in mapping.get("tiles", {}).values():
            if isinstance(value, list):
                num_pir_slots += len(value)
            else:
                num_pir_slots += 1
    return num_pir_slots
