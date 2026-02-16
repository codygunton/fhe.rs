# Tile Splitting Implementation Plan

## Executive Summary

**Problem**: The PIR map demo skips ~9,179 tiles (10.7% of the US dataset) that exceed the 20KB BFV plaintext capacity. This particularly affects zoom levels 0-6 where nearly all tiles are oversized (40-200KB compressed), leaving the map blank at low zoom. The result: users see nothing until z7+.

**Solution**: Split oversized tiles across multiple consecutive 20KB PIR database slots. The client issues parallel queries for each slot, concatenates the decrypted chunks, and decompresses the reassembled tile. The tile mapping format is extended backwards-compatibly: single-slot tiles remain integers, multi-slot tiles become arrays of consecutive slot indices.

**Technical approach**:
- `prepare_tiles.py` chunks oversized tile data into 20KB slots with a length header on the first chunk
- `tile_mapping.json` uses `[startIdx, startIdx+1, ...]` arrays for split tiles
- Frontend fires parallel PIR queries for multi-slot tiles, concatenates before decompression
- Proxy computes PIR dimensions from total *slot* count (not logical tile count)
- GPU server is unchanged — it just sees a larger flat database of 20KB slots

**Budget**: ~9,179 oversize tiles expand to ~14K extra slots. Total: ~90K slots (fits comfortably in 32GB VRAM at 128KB/slot = ~11.5GB).

### Data Flow

```
Tile > 20KB                                  Tile <= 20KB
    |                                             |
    v                                             v
[chunk_0: len_hdr+data][chunk_1: data]...    [len_hdr+data+padding]
    |            |                                |
    v            v                                v
 slot N       slot N+1  ...                    slot M
    |            |                                |
    ---- PIR query per slot (parallel) -----------|
    |            |                                |
    v            v                                v
 decrypt     decrypt                           decrypt
    |            |                                |
    --- concatenate chunks ---                    |
    |                                             |
    v                                             v
 strip padding + decompress gzip → raw PBF vector tile
```

## Goals & Objectives

### Primary Goals
- Include ALL tiles z0-z11 in the PIR database (currently missing z2-z6 almost entirely)
- Full US map coverage from zoom 0 to zoom 11 with no blank gaps

### Secondary Objectives
- Keep single-slot path fast (no regression for the 90% of tiles that already fit)
- Backwards-compatible mapping format (single-slot tiles unchanged)
- Parallel multi-slot fetching to minimize latency impact

## Solution Overview

### Approach

Each 20KB PIR slot is the atomic unit. Oversized tiles are split into consecutive slots. The first slot carries a 4-byte length header so the client knows how many bytes of real data to extract after concatenation. The mapping tells the client which slots belong to each tile.

### Key Components

1. **`prepare_tiles.py`**: Splits oversized tiles into consecutive 20KB chunks with length-prefix on first chunk
2. **`tile_mapping.json`**: Extended to use arrays for multi-slot tiles, adds `num_pir_slots` field
3. **`app.js` (frontend)**: Detects multi-slot tiles, fires parallel queries, concatenates before decompress
4. **`server.py` (proxy)**: Uses `num_pir_slots` for PIR dimension calculation
5. **`run_demo.sh`**: Reads `num_pir_slots` from mapping instead of `num_tiles` for GPU server `--num-tiles`

### Expected Outcomes
- All ~85,632 US tiles at z0-z11 included in the database (~90K PIR slots total)
- Map renders from z0 onward (low zoom tiles fully visible)
- Single-slot tiles (90%) have zero latency regression
- Multi-slot tiles add ~Nx latency (N = number of slots) but queries run in parallel

## Implementation Tasks

### Visual Dependency Tree

```
pir-map-demo/
├── tiles/
│   └── prepare_tiles.py          (Task #1: Add tile splitting + length-prefix chunking)
│
├── frontend/
│   └── app.js                    (Task #2: Multi-slot fetch + concatenation in fetchTileViaPIR)
│
├── proxy/
│   └── server.py                 (Task #3: Use num_pir_slots for PIR param calculation)
│
└── scripts/
    └── run_demo.sh               (Task #3: Read num_pir_slots for GPU server --num-tiles)
```

### Execution Plan

#### Group A: Data Pipeline (execute first — generates new tiles.bin)

- [x] **Task #1**: Update `prepare_tiles.py` to split oversized tiles into consecutive slots
  - File: `pir-map-demo/tiles/prepare_tiles.py`
  - Changes:
    - [ ] Remove the `tile_size - 4` overflow skip — instead, split tiles that exceed `tile_size - 4` bytes
    - [ ] Splitting logic:
      - First chunk: `[data_len: u32 LE][gzip_data_bytes][zero_padding]` (same as current single-slot format)
      - `data_len` = total compressed tile size (spans all chunks)
      - Subsequent chunks: `[continuation_data_bytes][zero_padding]`
      - Usable bytes per chunk: first chunk = `tile_size - 4`, subsequent chunks = `tile_size`
      - Number of chunks: `ceil((compressed_size + 4) / tile_size)` (the +4 accounts for the length header)
    - [ ] Update `tiles_dict` entries:
      - Single-slot tiles: `"z/x/y": slot_index` (unchanged)
      - Multi-slot tiles: `"z/x/y": [slot_index, slot_index+1, ..., slot_index+N-1]`
    - [ ] Track `slot_index` separately from logical tile `index` — slot_index increments by chunk count per tile
    - [ ] Add `num_pir_slots` to the output `tile_mapping` dict (= final slot_index)
    - [ ] Update log output to report both logical tile count and PIR slot count
    - [ ] Apply same splitting logic to `generate_synthetic_tiles()` for consistency (though synthetic tiles are small, keep the code path unified)
  - Validation:
    - [ ] After writing tiles.bin, verify: `file_size == num_pir_slots * tile_size`
    - [ ] Log: `"N tiles (M oversized, split into K total PIR slots)"`
  - Run after changes: `python3 prepare_tiles.py --input data/north-america.mbtiles --output pir-map-demo/tiles --max-zoom 11 --max-tiles 100000 --tile-size 20480`

#### Group B: Frontend + Server (execute in parallel after Group A)

- [x] **Task #2**: Update frontend `app.js` to handle multi-slot tiles
  - File: `pir-map-demo/frontend/app.js`
  - Changes to `fetchTileViaPIR(z, x, y, abortSignal)`:
    - [ ] Change lookup: `const pirIndex = tileMapping.get(key)` — value is now either a number or an array
    - [ ] Normalize to array: `const slots = Array.isArray(pirIndex) ? pirIndex : [pirIndex]`
    - [ ] Fire parallel queries for all slots:
      ```javascript
      const responses = await Promise.all(slots.map(async (slotIdx) => {
          const queryBytes = client.create_query(slotIdx);
          const resp = await fetch('/api/query', {
              method: 'POST',
              body: queryBytes,
              headers: { 'Content-Type': 'application/octet-stream' },
              signal: abortSignal,
          });
          if (!resp.ok) throw new Error(`Query failed for slot ${slotIdx}`);
          const encrypted = new Uint8Array(await resp.arrayBuffer());
          return client.decrypt_response(encrypted, slotIdx);
      }));
      ```
    - [ ] Concatenate decrypted slot data into single buffer (raw, before extracting):
      ```javascript
      const totalLen = responses.reduce((s, r) => s + r.length, 0);
      const combined = new Uint8Array(totalLen);
      let offset = 0;
      for (const part of responses) {
          combined.set(new Uint8Array(part), offset);
          offset += part.length;
      }
      ```
    - [ ] Apply `extractTileData(combined)` to the concatenated buffer — the 4-byte length header in the first chunk tells us the exact gzip data length across all chunks
    - [ ] Decompress and return as before
  - Changes to `initialize()`:
    - [ ] When loading tile mapping, preserve the original values (number or array) — current `new Map(Object.entries(mappingData.tiles))` already does this correctly since JSON arrays become JS arrays
  - Console logging:
    - [ ] Log multi-slot fetches: `console.log(\`PIR fetch: ${key} → ${slots.length} slots [${slots.join(',')}]\`)`

- [x] **Task #3**: Update proxy and launch script for slot-based PIR dimensions
  - File: `pir-map-demo/proxy/server.py`
  - Changes to `params()` endpoint:
    - [ ] Count total PIR slots from mapping (not just tile entries):
      ```python
      num_pir_slots = 0
      for value in mapping.get("tiles", {}).values():
          if isinstance(value, list):
              num_pir_slots += len(value)
          else:
              num_pir_slots += 1
      ```
    - [ ] Pass `num_pir_slots` to `compute_pir_params()` instead of `len(mapping["tiles"])`
    - [ ] Also read `num_pir_slots` directly from mapping if present (prefer explicit field over recomputing):
      ```python
      num_pir_slots = mapping.get("num_pir_slots") or count_slots(mapping["tiles"])
      ```
  - File: `pir-map-demo/scripts/run_demo.sh`
  - Changes:
    - [ ] Read `NUM_TILES` as `num_pir_slots` from tile_mapping.json (falling back to `num_tiles` for backwards compat):
      ```bash
      NUM_TILES=$(python3 -c "
      import json
      m = json.load(open('$TILES_DIR/tile_mapping.json'))
      print(m.get('num_pir_slots', m['num_tiles']))
      ")
      ```
    - [ ] This value is passed to GPU server as `--num-tiles`, which must match the actual slot count in tiles.bin

---

## Implementation Workflow

This plan file serves as the authoritative checklist for implementation. When implementing:

### Required Process
1. **Load Plan**: Read this entire plan file before starting
2. **Sync Tasks**: Create TodoWrite tasks matching the checkboxes above
3. **Execute & Update**: For each task:
   - Mark TodoWrite as `in_progress` when starting
   - Update checkbox `[ ]` to `[x]` when completing
   - Mark TodoWrite as `completed` when done
4. **Maintain Sync**: Keep this file and TodoWrite synchronized throughout

### Critical Rules
- This plan file is the source of truth for progress
- Update checkboxes in real-time as work progresses
- Never lose synchronization between plan file and TodoWrite
- Mark tasks complete only when fully implemented (no placeholders)
- Tasks should be run in parallel where dependencies allow

### Execution Order
1. **Task #1** first (generates the new tiles.bin and tile_mapping.json)
2. **Tasks #2 and #3** in parallel (both consume the new format, independent of each other)
3. Restart demo and verify tiles render at all zoom levels

### Progress Tracking
The checkboxes above represent the authoritative status of each task. Keep them updated as you work.
