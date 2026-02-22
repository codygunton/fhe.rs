# PIR Map Performance Optimizations — Implementation Plan

## Executive Summary

### Problem Statement
The PIR map demo fetches tiles sequentially: MapLibre fires protocol handler invocations per-tile, but WASM `create_query()` is synchronous and blocks the JS event loop before each `fetch()` goes out. Even though Flask runs `threaded=True`, requests arrive at the GPU server staggered — one per tile, separated by the WASM encryption time for that tile's slots. A 12-slot tile means 12 sequential WASM calls before the first byte hits the wire. With 8–10 tiles in a viewport at zoom 14, this means hundreds of milliseconds of stall before the GPU ever sees a query.

Meanwhile, once tiles are fetched, there is no application-level cache — MapLibre's internal 512-tile LRU handles re-fetches, but there is no speculative population of that cache for nearby tiles the user is likely to pan to.

### Proposed Solution

Four coordinated changes:

1. **Viewport request coalescing** — a `TileBatchDispatcher` that collects all tile requests arriving within a 50ms window, then fires them as a single `/api/batch-query` call. One HTTP round-trip per viewport pan instead of N.

2. **Raise GPU server batch limit** — the current hard cap of 64 queries will be exceeded at zoom 14 (8 tiles × 12 slots = 96 queries). Raise to 512.

3. **Application-level LRU tile cache** — store decrypted PBF tiles in a `LRUTileCache` (500MB limit). Protocol handler checks cache before issuing any PIR query. Prefetched tiles populate this cache.

4. **Speculative prefetch** — after a tile loads successfully, silently enqueue its 8 neighbours into the dispatcher at background priority. When the user pans, neighbours are already cached.

### Data Flow

```
MapLibre requests tile (z,x,y)
        │
        ▼
  LRUTileCache.has(key)?
  ┌─────┴─────┐
 yes          no
  │            │
  │     TileBatchDispatcher.enqueue(z,x,y,slots)
  │            │
  │     [50ms coalesce window]
  │            │ (multiple tiles arrive in same window)
  │            ▼
  │     flatten all slots from all pending tiles
  │     create_query() for each slot  (WASM, sequential)
  │     pack single /api/batch-query payload
  │            │
  │            ▼
  │       Flask proxy (threaded)
  │            │
  │            ▼
  │       GPU server: deserialize_batch_query
  │            │
  │            ▼
  │       process N queries sequentially on GPU
  │            │
  │            ▼
  │       serialize_batch_response
  │            │
  │       ◄────┘
  │       route responses back to each tile's Promise
  │       decrypt → gzip inflate → PBF
  │       LRUTileCache.set(key, pbf)
  │       prefetchNeighbors(z,x,y) → dispatcher (background)
  │
  └──► return {data: pbf}  to MapLibre
```

### Expected Outcomes
- All tiles in the viewport are queried in one HTTP request instead of N
- WASM `create_query()` calls are batched together before any fetch starts, eliminating staggered arrival at the server
- GPU processes queries back-to-back with no idle time between tiles
- Tiles seen before are served from cache instantly (zero PIR cost)
- Neighbours are fetched speculatively; panning into them renders immediately
- Batch limit raised to 512 supports up to ~25 tiles × 20 slots without splitting

---

## Goals & Objectives

### Primary Goals
- Eliminate serial tile request arrival at the GPU server — all viewport tiles should arrive in one batch
- Reduce visible tile load time on first pan by pre-loading neighbours

### Secondary Objectives
- Cache hits for tiles the user revisits (no repeated PIR queries)
- Minimal disruption to existing code paths; abort signal semantics preserved

---

## Solution Overview

### Key Components

1. **`TileBatchDispatcher` (app.js)**: Collects pending tile requests, coalesces within a 50ms window, fires one `/api/batch-query`, routes responses back to per-tile Promises. Handles abort by removing cancelled tiles before flush.

2. **`LRUTileCache` (app.js)**: Size-bounded (500MB) LRU Map keyed by `z/x/y`. Returns cached PBF on hit; tiles are stored on successful PIR fetch and prefetch.

3. **`prefetchNeighbors` (app.js)**: After a successful tile load, enqueues the 8 spatial neighbours (that exist in `tileMapping`) into the dispatcher with no abort signal. Failures silently ignored.

4. **Batch limit increase (wire_format.cu)**: Change `num_queries > 64` to `num_queries > 512`.

---

## Implementation Tasks

### CRITICAL IMPLEMENTATION RULES
1. **NO PLACEHOLDER CODE** — every implementation is production-ready.
2. **PRESERVE ABORT SEMANTICS** — tiles cancelled by MapLibre before flush are removed from the queue; after flush, their resolve is silently skipped.
3. **SINGLE SLOT TILES** — tiles with one slot still go through the dispatcher (no special-casing `fetchSlot` vs `fetchSlotsViaBatch` — both paths are replaced by the dispatcher).

### Visual Dependency Tree

```
pir-map-demo/frontend/
└── app.js
    ├── (Task #0) LRUTileCache class — standalone, no deps
    ├── (Task #0) TileBatchDispatcher class — depends on LRUTileCache for storing results
    ├── (Task #1) prefetchNeighbors() — depends on dispatcher and tileCache existing
    ├── (Task #1) protocol handler rewrite — depends on dispatcher and tileCache
    └── (Task #1) remove fetchSlot / fetchSlotsViaBatch — replaced by dispatcher

mulpir-gpu-server/src/serialization/
└── wire_format.cu
    └── (Task #0) raise batch limit 64 → 512 — independent
```

### Execution Plan

#### Group A: Foundation (run in parallel)

- [x] **Task #0a**: Raise GPU server batch limit
  - File: `mulpir-gpu-server/src/serialization/wire_format.cu`
  - Change line `if (num_queries == 0 || num_queries > 64)` to `if (num_queries == 0 || num_queries > 512)`
  - Rationale: zoom 14 viewport = ~8 tiles × ~15 slots = ~120 queries; current cap of 64 is too low
  - After change: rebuild with `cmake --build mulpir-gpu-server/build`
  - No other files need changing — the limit is only enforced here

- [x] **Task #0b**: Add `LRUTileCache` class to app.js
  - File: `pir-map-demo/frontend/app.js`
  - Insert before the `// ─── State ───` section (before line 1)
  - Full implementation:
    ```javascript
    class LRUTileCache {
        constructor(maxBytes = 500 * 1024 * 1024) {
            this._cache = new Map();   // key → {data: ArrayBuffer, size: number}
            this._totalSize = 0;
            this._maxSize = maxBytes;
        }
        has(key) { return this._cache.has(key); }
        get(key) {
            if (!this._cache.has(key)) return null;
            // Promote to MRU
            const entry = this._cache.get(key);
            this._cache.delete(key);
            this._cache.set(key, entry);
            return entry.data;
        }
        set(key, data) {
            const size = data.byteLength;
            if (size === 0) return;
            // Evict LRU until there's room
            const iter = this._cache.entries();
            while (this._totalSize + size > this._maxSize && this._cache.size > 0) {
                const { value: [oldKey, oldEntry] } = iter.next();
                this._cache.delete(oldKey);
                this._totalSize -= oldEntry.size;
            }
            if (this._cache.has(key)) {
                this._totalSize -= this._cache.get(key).size;
                this._cache.delete(key);
            }
            this._cache.set(key, { data, size });
            this._totalSize += size;
        }
        get size() { return this._cache.size; }
        get bytes() { return this._totalSize; }
    }
    ```
  - Add to module-level state (after `let client = null;`):
    ```javascript
    const tileCache = new LRUTileCache(500 * 1024 * 1024); // 500 MB
    ```

- [x] **Task #0c**: Add `TileBatchDispatcher` class to app.js
  - File: `pir-map-demo/frontend/app.js`
  - Insert after `LRUTileCache` class
  - Full implementation:
    ```javascript
    class TileBatchDispatcher {
        constructor(coalesceMs = 50) {
            this._pending = new Map();  // key → {slots, resolve, reject}
            this._timer = null;
            this._coalesceMs = coalesceMs;
        }

        // Enqueue a tile request. Returns a Promise<ArrayBuffer> (PBF or empty).
        enqueue(z, x, y, slots, abortSignal) {
            const key = `${z}/${x}/${y}`;
            return new Promise((resolve, reject) => {
                // If a request for this key already exists, the existing Promise wins.
                if (this._pending.has(key)) { resolve(new ArrayBuffer(0)); return; }
                this._pending.set(key, { slots, resolve, reject });
                // Remove from queue if MapLibre cancels before flush
                if (abortSignal) {
                    abortSignal.addEventListener('abort', () => {
                        const entry = this._pending.get(key);
                        if (entry) {
                            this._pending.delete(key);
                            resolve(new ArrayBuffer(0));
                        }
                    }, { once: true });
                }
                if (!this._timer) {
                    this._timer = setTimeout(() => this._flush(), this._coalesceMs);
                }
            });
        }

        async _flush() {
            this._timer = null;
            if (this._pending.size === 0) return;

            const batch = [...this._pending.entries()];
            this._pending.clear();

            // Build flat slot list + per-tile bookkeeping
            const tiles = [];
            const allSlots = [];
            for (const [key, { slots, resolve, reject }] of batch) {
                tiles.push({ key, startIdx: allSlots.length, count: slots.length, resolve, reject, slots });
                for (const s of slots) allSlots.push(s);
            }

            let respBuf;
            try {
                // Create all queries (WASM sync — all in one JS tick before any await)
                const queryParts = allSlots.map(idx => client.create_query(idx));

                // Pack batch payload: [u32 num_queries][u32 size][bytes]...
                let totalSize = 4;
                for (const q of queryParts) totalSize += 4 + q.length;
                const payload = new Uint8Array(totalSize);
                const view = new DataView(payload.buffer);
                view.setUint32(0, allSlots.length, true);
                let off = 4;
                for (const q of queryParts) {
                    view.setUint32(off, q.length, true); off += 4;
                    payload.set(q, off); off += q.length;
                }

                const resp = await fetch('/api/batch-query', {
                    method: 'POST',
                    body: payload,
                    headers: { 'Content-Type': 'application/octet-stream' },
                });
                if (!resp.ok) throw new Error(`Batch query failed: ${resp.status}`);
                respBuf = new Uint8Array(await resp.arrayBuffer());
            } catch (err) {
                for (const { reject } of tiles) reject(err);
                return;
            }

            // Parse response: [u32 num_responses][u32 size][bytes]...
            const rv = new DataView(respBuf.buffer);
            const numResponses = rv.getUint32(0, true);
            if (numResponses !== allSlots.length) {
                const err = new Error(`Batch response count mismatch: got ${numResponses}, expected ${allSlots.length}`);
                for (const { reject } of tiles) reject(err);
                return;
            }

            // Read all raw ciphertext blobs
            const encryptedResponses = [];
            let roff = 4;
            for (let i = 0; i < numResponses; i++) {
                const sz = rv.getUint32(roff, true); roff += 4;
                encryptedResponses.push(respBuf.subarray(roff, roff + sz)); roff += sz;
            }

            // Distribute to each tile
            for (const { startIdx, count, slots, resolve } of tiles) {
                try {
                    const parts = [];
                    for (let i = 0; i < count; i++) {
                        parts.push(client.decrypt_response(encryptedResponses[startIdx + i], slots[i]));
                    }
                    // Concatenate decrypted parts
                    const totalLen = parts.reduce((s, p) => s + p.length, 0);
                    const combined = new Uint8Array(totalLen);
                    let coff = 0;
                    for (const p of parts) { combined.set(new Uint8Array(p), coff); coff += p.length; }
                    // Extract length-prefixed gzip, inflate to PBF
                    const trimmed = extractTileData(combined);
                    if (trimmed.length === 0) { resolve(new ArrayBuffer(0)); continue; }
                    let pbf;
                    try { pbf = pako.inflate(trimmed); }
                    catch { resolve(new ArrayBuffer(0)); continue; }
                    resolve(pbf.buffer);
                } catch (err) {
                    resolve(new ArrayBuffer(0));
                }
            }
        }
    }
    ```
  - Add to module-level state (after `const tileCache = ...`):
    ```javascript
    const dispatcher = new TileBatchDispatcher(50);
    ```

#### Group B: Integration (after Group A)

- [x] **Task #1a**: Add `prefetchNeighbors` function
  - File: `pir-map-demo/frontend/app.js`
  - Add after the existing `fetchSlotsViaBatch` function (before `// ─── PIR tile fetching`)
  - Full implementation:
    ```javascript
    const NEIGHBOR_OFFSETS = [
        [-1,-1],[0,-1],[1,-1],
        [-1, 0],       [1, 0],
        [-1, 1],[0, 1],[1, 1],
    ];

    function prefetchNeighbors(z, x, y) {
        for (const [dx, dy] of NEIGHBOR_OFFSETS) {
            const nx = x + dx, ny = y + dy;
            const key = `${z}/${nx}/${ny}`;
            if (!tileMapping || !tileMapping.has(key)) continue;
            if (tileCache.has(key)) continue;
            const pirIndex = tileMapping.get(key);
            const slots = Array.isArray(pirIndex) ? pirIndex : [pirIndex];
            // Fire-and-forget: no abort signal, errors silently discarded
            dispatcher.enqueue(z, nx, ny, slots, null)
                .then(data => { if (data.byteLength > 0) tileCache.set(key, data); })
                .catch(() => {});
        }
    }
    ```

- [x] **Task #1b**: Rewrite `fetchTileViaPIR` and protocol handler
  - File: `pir-map-demo/frontend/app.js`
  - **Delete** `fetchSlot()` (lines 119-131) — no longer needed
  - **Delete** `fetchSlotsViaBatch()` (lines 133-183) — replaced by dispatcher
  - **Replace** `fetchTileViaPIR()` (lines 185-244) with:
    ```javascript
    async function fetchTileViaPIR(z, x, y, abortSignal) {
        const key = `${z}/${x}/${y}`;

        // Fast path: already cached
        const cached = tileCache.get(key);
        if (cached) {
            console.log(`PIR ${key}: cache hit`);
            return cached;
        }

        const pirIndex = tileMapping.get(key);
        if (pirIndex === undefined) return new ArrayBuffer(0);

        const slots = Array.isArray(pirIndex) ? pirIndex : [pirIndex];
        console.log(`PIR fetch: ${key} → ${slots.length} slot(s) [${slots.join(',')}]`);

        const t0 = performance.now();
        try {
            const pbf = await dispatcher.enqueue(z, x, y, slots, abortSignal);
            if (pbf.byteLength === 0) return pbf;

            // Store in cache and speculatively prefetch neighbours
            tileCache.set(key, pbf);
            prefetchNeighbors(z, x, y);

            const elapsed = performance.now() - t0;
            queryCount++;
            totalLatencyMs += elapsed;
            lastQueryMs = elapsed;
            updatePirStats();
            console.log(`PIR ${key}: OK ${slots.length} slot(s) in ${elapsed.toFixed(0)}ms`);
            return pbf;
        } catch (e) {
            if (e?.name !== 'AbortError') console.error(`PIR ${key}: fetch failed:`, e?.message || e);
            return new ArrayBuffer(0);
        }
    }
    ```
  - Note: logging of gzip/PBF sizes is moved into dispatcher's per-tile path; if desired add logging there too

---

## Implementation Workflow

This plan file is the authoritative checklist. When implementing:

### Required Process
1. **Load Plan**: Read this entire file before starting
2. **Sync Tasks**: Create TodoWrite tasks matching the checkboxes
3. **Execute & Update**: For each task:
   - Mark TodoWrite as `in_progress` when starting
   - Update checkbox `[ ]` to `[x]` when completing
4. **Execution Order**: Group A tasks run in parallel; Group B tasks start after Group A completes

### Build Steps After Implementation
1. After Task #0a: `cmake --build mulpir-gpu-server/build` (recompile GPU server)
2. After Task #0b/0c/1a/1b: no build needed (plain JS, hot-reload or restart demo)
3. Full test: `./run_demo.sh --proxy-port 7000` → open browser, pan map, verify tiles load in bursts rather than one at a time

### Verification
- Open browser DevTools Network tab — should see one `/api/batch-query` fire per pan instead of many `/api/query` calls
- Console should show `PIR fetch: z/x/y → N slot(s)` for first load, then `PIR z/x/y: cache hit` when panning back
- Neighbours should preload without any user-visible trigger
