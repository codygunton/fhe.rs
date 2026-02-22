import init, { PIRClient } from './pkg/fhe_wasm.js';
import { LRUTileCache } from '/shared/tile-cache.js';
import { TileBatchDispatcher } from '/shared/tile-batch.js';
import { decodeMultiSlotToPBF } from '/shared/tile-decoder.js';
import { initMap } from '/shared/map-setup.js';

// ─── State ────────────────────────────────────────────────────────────
let client = null;
let tileMapping = null;   // Map<string, number|number[]>  "z/x/y" → pirIndex or [indices]
let pirParams = null;
let queryCount = 0;
let totalLatencyMs = 0;
let lastQueryMs = 0;
const tileCache = new LRUTileCache(500 * 1024 * 1024); // 500 MB

// ─── MulPIR backend ───────────────────────────────────────────────────
const mulpirBackend = {
    processBatch: async (tiles, abortSignal) => {
        // tiles: [{key, z, x, y, slots}]
        // Build flat slot list across all tiles
        const allSlots = [];
        for (const tile of tiles) {
            tile.startIdx = allSlots.length;
            for (const s of tile.slots) allSlots.push(s);
        }

        console.log(`Dispatcher flush: ${tiles.length} tile(s), ${allSlots.length} slot(s)`);

        // Encrypt queries, yielding every 5 to keep the page responsive
        const queryParts = [];
        for (let i = 0; i < allSlots.length; i++) {
            queryParts.push(client.create_query(allSlots[i]));
            if ((i + 1) % 5 === 0) await new Promise(r => setTimeout(r, 0));
        }

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
        const respBuf = new Uint8Array(await resp.arrayBuffer());

        // Parse response: [u32 num_responses][u32 size][bytes]...
        const rv = new DataView(respBuf.buffer);
        const numResponses = rv.getUint32(0, true);
        if (numResponses !== allSlots.length) {
            const err = new Error(`Batch response count mismatch: got ${numResponses}, expected ${allSlots.length}`);
            console.error(err.message);
            throw err;
        }

        // Read all raw ciphertext blobs
        const encryptedResponses = [];
        let roff = 4;
        for (let i = 0; i < numResponses; i++) {
            const sz = rv.getUint32(roff, true); roff += 4;
            encryptedResponses.push(respBuf.subarray(roff, roff + sz)); roff += sz;
        }

        // Decrypt per tile, build result map
        const results = new Map();
        for (const tile of tiles) {
            let result;
            try {
                const parts = [];
                for (let i = 0; i < tile.slots.length; i++) {
                    parts.push(client.decrypt_response(encryptedResponses[tile.startIdx + i], tile.slots[i]));
                }
                result = decodeMultiSlotToPBF(parts.map(p => new Uint8Array(p)));
            } catch { result = new ArrayBuffer(0); }
            results.set(tile.key, result);
        }
        return results;
    }
};

const dispatcher = new TileBatchDispatcher(mulpirBackend, 50); // 50ms coalesce window

// ─── UI helpers ───────────────────────────────────────────────────────
function setStatus(msg) {
    document.getElementById('loading-status').textContent = msg;
}

function setProgress(pct) {
    document.getElementById('loading-bar').style.width = pct + '%';
}

// ─── Initialization ───────────────────────────────────────────────────
async function initialize() {
    try {
        // 1. Load WASM module
        setStatus('Loading WASM module...');
        setProgress(5);
        await init();

        // 2. Fetch PIR params from server
        setStatus('Fetching PIR parameters...');
        setProgress(10);
        const paramsResp = await fetch('/api/params');
        if (!paramsResp.ok) throw new Error('Failed to fetch /api/params');
        pirParams = await paramsResp.json();
        console.log('PIR params:', pirParams);

        // 3. Create PIR client — use num_pir_slots (total database slots)
        setStatus('Generating encryption keys...');
        setProgress(15);
        client = new PIRClient(pirParams.num_tiles, pirParams.tile_size);
        console.log('Client params:', client.get_params_json());

        // 4. Generate and upload Galois key (this is slow + large)
        setStatus('Generating Galois key (this may take a while)...');
        setProgress(20);
        const galoisKey = client.generate_galois_key();
        console.log(`Galois key: ${(galoisKey.length / 1024 / 1024).toFixed(1)} MB`);

        setStatus(`Uploading Galois key (${(galoisKey.length / 1024 / 1024).toFixed(1)} MB)...`);
        setProgress(40);
        const gkResp = await fetch('/api/setup-galois-key', {
            method: 'POST',
            body: galoisKey,
            headers: { 'Content-Type': 'application/octet-stream' },
        });
        if (!gkResp.ok) throw new Error('Failed to upload Galois key');
        const gkResult = await gkResp.json();
        if (gkResult.status !== 'ok') throw new Error(`Galois key error: ${gkResult.message}`);

        // 5. Generate and upload relinearization key
        setStatus('Generating relinearization key...');
        setProgress(60);
        const relinKey = client.generate_relin_key();
        console.log(`Relin key: ${(relinKey.length / 1024 / 1024).toFixed(1)} MB`);

        setStatus('Uploading relinearization key...');
        setProgress(70);
        const rkResp = await fetch('/api/setup-relin-key', {
            method: 'POST',
            body: relinKey,
            headers: { 'Content-Type': 'application/octet-stream' },
        });
        if (!rkResp.ok) throw new Error('Failed to upload relin key');
        const rkResult = await rkResp.json();
        if (rkResult.status !== 'ok') throw new Error(`Relin key error: ${rkResult.message}`);

        // 6. Fetch tile mapping
        setStatus('Loading tile mapping...');
        setProgress(80);
        const mappingResp = await fetch('/api/tile-mapping');
        if (!mappingResp.ok) throw new Error('Failed to fetch tile mapping');
        const mappingData = await mappingResp.json();
        // Values are either a number (single slot) or an array of numbers (multi-slot)
        tileMapping = new Map(Object.entries(mappingData.tiles));
        console.log(`Tile mapping: ${tileMapping.size} tiles, z${mappingData.min_zoom}-${mappingData.max_zoom}`);

        // 7. Init map
        setStatus('Starting map...');
        setProgress(95);
        initMap(mappingData, fetchTileViaPIR);

        // 8. Show UI, hide loading
        setProgress(100);
        setTimeout(() => {
            document.getElementById('loading-screen').style.display = 'none';
            document.getElementById('pir-badge').style.display = 'flex';
            document.getElementById('gpu-metrics').style.display = 'block';
        }, 300);

        // 9. Start metrics polling
        startMetricsPolling();

    } catch (err) {
        console.error('Init failed:', err);
        setStatus(`Error: ${err.message}`);
        document.querySelector('.loading-spinner').style.display = 'none';
    }
}

// ─── Speculative prefetch for spatial neighbours ──────────────────────
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

// ─── PIR tile fetching ────────────────────────────────────────────────
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

function updatePirStats() {
    const avg = queryCount > 0 ? (totalLatencyMs / queryCount).toFixed(0) : '—';
    document.getElementById('pir-stats').textContent =
        `${queryCount} queries | avg ${avg}ms`;
    document.getElementById('query-time').textContent =
        `${lastQueryMs.toFixed(0)}ms`;
}

// ─── GPU metrics polling ──────────────────────────────────────────────
function startMetricsPolling() {
    async function poll() {
        try {
            const resp = await fetch('/api/metrics');
            if (!resp.ok) return;
            const m = await resp.json();
            if (m.error) return;
            document.getElementById('gpu-util').textContent = `${m.gpu_utilization}%`;
            document.getElementById('gpu-mem').textContent =
                `${(m.memory_used_mb / 1024).toFixed(1)} / ${(m.memory_total_mb / 1024).toFixed(1)} GB`;
            document.getElementById('gpu-temp').textContent = `${m.temperature_c}°C`;
        } catch {
            // Metrics unavailable — not critical
        }
    }
    poll();
    setInterval(poll, 2000);
}

// ─── Start ────────────────────────────────────────────────────────────
initialize();
