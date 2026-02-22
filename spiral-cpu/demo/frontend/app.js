import init, { SpiralClient } from './pkg/spiral_wasm.js';
import { LRUTileCache } from '/shared/tile-cache.js';
import { TileBatchDispatcher } from '/shared/tile-batch.js';
import { decodeSlotToPBF, decodeMultiSlotToPBF } from '/shared/tile-decoder.js';
import { initMap } from '/shared/map-setup.js';

// ─── State ────────────────────────────────────────────────────────────
let client = null;
let sessionUuid = null;   // UUID string returned by /api/setup
let tileMapping = null;   // Map<string, number|number[]>  "z/x/y" → pirIndex or [indices]
let pirParams = null;
let queryCount = 0;
let totalLatencyMs = 0;
let lastQueryMs = 0;
const tileCache = new LRUTileCache(500 * 1024 * 1024); // 500 MB

// ─── Spiral PIR backend ───────────────────────────────────────────────
const spiralBackend = {
    processBatch: async (tiles, abortSignal) => {
        // tiles: [{key, z, x, y, slots}]
        // Build flat list of (tileIdx, pirIdx) pairs — one query per slot
        const queryList = [];
        for (let ti = 0; ti < tiles.length; ti++) {
            for (const pirIdx of tiles[ti].slots) {
                queryList.push({ tileIdx: ti, pirIdx });
            }
        }

        console.log(`Dispatcher flush: ${tiles.length} tile(s), ${queryList.length} query(ies)`);

        // Generate all query payloads, yielding every 5 to keep the page responsive
        const uuidBytes = new TextEncoder().encode(sessionUuid);
        const queryPayloads = [];
        for (let i = 0; i < queryList.length; i++) {
            const qBytes = new Uint8Array(client.generate_query(queryList[i].pirIdx));
            const payload = new Uint8Array(36 + qBytes.length);
            payload.set(uuidBytes, 0);
            payload.set(qBytes, 36);
            queryPayloads.push(payload);
            if ((i + 1) % 5 === 0) await new Promise(r => setTimeout(r, 0));
        }

        // Send all queries in parallel
        const responsePromises = queryPayloads.map(payload =>
            fetch('/api/private-read', {
                method: 'POST',
                body: payload,
                headers: { 'Content-Type': 'application/octet-stream' },
                signal: abortSignal,
            }).then(r => {
                if (!r.ok) throw new Error(`/api/private-read failed: ${r.status}`);
                return r.arrayBuffer();
            })
        );
        const rawResponses = await Promise.all(responsePromises);

        // Group by tile, decode, build result map
        const slotParts = tiles.map(() => []);
        for (let i = 0; i < queryList.length; i++) {
            slotParts[queryList[i].tileIdx].push(new Uint8Array(rawResponses[i]));
        }

        const results = new Map();
        for (let ti = 0; ti < tiles.length; ti++) {
            const tile = tiles[ti];
            let result;
            try {
                const decoded = slotParts[ti].map(raw => new Uint8Array(client.decode_response(raw)));
                result = decoded.length > 1
                    ? decodeMultiSlotToPBF(decoded)
                    : decodeSlotToPBF(decoded[0]);
            } catch { result = new ArrayBuffer(0); }
            results.set(tile.key, result);
        }
        return results;
    }
};

const dispatcher = new TileBatchDispatcher(spiralBackend, 50); // 50ms coalesce window

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

        // 3. Create Spiral client from server-provided params JSON
        setStatus('Initializing Spiral PIR client...');
        setProgress(15);
        client = new SpiralClient(pirParams.spiral_params);
        console.log(`Spiral client: ${client.num_items()} items, ${client.query_bytes()} B/query`);

        // 4. Generate public parameters (encryption keys)
        setStatus('Generating encryption keys...');
        setProgress(20);
        const setupBytes = client.generate_keys();
        console.log(`Setup data: ${(setupBytes.length / 1024).toFixed(1)} KB`);

        // 5. Upload public parameters to server
        setStatus(`Uploading keys (${(setupBytes.length / 1024).toFixed(1)} KB)...`);
        setProgress(50);
        const setupResp = await fetch('/api/setup', {
            method: 'POST',
            body: setupBytes,
            headers: { 'Content-Type': 'application/octet-stream' },
        });
        if (!setupResp.ok) throw new Error(`Failed to upload keys: ${setupResp.status}`);
        sessionUuid = (await setupResp.text()).trim();
        console.log(`Session UUID: ${sessionUuid}`);

        // 6. Fetch tile mapping
        setStatus('Loading tile mapping...');
        setProgress(80);
        const mappingResp = await fetch('/api/tile-mapping');
        if (!mappingResp.ok) throw new Error('Failed to fetch tile mapping');
        const mappingData = await mappingResp.json();
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
            document.getElementById('cpu-metrics').style.display = 'block';
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
        // Fire-and-forget: errors silently discarded
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

// ─── CPU metrics polling ──────────────────────────────────────────────
function startMetricsPolling() {
    async function poll() {
        try {
            const resp = await fetch('/api/metrics');
            if (!resp.ok) return;
            const m = await resp.json();
            if (m.error) return;
            document.getElementById('cpu-util').textContent = `${m.cpu_percent}%`;
            document.getElementById('cpu-mem').textContent =
                `${m.memory_used_mb} / ${m.memory_total_mb} MB`;
        } catch {
            // Metrics unavailable — not critical
        }
    }
    poll();
    setInterval(poll, 2000);
}

// ─── Start ────────────────────────────────────────────────────────────
initialize();
