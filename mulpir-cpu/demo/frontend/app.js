import init, { PIRClient } from './pkg/mulpir_cpu_wasm.js';
import { LRUTileCache } from '/shared/tile-cache.js';
import { TileBatchDispatcher } from '/shared/tile-batch.js';
import { decodeSlotToPBF, decodeMultiSlotToPBF } from '/shared/tile-decoder.js';
import { initMap } from '/shared/map-setup.js';

// ─── State ────────────────────────────────────────────────────────────
let client = null;
let sessionUuid = null;
let tileMapping = null;
let queryCount = 0;
let totalLatencyMs = 0;
let lastQueryMs = 0;
const tileCache = new LRUTileCache(500 * 1024 * 1024);

// ─── MulPIR CPU backend ───────────────────────────────────────────────
const mulpirCpuBackend = {
    processBatch: async (tiles, abortSignal) => {
        // Build flat list of (tileIdx, pirIdx) pairs — one query per slot
        const queryList = [];
        for (let ti = 0; ti < tiles.length; ti++) {
            for (const pirIdx of tiles[ti].slots) {
                queryList.push({ tileIdx: ti, pirIdx });
            }
        }

        console.log(`Dispatcher flush: ${tiles.length} tile(s), ${queryList.length} query(ies)`);

        // Generate all query ciphertexts, yielding to keep UI responsive
        const queryParts = [];
        for (let i = 0; i < queryList.length; i++) {
            queryParts.push(new Uint8Array(client.create_query(queryList[i].pirIdx)));
            if ((i + 1) % 5 === 0) await new Promise(r => setTimeout(r, 0));
        }

        // Pack batch payload: [UUID:36B][num:4B LE][q1_len:4B LE][q1]...[qN_len:4B LE][qN]
        const uuidBytes = new TextEncoder().encode(sessionUuid);
        const totalQueryBytes = queryParts.reduce((s, q) => s + 4 + q.length, 0);
        const payload = new Uint8Array(36 + 4 + totalQueryBytes);
        const view = new DataView(payload.buffer);
        payload.set(uuidBytes, 0);
        view.setUint32(36, queryParts.length, true);
        let off = 40;
        for (const q of queryParts) {
            view.setUint32(off, q.length, true);
            payload.set(q, off + 4);
            off += 4 + q.length;
        }

        // POST /api/batch-query
        const resp = await fetch('/api/batch-query', {
            method: 'POST',
            body: payload,
            headers: { 'Content-Type': 'application/octet-stream' },
            signal: abortSignal,
        });
        if (!resp.ok) throw new Error(`/api/batch-query failed: ${resp.status}`);

        // Parse response: [num:4B LE][r1_len:4B LE][r1]...[rN_len:4B LE][rN]
        const respBuf = new Uint8Array(await resp.arrayBuffer());
        const rv = new DataView(respBuf.buffer);
        const numResp = rv.getUint32(0, true);
        const encryptedResponses = [];
        let roff = 4;
        for (let i = 0; i < numResp; i++) {
            const sz = rv.getUint32(roff, true);
            encryptedResponses.push(respBuf.subarray(roff + 4, roff + 4 + sz));
            roff += 4 + sz;
        }

        // Decrypt each response and group by tile
        const slotParts = tiles.map(() => []);
        for (let i = 0; i < queryList.length; i++) {
            const raw = new Uint8Array(
                client.decrypt_response(encryptedResponses[i], queryList[i].pirIdx)
            );
            slotParts[queryList[i].tileIdx].push(raw);
        }

        // Decode slots to PBF and build result map
        const results = new Map();
        for (let ti = 0; ti < tiles.length; ti++) {
            const tile = tiles[ti];
            let result;
            try {
                const decoded = slotParts[ti];
                result = decoded.length > 1
                    ? decodeMultiSlotToPBF(decoded)
                    : decodeSlotToPBF(decoded[0]);
            } catch { result = new ArrayBuffer(0); }
            results.set(tile.key, result);
        }
        return results;
    }
};

const dispatcher = new TileBatchDispatcher(mulpirCpuBackend, 50);

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
        const pirParams = await paramsResp.json();
        console.log('PIR params:', pirParams);

        // 3. Create MulPIR CPU client
        setStatus('Initializing MulPIR CPU client...');
        setProgress(15);
        client = new PIRClient(pirParams.num_tiles, pirParams.tile_size);

        // 4. Generate evaluation key (Galois keys)
        setStatus('Generating evaluation key (~5-30s on CPU)...');
        setProgress(20);
        await new Promise(r => setTimeout(r, 0));
        const galoisBytes = new Uint8Array(client.generate_galois_key());
        console.log(`Galois key: ${(galoisBytes.length / 1024).toFixed(1)} KB`);

        // 5. Generate relinearization key
        setStatus('Generating relinearization key...');
        setProgress(40);
        await new Promise(r => setTimeout(r, 0));
        const relinBytes = new Uint8Array(client.generate_relin_key());
        console.log(`Relin key: ${(relinBytes.length / 1024).toFixed(1)} KB`);

        // 6. Upload keys to server: [galois_len:4B LE][galoisBytes][relinBytes]
        setStatus(`Uploading keys (${((galoisBytes.length + relinBytes.length) / 1024).toFixed(1)} KB)...`);
        setProgress(50);
        const setupView = new DataView(new ArrayBuffer(4));
        setupView.setUint32(0, galoisBytes.length, true);
        const setupPayload = new Uint8Array(4 + galoisBytes.length + relinBytes.length);
        setupPayload.set(new Uint8Array(setupView.buffer), 0);
        setupPayload.set(galoisBytes, 4);
        setupPayload.set(relinBytes, 4 + galoisBytes.length);

        const setupResp = await fetch('/api/setup', {
            method: 'POST',
            body: setupPayload,
            headers: { 'Content-Type': 'application/octet-stream' },
        });
        if (!setupResp.ok) throw new Error(`Failed to upload keys: ${setupResp.status}`);
        sessionUuid = (await setupResp.text()).trim();
        console.log(`Session UUID: ${sessionUuid}`);

        // 7. Fetch tile mapping
        setStatus('Loading tile mapping...');
        setProgress(80);
        const mappingResp = await fetch('/api/tile-mapping');
        if (!mappingResp.ok) throw new Error('Failed to fetch tile mapping');
        const mappingData = await mappingResp.json();
        tileMapping = new Map(Object.entries(mappingData.tiles));
        console.log(`Tile mapping: ${tileMapping.size} tiles, z${mappingData.min_zoom}-${mappingData.max_zoom}`);

        // 8. Init map
        setStatus('Starting map...');
        setProgress(95);
        initMap(mappingData, fetchTileViaPIR);

        // 9. Show UI, hide loading
        setProgress(100);
        setTimeout(() => {
            document.getElementById('loading-screen').style.display = 'none';
            document.getElementById('pir-badge').style.display = 'flex';
            document.getElementById('cpu-metrics').style.display = 'block';
        }, 300);

        // 10. Start metrics polling
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
        dispatcher.enqueue(z, nx, ny, slots, null)
            .then(data => { if (data.byteLength > 0) tileCache.set(key, data); })
            .catch(() => {});
    }
}

// ─── PIR tile fetching ────────────────────────────────────────────────
async function fetchTileViaPIR(z, x, y, abortSignal) {
    const key = `${z}/${x}/${y}`;

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
