import init, { PIRClient } from './pkg/fhe_wasm.js';

// ─── LRU tile cache ───────────────────────────────────────────────────
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
        // Evict LRU entries until there's room
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

// ─── Batch dispatcher — coalesces tile requests within a time window ──
class TileBatchDispatcher {
    constructor(coalesceMs = 50) {
        this._pending = new Map();  // key → {slots, resolvers: [], reject}
        this._timer = null;
        this._coalesceMs = coalesceMs;
    }

    // Enqueue a tile request. Returns a Promise<ArrayBuffer> (PBF or empty).
    // Multiple calls for the same key within the coalesce window all receive
    // the same result (e.g. a prefetch and a MapLibre request for the same tile).
    enqueue(z, x, y, slots, abortSignal) {
        const key = `${z}/${x}/${y}`;
        return new Promise((resolve, reject) => {
            if (this._pending.has(key)) {
                // Piggyback: both this resolver and the original get the real data
                const entry = this._pending.get(key);
                entry.resolvers.push(resolve);
                if (abortSignal) {
                    abortSignal.addEventListener('abort', () => {
                        const e = this._pending.get(key);
                        if (e) {
                            const idx = e.resolvers.indexOf(resolve);
                            if (idx >= 0) e.resolvers.splice(idx, 1);
                        }
                        resolve(new ArrayBuffer(0));
                    }, { once: true });
                }
                return;
            }
            this._pending.set(key, { slots, resolvers: [resolve], reject });
            // Remove from queue if MapLibre cancels before flush
            if (abortSignal) {
                abortSignal.addEventListener('abort', () => {
                    const entry = this._pending.get(key);
                    if (entry) {
                        const idx = entry.resolvers.indexOf(resolve);
                        if (idx >= 0) entry.resolvers.splice(idx, 1);
                        // Only remove from pending if no resolvers remain
                        if (entry.resolvers.length === 0) {
                            this._pending.delete(key);
                        }
                    }
                    resolve(new ArrayBuffer(0));
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
        for (const [key, { slots, resolvers, reject }] of batch) {
            tiles.push({ key, startIdx: allSlots.length, count: slots.length, resolvers, reject, slots });
            for (const s of slots) allSlots.push(s);
        }

        console.log(`Dispatcher flush: ${tiles.length} tile(s), ${allSlots.length} slot(s)`);

        let respBuf;
        try {
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
            respBuf = new Uint8Array(await resp.arrayBuffer());
        } catch (err) {
            for (const { resolvers, reject } of tiles) {
                reject(err);  // only the first resolver's rejection is observed
                for (const r of resolvers.slice(1)) r(new ArrayBuffer(0));
            }
            return;
        }

        // Parse response: [u32 num_responses][u32 size][bytes]...
        const rv = new DataView(respBuf.buffer);
        const numResponses = rv.getUint32(0, true);
        if (numResponses !== allSlots.length) {
            const err = new Error(`Batch response count mismatch: got ${numResponses}, expected ${allSlots.length}`);
            for (const { resolvers } of tiles) for (const r of resolvers) r(new ArrayBuffer(0));
            console.error(err.message);
            return;
        }

        // Read all raw ciphertext blobs
        const encryptedResponses = [];
        let roff = 4;
        for (let i = 0; i < numResponses; i++) {
            const sz = rv.getUint32(roff, true); roff += 4;
            encryptedResponses.push(respBuf.subarray(roff, roff + sz)); roff += sz;
        }

        // Decrypt and distribute tile by tile, yielding after each tile so
        // MapLibre can render progressive results rather than one big-bang update.
        for (const { startIdx, count, slots, resolvers } of tiles) {
            let result;
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
                if (trimmed.length === 0) { result = new ArrayBuffer(0); }
                else {
                    try { result = pako.inflate(trimmed).buffer; }
                    catch { result = new ArrayBuffer(0); }
                }
            } catch { result = new ArrayBuffer(0); }
            for (const r of resolvers) r(result);
            // Yield: allows fetchTileViaPIR microtask to run and MapLibre to render
            // this tile before we decrypt the next one.
            await new Promise(r => setTimeout(r, 0));
        }
    }
}

// ─── State ────────────────────────────────────────────────────────────
let client = null;
let tileMapping = null;   // Map<string, number|number[]>  "z/x/y" → pirIndex or [indices]
let pirParams = null;
let queryCount = 0;
let totalLatencyMs = 0;
let lastQueryMs = 0;
const tileCache = new LRUTileCache(500 * 1024 * 1024); // 500 MB
const dispatcher = new TileBatchDispatcher(50);         // 50ms coalesce window

// ─── UI helpers ───────────────────────────────────────────────────────
function setStatus(msg) {
    document.getElementById('loading-status').textContent = msg;
}

function setProgress(pct) {
    document.getElementById('loading-bar').style.width = pct + '%';
}

// ─── Extract tile data from a length-prefixed buffer ──────────────────
// Format: [data_len: u32 LE][gzip data][zero padding...]
// For multi-slot tiles the buffer is the concatenation of all decrypted slots.
function extractTileData(data) {
    if (data.length < 4) return new Uint8Array(0);
    const len = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
    if (len <= 0 || len + 4 > data.length) return new Uint8Array(0);
    return data.subarray(4, 4 + len);
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
        initMap(mappingData);

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

// ─── MapLibre setup ───────────────────────────────────────────────────
function parseTileURL(url) {
    // URL format: pir://{z}/{x}/{y} or pir://tiles/{z}/{x}/{y}
    const parts = url.replace('pir://', '').split('/');
    // Handle both pir://z/x/y and pir://tiles/z/x/y
    if (parts[0] === 'tiles') parts.shift();
    return parts.map(Number);
}

function initMap(mappingData) {
    // Register pir:// protocol
    maplibregl.addProtocol('pir', async (params, abortController) => {
        const [z, x, y] = parseTileURL(params.url);
        const data = await fetchTileViaPIR(z, x, y, abortController.signal);
        return { data };
    });

    const center = mappingData.center || [-73.9857, 40.7484];
    const maxZoom = mappingData.max_zoom || 11;

    const map = new maplibregl.Map({
        container: 'map',
        center: center,
        zoom: 14,
        minZoom: 0,
        maxZoom: 16,
        style: {
            version: 8,
            name: 'PIR Vector Tiles',
            sources: {
                pir: {
                    type: 'vector',
                    tiles: ['pir://tiles/{z}/{x}/{y}'],
                    minzoom: 0,
                    maxzoom: maxZoom,
                },
            },
            layers: [
                // Background
                {
                    id: 'background',
                    type: 'background',
                    paint: { 'background-color': '#1a1a2e' },
                },
                // Water
                {
                    id: 'water',
                    type: 'fill',
                    source: 'pir',
                    'source-layer': 'water',
                    paint: {
                        'fill-color': '#1a3a5c',
                        'fill-opacity': 0.8,
                    },
                },
                // Landcover
                {
                    id: 'landcover',
                    type: 'fill',
                    source: 'pir',
                    'source-layer': 'landcover',
                    paint: {
                        'fill-color': '#1e3a1e',
                        'fill-opacity': 0.4,
                    },
                },
                // Landuse
                {
                    id: 'landuse',
                    type: 'fill',
                    source: 'pir',
                    'source-layer': 'landuse',
                    paint: {
                        'fill-color': '#2a2a3e',
                        'fill-opacity': 0.5,
                    },
                },
                // Park
                {
                    id: 'park',
                    type: 'fill',
                    source: 'pir',
                    'source-layer': 'park',
                    paint: {
                        'fill-color': '#1e4a1e',
                        'fill-opacity': 0.3,
                    },
                },
                // Buildings
                {
                    id: 'building',
                    type: 'fill',
                    source: 'pir',
                    'source-layer': 'building',
                    minzoom: 10,
                    paint: {
                        'fill-color': '#3a3a5e',
                        'fill-opacity': 0.6,
                        'fill-outline-color': '#4a4a6e',
                    },
                },
                // Roads — highway
                {
                    id: 'road-highway',
                    type: 'line',
                    source: 'pir',
                    'source-layer': 'transportation',
                    filter: ['==', 'class', 'motorway'],
                    paint: {
                        'line-color': '#f0a050',
                        'line-width': ['interpolate', ['linear'], ['zoom'], 5, 0.5, 10, 3, 14, 6],
                    },
                },
                // Roads — major
                {
                    id: 'road-major',
                    type: 'line',
                    source: 'pir',
                    'source-layer': 'transportation',
                    filter: ['in', 'class', 'trunk', 'primary'],
                    paint: {
                        'line-color': '#c0a060',
                        'line-width': ['interpolate', ['linear'], ['zoom'], 7, 0.3, 10, 1.5, 14, 4],
                    },
                },
                // Roads — secondary
                {
                    id: 'road-secondary',
                    type: 'line',
                    source: 'pir',
                    'source-layer': 'transportation',
                    filter: ['in', 'class', 'secondary', 'tertiary'],
                    minzoom: 8,
                    paint: {
                        'line-color': '#808090',
                        'line-width': ['interpolate', ['linear'], ['zoom'], 8, 0.3, 14, 2],
                    },
                },
                // Roads — minor
                {
                    id: 'road-minor',
                    type: 'line',
                    source: 'pir',
                    'source-layer': 'transportation',
                    filter: ['in', 'class', 'minor', 'service', 'path'],
                    minzoom: 10,
                    paint: {
                        'line-color': '#606070',
                        'line-width': ['interpolate', ['linear'], ['zoom'], 10, 0.2, 14, 1],
                    },
                },
                // Boundaries
                {
                    id: 'boundary',
                    type: 'line',
                    source: 'pir',
                    'source-layer': 'boundary',
                    paint: {
                        'line-color': '#6a6a8e',
                        'line-width': 1,
                        'line-dasharray': [3, 2],
                    },
                },
            ],
        },
    });

    map.addControl(new maplibregl.NavigationControl(), 'top-left');
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
