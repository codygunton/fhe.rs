import init, { PIRClient } from './pkg/fhe_wasm.js';

// ─── State ────────────────────────────────────────────────────────────
let client = null;
let tileMapping = null;   // Map<string, number|number[]>  "z/x/y" → pirIndex or [indices]
let pirParams = null;
let queryCount = 0;
let totalLatencyMs = 0;
let lastQueryMs = 0;

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

// ─── Fetch a single PIR slot ─────────────────────────────────────────
async function fetchSlot(slotIdx, abortSignal) {
    const queryBytes = client.create_query(slotIdx);
    const resp = await fetch('/api/query', {
        method: 'POST',
        body: queryBytes,
        headers: { 'Content-Type': 'application/octet-stream' },
        signal: abortSignal,
    });
    if (!resp.ok) throw new Error(`Query failed for slot ${slotIdx}: ${resp.status}`);
    const encrypted = new Uint8Array(await resp.arrayBuffer());
    return client.decrypt_response(encrypted, slotIdx);
}

// ─── PIR tile fetching (supports multi-slot tiles) ───────────────────
async function fetchTileViaPIR(z, x, y, abortSignal) {
    const key = `${z}/${x}/${y}`;
    const pirIndex = tileMapping.get(key);

    if (pirIndex === undefined) {
        return new ArrayBuffer(0);
    }

    // Normalize: single slot → [slot], multi-slot array stays as-is
    const slots = Array.isArray(pirIndex) ? pirIndex : [pirIndex];
    console.log(`PIR fetch: ${key} → ${slots.length} slot(s) [${slots.join(',')}]`);

    const t0 = performance.now();

    try {
        // Fetch all slots in parallel (Flask proxy is threaded)
        const decryptedParts = await Promise.all(
            slots.map(slotIdx => fetchSlot(slotIdx, abortSignal))
        );

        // Concatenate all decrypted slot data into one buffer
        const totalLen = decryptedParts.reduce((s, part) => s + part.length, 0);
        const combined = new Uint8Array(totalLen);
        let offset = 0;
        for (const part of decryptedParts) {
            combined.set(new Uint8Array(part), offset);
            offset += part.length;
        }

        // Extract length-prefixed tile data from the concatenated buffer
        const trimmed = extractTileData(combined);
        if (trimmed.length === 0) {
            console.warn(`PIR ${key}: extractTileData returned empty (combined ${totalLen}B)`);
            return new ArrayBuffer(0);
        }

        // Decompress gzip → raw PBF
        let pbf;
        try {
            pbf = pako.inflate(trimmed);
        } catch (e) {
            console.warn(`Decompress failed for ${key} (${trimmed.length}B gzip): first bytes = ${Array.from(trimmed.slice(0, 8)).map(b => b.toString(16).padStart(2, '0')).join(' ')}`, e);
            return new ArrayBuffer(0);
        }

        console.log(`PIR ${key}: OK ${slots.length} slots → ${trimmed.length}B gzip → ${pbf.length}B PBF`);

        const elapsed = performance.now() - t0;
        queryCount++;
        totalLatencyMs += elapsed;
        lastQueryMs = elapsed;
        updatePirStats();

        return pbf.buffer;
    } catch (e) {
        console.error(`PIR ${key}: fetch failed:`, e.message || e);
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

    const center = mappingData.center || [-98.5, 39.8];
    const maxZoom = mappingData.max_zoom || 11;

    const map = new maplibregl.Map({
        container: 'map',
        center: center,
        zoom: 4,
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
