// pir-map-shared/frontend/map-setup.js
// fetchTileFn: async (z, x, y, abortSignal) => ArrayBuffer
export function initMap(mappingData, fetchTileFn) {
    // Register pir:// protocol
    maplibregl.addProtocol('pir', async (params, abortController) => {
        // URL format: pir://{z}/{x}/{y} or pir://tiles/{z}/{x}/{y}
        const parts = params.url.replace('pir://', '').split('/');
        // Handle both pir://z/x/y and pir://tiles/z/x/y
        if (parts[0] === 'tiles') parts.shift();
        const [z, x, y] = parts.map(Number);
        const data = await fetchTileFn(z, x, y, abortController.signal);
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
