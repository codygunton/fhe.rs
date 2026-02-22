// pir-map-shared/frontend/tile-batch.js
export class TileBatchDispatcher {
    // pirBackend: object with processBatch(tiles, abortSignal) method
    // coalesceMs: time window in ms (default 50)
    constructor(pirBackend, coalesceMs = 50) {
        this._backend = pirBackend;
        this._pending = new Map();  // key → {z, x, y, slots, resolvers: [], reject}
        this._timer = null;
        this._coalesceMs = coalesceMs;
    }

    // Enqueue a tile request. Returns Promise<ArrayBuffer> (PBF or empty).
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
            this._pending.set(key, { z, x, y, slots, resolvers: [resolve], reject });
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

        // Build tiles array for processBatch
        const tiles = batch.map(([key, {z, x, y, slots, resolvers, reject}]) =>
            ({key, z, x, y, slots, resolvers, reject}));

        console.log(`Dispatcher flush: ${tiles.length} tile(s)`);

        let results;
        try {
            results = await this._backend.processBatch(
                tiles.map(t => ({key: t.key, z: t.z, x: t.x, y: t.y, slots: t.slots})),
                null
            );
        } catch (err) {
            for (const {resolvers, reject} of tiles) {
                reject(err);
                for (const r of resolvers.slice(1)) r(new ArrayBuffer(0));
            }
            return;
        }

        // Distribute results, yield between tiles for progressive rendering
        for (const {key, resolvers} of tiles) {
            const result = results.get(key) ?? new ArrayBuffer(0);
            for (const r of resolvers) r(result);
            await new Promise(r => setTimeout(r, 0));
        }
    }
}
