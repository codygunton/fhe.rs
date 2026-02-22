// pir-map-shared/frontend/tile-cache.js
export class LRUTileCache {
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
