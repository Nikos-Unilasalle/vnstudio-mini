/**
 * A persistent cache for downloaded rasters.
 *
 * The desktop writes each Copernicus request to a GeoTIFF under
 * `copernicus_cache/` and reads it back on the next run. The browser has no
 * disk to write to, but it does have IndexedDB — which, unlike localStorage,
 * has room for tens of megabytes and stores typed arrays natively through the
 * structured clone algorithm, so a `GeoRaster` goes in and comes back out
 * without being serialised to text and re-parsed.
 *
 * Without this, reloading the page costs the whole download again: forty
 * seconds for a modest study area, and several minutes for a large one at ten
 * metres. It is available in workers, which is where the nodes run.
 */
import type { GeoRaster } from '../geo'

const DB_NAME = 'vnstudio-geo'
const DB_VERSION = 1
const STORE = 'rasters'

/**
 * How much of the origin's storage the cache may take.
 *
 * A single six-band scene at 10 m over a 25 km box is close to 90 MB, so the
 * budget has to be generous or it would evict the very thing it just stored.
 * It is still bounded, and still a fraction of what the browser grants.
 */
const MAX_BYTES = 1_500_000_000
const BUDGET_FRACTION = 0.5

interface Record_ {
  key: string
  raster: GeoRaster
  meta: Record<string, unknown>
  bytes: number
  savedAt: number
  lastUsed: number
}

let dbPromise: Promise<IDBDatabase | null> | null = null

function openDatabase(): Promise<IDBDatabase | null> {
  if (dbPromise) return dbPromise
  dbPromise = new Promise((resolve) => {
    // Private windows and storage-blocking settings make this throw or fail;
    // a missing cache must never stop a fetch, so every path resolves to null.
    let request: IDBOpenDBRequest
    try {
      request = indexedDB.open(DB_NAME, DB_VERSION)
    } catch {
      resolve(null)
      return
    }
    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'key' })
        store.createIndex('lastUsed', 'lastUsed')
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => resolve(null)
    request.onblocked = () => resolve(null)
  })
  return dbPromise
}

function promisify<T>(request: IDBRequest<T>): Promise<T | null> {
  return new Promise((resolve) => {
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => resolve(null)
  })
}

/** Bytes a raster occupies, which is what the budget is spent on. */
function sizeOf(raster: GeoRaster): number {
  let total = 0
  for (const band of raster.bands) total += band.byteLength
  return total
}

export interface StoredRaster {
  raster: GeoRaster
  meta: Record<string, unknown>
  savedAt: number
}

/** The cached raster for this query, or null. Reading marks it recently used. */
export async function readRaster(key: string): Promise<StoredRaster | null> {
  const db = await openDatabase()
  if (!db) return null
  try {
    const transaction = db.transaction(STORE, 'readwrite')
    const store = transaction.objectStore(STORE)
    const found = (await promisify(store.get(key))) as Record_ | null | undefined
    if (!found) return null
    // Touch it so eviction sweeps the genuinely stale entries first.
    store.put({ ...found, lastUsed: Date.now() })
    return { raster: found.raster, meta: found.meta, savedAt: found.savedAt }
  } catch {
    return null
  }
}

/** How many bytes the cache may hold, given what the browser is willing to grant. */
async function budget(): Promise<number> {
  try {
    const estimate = await navigator.storage?.estimate?.()
    if (estimate?.quota) return Math.min(MAX_BYTES, estimate.quota * BUDGET_FRACTION)
  } catch {
    // No estimate available; fall back to the fixed ceiling.
  }
  return MAX_BYTES
}

/**
 * Store a raster, evicting least-recently-used entries to stay inside budget.
 *
 * A raster larger than the whole budget is not stored at all rather than
 * emptying the cache to make room for something that will not fit twice.
 */
export async function writeRaster(
  key: string,
  raster: GeoRaster,
  meta: Record<string, unknown>
): Promise<void> {
  const db = await openDatabase()
  if (!db) return
  const bytes = sizeOf(raster)
  const allowed = await budget()
  if (bytes > allowed) return

  try {
    const transaction = db.transaction(STORE, 'readwrite')
    const store = transaction.objectStore(STORE)
    const existing = ((await promisify(store.getAll())) ?? []) as Record_[]

    let total = bytes
    for (const entry of existing) if (entry.key !== key) total += entry.bytes
    if (total > allowed) {
      const byAge = existing.filter((e) => e.key !== key).sort((a, b) => a.lastUsed - b.lastUsed)
      for (const victim of byAge) {
        if (total <= allowed) break
        store.delete(victim.key)
        total -= victim.bytes
      }
    }

    const now = Date.now()
    store.put({ key, raster, meta, bytes, savedAt: now, lastUsed: now } satisfies Record_)
  } catch {
    // A full or unavailable store is not a reason to fail a fetch that worked.
  }
}

export interface CacheSummary {
  entries: number
  bytes: number
}

/** What the cache currently holds — for reporting, and for a clear button. */
export async function summarise(): Promise<CacheSummary> {
  const db = await openDatabase()
  if (!db) return { entries: 0, bytes: 0 }
  try {
    const store = db.transaction(STORE, 'readonly').objectStore(STORE)
    const all = ((await promisify(store.getAll())) ?? []) as Record_[]
    return { entries: all.length, bytes: all.reduce((sum, e) => sum + e.bytes, 0) }
  } catch {
    return { entries: 0, bytes: 0 }
  }
}

/** Drop everything. */
export async function clearRasters(): Promise<void> {
  const db = await openDatabase()
  if (!db) return
  try {
    db.transaction(STORE, 'readwrite').objectStore(STORE).clear()
  } catch {
    // Nothing to do; the cache is best-effort in both directions.
  }
}
