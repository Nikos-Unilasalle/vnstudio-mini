/**
 * Regenerates web-engine/schemas.json from the desktop engine's own registry.
 *
 * The desktop app receives its node schemas over the WebSocket at runtime. The
 * web build has no engine to ask, so the schemas are baked in at build time —
 * filtered down to the nodes that have a browser implementation. Keeping them
 * generated (rather than hand-written) means ports, params and labels can never
 * drift from the Python plugins they mirror.
 *
 * Usage:
 *   1. Dump the full registry from the desktop repo:
 *      .venv/bin/python -c "...; json.dump(registry.NODE_SCHEMAS, open('/tmp/all_schemas.json','w'), default=str)"
 *   2. node scripts/gen-schemas.mjs /tmp/all_schemas.json
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { SUPPORTED_TYPES } from '../web-engine/supported.mjs'

/**
 * Parameters the browser build cannot honour, removed so the panel never offers
 * a control that does nothing. Credentials are the important case: leaving them
 * as graph parameters would write a user's Copernicus secret into every `.vn`
 * file they share. The web node reads them from the local key vault instead.
 */
const DROPPED_PARAMS = {
  geo_copernicus: [
    'client_id', 'client_secret',      // live in ~/.vnstudio/secrets.json, not the graph
    '_sec_download', 'max_tile_px', 'cache_dir',  // no disk cache, no tiled downloader
    'mosaic_mode',                     // CLOUD_FREE mosaicking is not implemented
    'stac_to_db',                      // decided by the collection, not the user
    'stac_scene_timeout', 'stac_min_ok',  // fetch is a single awaited request here
  ],
  geo_land_cover: ['gcp_project'],     // Earth Engine is unreachable; the data comes from STAC
}

const source = process.argv[2] ?? '/tmp/all_schemas.json'
const all = JSON.parse(readFileSync(source, 'utf8'))
const bySupported = new Map(all.filter((s) => SUPPORTED_TYPES.includes(s.type)).map((s) => [s.type, s]))

const missing = SUPPORTED_TYPES.filter((t) => !bySupported.has(t))
if (missing.length > 0) {
  console.error(`Unknown type_ids (not in the desktop registry): ${missing.join(', ')}`)
  process.exit(1)
}

const ordered = SUPPORTED_TYPES.map((t) => {
  const schema = bySupported.get(t)
  const dropped = DROPPED_PARAMS[t]
  if (!dropped || !Array.isArray(schema.params)) return schema
  return { ...schema, params: schema.params.filter((p) => !dropped.includes(p.id)) }
})
writeFileSync(new URL('../web-engine/schemas.json', import.meta.url), JSON.stringify(ordered, null, 2))
console.log(`Wrote ${ordered.length} schemas (of ${all.length} in the desktop registry).`)
