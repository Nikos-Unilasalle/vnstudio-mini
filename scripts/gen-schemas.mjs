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

const source = process.argv[2] ?? '/tmp/all_schemas.json'
const all = JSON.parse(readFileSync(source, 'utf8'))
const bySupported = new Map(all.filter((s) => SUPPORTED_TYPES.includes(s.type)).map((s) => [s.type, s]))

const missing = SUPPORTED_TYPES.filter((t) => !bySupported.has(t))
if (missing.length > 0) {
  console.error(`Unknown type_ids (not in the desktop registry): ${missing.join(', ')}`)
  process.exit(1)
}

const ordered = SUPPORTED_TYPES.map((t) => bySupported.get(t))
writeFileSync(new URL('../web-engine/schemas.json', import.meta.url), JSON.stringify(ordered, null, 2))
console.log(`Wrote ${ordered.length} schemas (of ${all.length} in the desktop registry).`)
