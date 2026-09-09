/**
 * Microsoft Planetary Computer, read straight from the browser.
 *
 * This is the credential-free half of the Copernicus node. The catalogue is
 * open, its assets are cloud-optimised GeoTIFFs, and a short-lived SAS token
 * signs the reads — so a Sentinel-2 scene can be windowed over HTTP range
 * requests without a server, a proxy, or an account. What the desktop delegates
 * to `pystac_client` + `odc.stac` happens here in about two hundred lines.
 */
import { buildGrid, composite, warpToGrid, type LonLatBox, type SourceRaster, type TargetGrid } from './grid'
import { lonLatToUtm, utmToLonLat } from '../proj'
import { request } from './request'

const STAC_ROOT = 'https://planetarycomputer.microsoft.com/api/stac/v1'
const SAS_ROOT = 'https://planetarycomputer.microsoft.com/api/sas/v1/token'

export interface StacItem {
  id: string
  collection: string
  datetime: string | null
  cloudCover: number | null
  orbitState: string | null
  assets: Record<string, { href: string }>
  properties: Record<string, unknown>
}

interface SasToken {
  token: string
  expiresAt: number
}

const sasCache = new Map<string, SasToken>()

/** A signing token for a collection, reused until a minute before it lapses. */
async function sasToken(collection: string): Promise<string> {
  const cached = sasCache.get(collection)
  if (cached && cached.expiresAt - 60_000 > Date.now()) return cached.token
  const response = await request(`${SAS_ROOT}/${collection}`)
  if (!response.ok) throw new Error(`jeton SAS refusé pour ${collection} (${response.status})`)
  const body = (await response.json()) as { token?: string; ['msft:expiry']?: string }
  const token = body.token ?? ''
  const expiry = body['msft:expiry'] ? Date.parse(body['msft:expiry']) : Date.now() + 30 * 60_000
  sasCache.set(collection, { token, expiresAt: expiry })
  return token
}

function signed(href: string, token: string): string {
  if (!token) return href
  return href.includes('?') ? `${href}&${token}` : `${href}?${token}`
}

export interface SearchOptions {
  collection: string
  box: LonLatBox
  /** `null` for the static products, which carry no meaningful date range. */
  dateRange: [string, string] | null
  cloudMax: number | null
  orbit: 'ascending' | 'descending' | null
  limit: number
}

/** Search the catalogue, newest last, already filtered on cloud and orbit. */
export async function searchStac(options: SearchOptions): Promise<StacItem[]> {
  const body: Record<string, unknown> = {
    collections: [options.collection],
    bbox: [options.box.west, options.box.south, options.box.east, options.box.north],
    limit: 200,
  }
  if (options.dateRange) body.datetime = `${options.dateRange[0]}/${options.dateRange[1]}`

  const response = await request(`${STAC_ROOT}/search`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(`recherche STAC échouée (${response.status})`)
  const payload = (await response.json()) as { features?: Array<Record<string, any>> }

  let items: StacItem[] = (payload.features ?? []).map((feature) => ({
    id: String(feature.id),
    collection: String(feature.collection ?? options.collection),
    datetime: feature.properties?.datetime ?? null,
    cloudCover:
      feature.properties?.['eo:cloud_cover'] !== undefined
        ? Number(feature.properties['eo:cloud_cover'])
        : null,
    orbitState: feature.properties?.['sat:orbit_state'] ?? null,
    assets: feature.assets ?? {},
    properties: feature.properties ?? {},
  }))

  if (options.orbit) {
    items = items.filter((item) => (item.orbitState ?? '').toLowerCase() === options.orbit)
  }
  if (options.cloudMax !== null) {
    items = items.filter((item) => item.cloudCover === null || item.cloudCover <= options.cloudMax!)
  }
  items.sort((a, b) => (a.datetime ?? '').localeCompare(b.datetime ?? '') || a.id.localeCompare(b.id))
  return items
}

/** Spread `count` picks evenly over the scene list, as the desktop does. */
export function subsample<T>(items: T[], count: number): T[] {
  if (items.length <= count) return items
  if (count <= 1) return items.slice(0, Math.max(0, count))
  const out: T[] = []
  for (let i = 0; i < count; i++) {
    out.push(items[Math.round((i * (items.length - 1)) / (count - 1))])
  }
  return out
}

/**
 * Read the part of one COG that covers `grid`.
 *
 * The overview pyramid is walked by hand: the coarsest level still finer than
 * the target resolution is the one read, which is what keeps a 10980×10980
 * Sentinel-2 band down to a few hundred kilobytes over the wire.
 */
export async function readCogWindow(url: string, grid: TargetGrid): Promise<SourceRaster | null> {
  const { fromUrl } = await import('geotiff')
  const tiff = await fromUrl(url)
  const count = await tiff.getImageCount()

  const full = await tiff.getImage(0)
  const epsg = readEpsg(full)

  // Target footprint expressed in the source CRS.
  const footprint = sourceFootprint(grid, epsg)
  if (!footprint) return null

  // Overviews run fine to coarse. Reading at the target resolution would leave
  // the warp nothing to average and shifts every sample by half a pixel, so the
  // level taken is the coarsest one still at least twice as fine as the target —
  // enough detail for the bilinear pass, without pulling the full raster.
  const targetInSourceUnits = epsg === 4326 ? grid.resolution / 111320 : grid.resolution
  let chosen = full
  for (let i = 1; i < count; i++) {
    const image = await tiff.getImage(i)
    // Overview IFDs carry no affine transform of their own; the full-resolution
    // image is the reference geotiff.js scales against.
    const [rx] = image.getResolution(full).map(Math.abs)
    if (rx > targetInSourceUnits / 2) break
    chosen = image
  }

  const [ox, oy] = full.getOrigin()
  const [rx, ry] = chosen.getResolution(full)
  const iw = chosen.getWidth()
  const ih = chosen.getHeight()

  // Pixel window covering the footprint, clamped to the image and padded by one
  // pixel so bilinear sampling at the edge has a neighbour to reach for.
  const left = Math.floor((footprint.minX - ox) / rx) - 1
  const right = Math.ceil((footprint.maxX - ox) / rx) + 1
  // ry is negative for a north-up image, so the row order follows maxY → minY.
  const top = Math.floor((footprint.maxY - oy) / ry) - 1
  const bottom = Math.ceil((footprint.minY - oy) / ry) + 1

  const x0 = Math.max(0, Math.min(iw, Math.min(left, right)))
  const x1 = Math.max(0, Math.min(iw, Math.max(left, right)))
  const y0 = Math.max(0, Math.min(ih, Math.min(top, bottom)))
  const y1 = Math.max(0, Math.min(ih, Math.max(top, bottom)))
  if (x1 - x0 < 1 || y1 - y0 < 1) return null

  const rasters = (await chosen.readRasters({
    window: [x0, y0, x1, y1],
    interleave: false,
    fillValue: NaN,
  })) as unknown as { width: number; height: number; [index: number]: ArrayLike<number> }

  const band = rasters[0]
  const width = rasters.width
  const height = rasters.height
  const data = new Float32Array(width * height)
  const nodata = readNodata(chosen)
  for (let i = 0; i < data.length; i++) {
    const v = band[i]
    // A categorical product uses 0 for "no data", which is also a legal class
    // code nowhere in these palettes, so it is safe to blank either way.
    data[i] = nodata !== null && v === nodata ? NaN : v
  }

  return {
    data,
    width,
    height,
    minX: ox + x0 * rx,
    maxX: ox + x1 * rx,
    maxY: oy + y0 * ry,
    minY: oy + y1 * ry,
    epsg,
  }
}

function readEpsg(image: any): number | null {
  const keys = image.getGeoKeys?.() ?? image.geoKeys ?? {}
  const projected = keys.ProjectedCSTypeGeoKey
  if (typeof projected === 'number' && projected > 0) return projected
  const geographic = keys.GeographicTypeGeoKey
  if (geographic === 4326) return 4326
  return null
}

function readNodata(image: any): number | null {
  const raw = image.getFileDirectory?.()?.GDAL_NODATA
  if (raw === undefined || raw === null) return null
  const value = Number(String(raw).replace(/\0/g, '').trim())
  return Number.isFinite(value) ? value : null
}

/** The grid's footprint expressed in the source CRS, as an axis-aligned box. */
function sourceFootprint(
  grid: TargetGrid,
  epsg: number | null
): { minX: number; minY: number; maxX: number; maxY: number } | null {
  if (epsg === null || epsg === grid.zone.epsg) {
    return { minX: grid.minX, minY: grid.minY, maxX: grid.maxX, maxY: grid.maxY }
  }
  if (epsg === 4326) {
    const b = gridLonLatBounds(grid)
    return { minX: b.west, minY: b.south, maxX: b.east, maxY: b.north }
  }
  // Another UTM zone: project the geographic outline into it.
  const b = gridLonLatBounds(grid)
  const zone = { zone: epsg % 100, north: epsg < 32700, epsg }
  const corners = [
    [b.west, b.south],
    [b.west, b.north],
    [b.east, b.south],
    [b.east, b.north],
  ].map(([lon, lat]) => lonLatToUtm(lon, lat, zone))
  return {
    minX: Math.min(...corners.map((c) => c[0])),
    maxX: Math.max(...corners.map((c) => c[0])),
    minY: Math.min(...corners.map((c) => c[1])),
    maxY: Math.max(...corners.map((c) => c[1])),
  }
}

function gridLonLatBounds(grid: TargetGrid): LonLatBox {
  let west = Infinity
  let east = -Infinity
  let south = Infinity
  let north = -Infinity
  const STEPS = 8
  for (let i = 0; i <= STEPS; i++) {
    const t = i / STEPS
    const x = grid.minX + (grid.maxX - grid.minX) * t
    const y = grid.minY + (grid.maxY - grid.minY) * t
    for (const [lon, lat] of [
      utmToLonLat(x, grid.minY, grid.zone),
      utmToLonLat(x, grid.maxY, grid.zone),
      utmToLonLat(grid.minX, y, grid.zone),
      utmToLonLat(grid.maxX, y, grid.zone),
    ]) {
      if (lon < west) west = lon
      if (lon > east) east = lon
      if (lat < south) south = lat
      if (lat > north) north = lat
    }
  }
  return { west, east, south, north }
}

export interface StacFetchOptions extends SearchOptions {
  /** STAC asset keys, one per output band. */
  assetKeys: string[]
  resolution: number
  categorical: boolean
  maxScenes: number
  method: 'median' | 'mean' | 'first' | 'min' | 'max'
  /**
   * Per-scene conversion from stored counts to physical units. It has to be per
   * scene, not per collection: Sentinel-2's harmonisation offset changed with
   * the processing baseline, so two scenes of the same tile a year apart need
   * different arithmetic before they can be composited together.
   */
  rescale?: (item: StacItem) => { scale: number; offset: number } | null
  onProgress?: (fraction: number, message: string) => void
}

export interface StacResult {
  bands: Float32Array[]
  grid: TargetGrid
  scenes: StacItem[]
}

/** Search, read and composite — the whole STAC path in one call. */
export async function fetchStac(options: StacFetchOptions): Promise<StacResult> {
  const grid = buildGrid(options.box, options.resolution)
  const found = await searchStac(options)
  if (found.length === 0) throw new Error('aucune scène pour cette emprise et ces dates')

  // A categorical product is a mosaic of static tiles, not a time series: every
  // tile intersecting the box is needed, and averaging class codes would be
  // meaningless. Newest first, so the "first non-NaN wins" mosaic below prefers
  // the most recent edition wherever two of them overlap.
  const scenes = options.categorical ? found.slice().reverse() : subsample(found, Math.max(1, options.maxScenes))
  const token = await sasToken(scenes[0].collection)

  const bands: Float32Array[] = []
  const total = scenes.length * options.assetKeys.length
  let done = 0
  let lastFailure: unknown = null

  for (const assetKey of options.assetKeys) {
    const layers: Float32Array[] = []
    for (const scene of scenes) {
      const asset = scene.assets[assetKey]
      done += 1
      options.onProgress?.(done / total, `${assetKey} — ${scene.id}`)
      if (!asset?.href) continue
      try {
        const source = await readCogWindow(signed(asset.href, token), grid)
        if (!source) continue
        const layer = warpToGrid(source, grid, options.categorical)
        const conversion = options.rescale?.(scene)
        if (conversion) {
          for (let i = 0; i < layer.length; i++) {
            layer[i] = (layer[i] + conversion.offset) * conversion.scale
          }
        }
        layers.push(layer)
      } catch (error) {
        // One unreachable scene should not sink a composite built from many,
        // but a silent skip would make a whole-collection failure look empty.
        console.warn(`[STAC] ${scene.id} / ${assetKey} illisible :`, error)
        lastFailure = error
      }
    }
    if (layers.length === 0) {
      // Say why, not just that: every scene failing for the same reason is
      // almost always one cause, and it is the one worth reporting.
      const cause = lastFailure instanceof Error ? ` — ${lastFailure.message}` : ''
      throw new Error(`aucune donnée lisible pour l’asset « ${assetKey} »${cause}`)
    }
    // Categorical tiles are disjoint in space, so "first non-NaN" mosaics them.
    bands.push(composite(layers, options.categorical ? 'first' : options.method))
  }

  return { bands, grid, scenes }
}
