/**
 * The remote imagery sources: `geo_copernicus` and `geo_land_cover`.
 *
 * The desktop reaches these through rasterio, odc.stac and the Earth Engine
 * SDK. None of that exists in a browser, but two of the three backends turn out
 * not to need it: Planetary Computer's catalogue is open and its assets are
 * cloud-optimised, and tile servers are just HTTP. Earth Engine is the one that
 * genuinely cannot be reached — it refuses cross-origin requests and expects a
 * service-account OAuth flow — so `geo_land_cover` serves the same ESA
 * WorldCover product from Planetary Computer instead, which needs no account.
 *
 * Fetching is deliberately on a trigger. A graph re-runs every frame; a network
 * read that takes seconds must not, so the result is cached against the inputs
 * that produced it and only a Fetch press (or a changed query) goes to the wire.
 */
import type { NodeImpl, RunContext } from '../types'
import { autoStretch, GeoRaster } from '../geo'
import { buildGrid, gridBounds, gridTransform, warpToGrid, type LonLatBox, type TargetGrid } from '../remote/grid'
import { utmZoneFromEpsg } from '../proj'
import { fetchStac } from '../remote/stac'
import { fetchBasemap } from '../remote/basemap'
import { fetchCdse } from '../remote/cdse'

/* ----------------------------------------------------------- collections */

type Backend = 'stac' | 'basemap' | 'sh'

interface Collection {
  backend: Backend
  /** STAC collection id, Sentinel Hub type id, or XYZ url template. */
  source: string
  bands: string[]
  /** STAC asset keys, one per band; unused by the other backends. */
  assetKeys?: string[]
  /** Band indices (into `bands`) used for the RGB preview. */
  rgb: [number, number, number]
  cloudFilter: boolean
  categorical: boolean
  palette?: Record<number, [number, number, number]>
  /** Static products carry no meaningful date range. */
  ignoreDate: boolean
  units?: string | null
  toDecibels?: boolean
  mosaickingOrder?: 'leastCC' | 'mostRecent'
  /** Sentinel Hub only: which Copernicus DEM to serve under the shared `dem` type. */
  demInstance?: 'COPERNICUS_30' | 'COPERNICUS_90'
}

const WORLDCOVER_PALETTE: Record<number, [number, number, number]> = {
  10: [0, 100, 0], 20: [34, 187, 255], 30: [76, 255, 255], 40: [255, 150, 240],
  50: [0, 0, 250], 60: [180, 180, 180], 70: [240, 240, 240], 80: [200, 100, 0],
  90: [160, 150, 0], 95: [117, 207, 0], 100: [160, 230, 250],
}

const IO_LULC_PALETTE: Record<number, [number, number, number]> = {
  1: [171, 91, 26], 2: [33, 130, 53], 4: [174, 196, 123], 5: [92, 219, 255],
  7: [30, 30, 194], 8: [75, 113, 149], 9: [245, 245, 245], 10: [220, 200, 200],
  11: [79, 175, 177],
}

/**
 * The collection list, in the same order as the desktop's enum so a `.vn` file
 * written there selects the same entry here. Entries the browser cannot serve
 * are kept in place and refused at run time with an explanation, rather than
 * silently shifting every index after them.
 */
const COLLECTIONS: Record<string, Collection> = {
  'Sentinel-2 L2A': {
    backend: 'sh', source: 'sentinel-2-l2a',
    bands: ['B04', 'B03', 'B02', 'B08'], rgb: [0, 1, 2],
    cloudFilter: true, categorical: false, ignoreDate: false, units: 'REFLECTANCE',
    mosaickingOrder: 'leastCC',
  },
  'Sentinel-2 L1C': {
    backend: 'sh', source: 'sentinel-2-l1c',
    bands: ['B04', 'B03', 'B02', 'B08'], rgb: [0, 1, 2],
    cloudFilter: true, categorical: false, ignoreDate: false, units: 'REFLECTANCE',
    mosaickingOrder: 'leastCC',
  },
  'Sentinel-1 GRD': {
    backend: 'sh', source: 'sentinel-1-grd',
    bands: ['VV', 'VH'], rgb: [0, 1, 0],
    cloudFilter: false, categorical: false, ignoreDate: false,
    units: 'LINEAR_POWER', toDecibels: true, mosaickingOrder: 'mostRecent',
  },
  'Copernicus DEM GLO-30': {
    backend: 'sh', source: 'dem', bands: ['DEM'], rgb: [0, 0, 0],
    cloudFilter: false, categorical: false, ignoreDate: true, units: null,
    mosaickingOrder: 'mostRecent', demInstance: 'COPERNICUS_30',
  },
  'Copernicus DEM GLO-90': {
    backend: 'sh', source: 'dem', bands: ['DEM'], rgb: [0, 0, 0],
    cloudFilter: false, categorical: false, ignoreDate: true, units: null,
    mosaickingOrder: 'mostRecent', demInstance: 'COPERNICUS_90',
  },
  'Sentinel-1 RTC (Planetary)': {
    backend: 'stac', source: 'sentinel-1-rtc',
    bands: ['vv', 'vh'], assetKeys: ['vv', 'vh'], rgb: [0, 1, 0],
    cloudFilter: false, categorical: false, ignoreDate: false, toDecibels: true,
  },
  'ESA WorldCover (10m)': {
    backend: 'stac', source: 'esa-worldcover',
    bands: ['lulc_class'], assetKeys: ['map'], rgb: [0, 0, 0],
    cloudFilter: false, categorical: true, ignoreDate: true, palette: WORLDCOVER_PALETTE,
  },
  'io-lulc Annual': {
    backend: 'stac', source: 'io-lulc-annual-v02',
    bands: ['lulc_class'], assetKeys: ['data'], rgb: [0, 0, 0],
    cloudFilter: false, categorical: true, ignoreDate: true, palette: IO_LULC_PALETTE,
  },
  'Sentinel-2 L2A (Planetary)': {
    backend: 'stac', source: 'sentinel-2-l2a',
    bands: ['B04', 'B03', 'B02', 'B08', 'B11'],
    assetKeys: ['B04', 'B03', 'B02', 'B08', 'B11'], rgb: [0, 1, 2],
    cloudFilter: true, categorical: false, ignoreDate: false,
  },
  'Copernicus DEM GLO-30 (Planetary)': {
    backend: 'stac', source: 'cop-dem-glo-30',
    bands: ['data'], assetKeys: ['data'], rgb: [0, 0, 0],
    cloudFilter: false, categorical: false, ignoreDate: true,
  },
  'JRC Global Surface Water': {
    backend: 'stac', source: 'jrc-gsw',
    bands: ['occurrence'], assetKeys: ['occurrence'], rgb: [0, 0, 0],
    cloudFilter: false, categorical: false, ignoreDate: true,
  },
  'Google Satellite': {
    backend: 'basemap', source: 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    bands: ['R', 'G', 'B'], rgb: [0, 1, 2], cloudFilter: false, categorical: false, ignoreDate: true,
  },
  'Google Hybrid': {
    backend: 'basemap', source: 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    bands: ['R', 'G', 'B'], rgb: [0, 1, 2], cloudFilter: false, categorical: false, ignoreDate: true,
  },
  'Google Roadmap': {
    backend: 'basemap', source: 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
    bands: ['R', 'G', 'B'], rgb: [0, 1, 2], cloudFilter: false, categorical: false, ignoreDate: true,
  },
  'Google Terrain': {
    backend: 'basemap', source: 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
    bands: ['R', 'G', 'B'], rgb: [0, 1, 2], cloudFilter: false, categorical: false, ignoreDate: true,
  },
  'OpenStreetMap': {
    backend: 'basemap', source: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    bands: ['R', 'G', 'B'], rgb: [0, 1, 2], cloudFilter: false, categorical: false, ignoreDate: true,
  },
  'Carto Positron': {
    backend: 'basemap', source: 'https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
    bands: ['R', 'G', 'B'], rgb: [0, 1, 2], cloudFilter: false, categorical: false, ignoreDate: true,
  },
  'Carto Dark Matter': {
    backend: 'basemap', source: 'https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
    bands: ['R', 'G', 'B'], rgb: [0, 1, 2], cloudFilter: false, categorical: false, ignoreDate: true,
  },
}

const COLLECTION_NAMES = Object.keys(COLLECTIONS)

/* ---------------------------------------------------------------- helpers */

function parseBox(text: string): LonLatBox | null {
  const parts = text.split(',').map((v) => Number(v.trim()))
  if (parts.length !== 4 || parts.some((v) => !Number.isFinite(v))) return null
  const [west, south, east, north] = parts
  if (east <= west || north <= south) return null
  return { west, south, east, north }
}

function toRaster(bands: Float32Array[], names: string[], grid: TargetGrid, categorical: boolean): GeoRaster {
  return {
    bands,
    band_names: names,
    count: bands.length,
    width: grid.width,
    height: grid.height,
    crs: `EPSG:${grid.zone.epsg}`,
    transform: gridTransform(grid),
    bounds: gridBounds(grid),
    nodata: categorical ? 0 : null,
    dtype: categorical ? 'uint8' : 'float32',
  }
}

/** RGB preview: a class palette when categorical, a percentile stretch otherwise. */
function renderPreview(cv: any, ctx: RunContext, raster: GeoRaster, collection: Collection): any {
  const preview = ctx.track(new cv.Mat(raster.height, raster.width, cv.CV_8UC3))
  const bytes = preview.data as Uint8Array

  if (collection.categorical && collection.palette) {
    const values = raster.bands[0]
    for (let p = 0; p < values.length; p++) {
      const colour = collection.palette[Math.round(values[p])] ?? [40, 40, 40]
      bytes[p * 3] = colour[0]
      bytes[p * 3 + 1] = colour[1]
      bytes[p * 3 + 2] = colour[2]
    }
    return preview
  }

  const pick = (index: number) => raster.bands[Math.min(index, raster.bands.length - 1)]
  const r = autoStretch(pick(collection.rgb[0]))
  const g = autoStretch(pick(collection.rgb[1]))
  const b = autoStretch(pick(collection.rgb[2]))
  for (let p = 0; p < r.length; p++) {
    bytes[p * 3] = b[p]
    bytes[p * 3 + 1] = g[p]
    bytes[p * 3 + 2] = r[p]
  }
  return preview
}

interface FetchCache {
  key: string
  raster: GeoRaster
  meta: Record<string, unknown>
  collection: Collection
}

/* ------------------------------------------------------------ the fetcher */

interface Query {
  collectionName: string
  collection: Collection
  box: LonLatBox
  resolution: number
  dateStart: string
  dateEnd: string
  cloudMax: number
  bands: string[]
  method: 'median' | 'mean' | 'first' | 'min' | 'max'
  maxScenes: number
  orbit: 'ascending' | 'descending' | null
}

async function runQuery(query: Query, ctx: RunContext): Promise<{ raster: GeoRaster; meta: Record<string, unknown> }> {
  const { collection } = query
  const grid = buildGrid(query.box, query.resolution)
  // The only channel a worker-side node has to say "still working" is a live field.
  const report = (fraction: number, message: string) =>
    ctx.emit('status', `${Math.round(fraction * 100)} % — ${message}`)

  if (collection.backend === 'basemap') {
    const result = await fetchBasemap(collection.source, grid, report)
    const raster = toRaster(result.bands, result.bandNames, grid, false)
    return {
      raster,
      meta: {
        source: query.collectionName, backend: 'basemap', zoom: result.zoom,
        tiles: result.tiles, crs: raster.crs, width: grid.width, height: grid.height,
        resolution_m: grid.resolution, bounds: raster.bounds, band_names: result.bandNames,
      },
    }
  }

  if (collection.backend === 'sh') {
    const bands = await fetchCdse({
      collectionId: collection.source,
      bands: query.bands,
      grid,
      dateRange: [query.dateStart, query.dateEnd],
      cloudMax: collection.cloudFilter ? query.cloudMax : null,
      mosaickingOrder: collection.mosaickingOrder ?? 'mostRecent',
      toDecibels: collection.toDecibels ?? false,
      units: collection.units ?? null,
      demInstance: collection.demInstance ?? null,
    })
    const raster = toRaster(bands, query.bands.slice(0, bands.length), grid, false)
    return {
      raster,
      meta: {
        source: query.collectionName, backend: 'sentinel-hub (CDSE)',
        dates: `${query.dateStart} → ${query.dateEnd}`, crs: raster.crs,
        width: grid.width, height: grid.height, resolution_m: grid.resolution,
        bounds: raster.bounds, band_names: raster.band_names,
      },
    }
  }

  const assetKeys = collection.assetKeys ?? collection.bands
  const result = await fetchStac({
    collection: collection.source,
    box: query.box,
    dateRange: collection.ignoreDate ? null : [query.dateStart, query.dateEnd],
    cloudMax: collection.cloudFilter ? query.cloudMax : null,
    orbit: query.orbit,
    limit: 200,
    assetKeys,
    resolution: query.resolution,
    categorical: collection.categorical,
    maxScenes: query.maxScenes,
    method: query.method,
    onProgress: report,
  })

  let bands = result.bands
  if (collection.toDecibels) {
    // Sentinel-1 RTC arrives as linear power; every downstream reading of SAR
    // expects decibels, and the floor keeps log(0) from poisoning the stack.
    bands = bands.map((source) => {
      const out = new Float32Array(source.length)
      for (let i = 0; i < source.length; i++) {
        out[i] = source[i] > 0 ? 10 * Math.log10(source[i]) : NaN
      }
      return out
    })
  }

  const raster = toRaster(bands, collection.bands.slice(0, bands.length), result.grid, collection.categorical)
  return {
    raster,
    meta: {
      source: query.collectionName, backend: 'planetary computer (STAC)',
      scenes: result.scenes.length,
      scene_ids: result.scenes.slice(0, 6).map((s) => s.id),
      dates: collection.ignoreDate ? 'produit statique' : `${query.dateStart} → ${query.dateEnd}`,
      composite: collection.categorical ? 'mosaïque' : query.method,
      crs: raster.crs, width: raster.width, height: raster.height,
      resolution_m: result.grid.resolution, bounds: raster.bounds,
      band_names: raster.band_names,
    },
  }
}

/* ------------------------------------------------------- geo_copernicus */

const METHODS = ['median', 'mean', 'first', 'min', 'max'] as const
const ORBITS = [null, 'ascending', 'descending'] as const

export const geoCopernicus: NodeImpl = async (inputs, params, ctx) => {
  const cv = ctx.cv
  const empty = { geotiff: null, main: null, preview: null, meta: null }

  const collectionName =
    COLLECTION_NAMES[Math.round(Number(params.collection) || 0)] ?? COLLECTION_NAMES[0]
  const collection = COLLECTIONS[collectionName]

  const boxText = String(inputs.bbox ?? params.bbox ?? '').trim()
  const box = parseBox(boxText)
  if (!box) {
    if (boxText) throw new Error(`emprise invalide : « ${boxText} » (attendu ouest,sud,est,nord)`)
    return empty
  }

  const bandText = String(params.bands ?? '').trim()
  const bands = bandText ? bandText.split(',').map((b) => b.trim()).filter(Boolean) : collection.bands

  const query: Query = {
    collectionName,
    collection,
    box,
    resolution: Math.max(1, Math.round(Number(params.resolution) || 10)),
    dateStart: String(inputs.date_start ?? params.date_start ?? '2024-01-01'),
    dateEnd: String(inputs.date_end ?? params.date_end ?? '2024-06-01'),
    cloudMax: Math.min(100, Math.max(0, Math.round(Number(params.cloud_max) || 0))),
    bands: collection.backend === 'sh' ? bands : collection.bands,
    method: METHODS[Math.round(Number(params.stac_composite) || 0)] ?? 'median',
    maxScenes: Math.max(1, Math.round(Number(params.stac_max_scenes) || 12)),
    orbit: ORBITS[Math.round(Number(params.stac_orbit) || 0)] ?? null,
  }

  const key = JSON.stringify(query, (name, value) => (name === 'collection' ? undefined : value))
  const stateKey = `${ctx.nodeId}:copernicus`
  const cache = ctx.state.get(stateKey) as FetchCache | undefined

  const trigger = Number(params.fetch) ? 1 : 0
  const previous = (ctx.state.get(`${stateKey}:trigger`) as number) ?? 0
  ctx.state.set(`${stateKey}:trigger`, trigger)
  const pressed = trigger === 1 && previous === 0

  if (cache && cache.key === key) {
    const preview = renderPreview(cv, ctx, cache.raster, cache.collection)
    return { geotiff: cache.raster, main: preview, preview, meta: cache.meta }
  }

  // A changed query invalidates the cache but does not itself go to the network:
  // downloading a scene on every keystroke would be hostile. Fetch is a button.
  if (!pressed) return empty

  const { raster, meta } = await runQuery(query, ctx)
  const fresh: FetchCache = { key, raster, meta, collection }
  ctx.state.set(stateKey, fresh)
  const preview = renderPreview(cv, ctx, raster, collection)
  return { geotiff: raster, main: preview, preview, meta }
}

/* ------------------------------------------------------- geo_land_cover */

/**
 * The grid an input raster already sits on.
 *
 * The class mask has to line up with the raster it describes, pixel for pixel,
 * or every downstream comparison is measuring registration error instead of
 * land cover. When the input carries a north-up UTM transform its grid is
 * reused verbatim; otherwise the best available guess is its bounds at its own
 * pixel count.
 */
function gridFromRaster(raster: GeoRaster): TargetGrid | null {
  const epsg = Number(String(raster.crs ?? '').replace(/^EPSG:/i, ''))
  const zone = Number.isFinite(epsg) ? utmZoneFromEpsg(epsg) : null
  const t = raster.transform
  if (zone && t && t.length >= 6 && t[1] === 0 && t[3] === 0 && t[0] > 0 && t[4] < 0) {
    const resolution = t[0]
    const minX = t[2]
    const maxY = t[5]
    return {
      width: raster.width,
      height: raster.height,
      resolution,
      zone,
      minX,
      maxY,
      maxX: minX + raster.width * resolution,
      minY: maxY - raster.height * resolution,
      box: raster.bounds ?? { west: 0, south: 0, east: 0, north: 0 },
    }
  }
  if (!raster.bounds) return null
  // No usable transform: fall back to the bounds, at whatever resolution gives
  // roughly the raster's own pixel count.
  const span = Math.max(1e-9, raster.bounds.east - raster.bounds.west)
  const metres = span * 111320 * Math.cos((((raster.bounds.north + raster.bounds.south) / 2) * Math.PI) / 180)
  return buildGrid(raster.bounds, Math.max(1, Math.round(metres / Math.max(1, raster.width))))
}

const LAND_CLASS_NAMES: Record<number, string> = {
  10: 'arbres', 20: 'arbustes', 30: 'prairie', 40: 'cultures', 50: 'bâti',
  60: 'sol nu', 70: 'neige et glace', 80: 'eau', 90: 'zone humide',
  95: 'mangrove', 100: 'lichens et mousses',
}

interface LandCoverCache {
  key: string
  raster: GeoRaster | null
  error: string | null
}

export const geoLandCover: NodeImpl = async (inputs, params, ctx) => {
  const cv = ctx.cv
  const empty = { mask: null, main: null, overlay: null, meta: null }

  const source = inputs.geotiff as GeoRaster | undefined
  if (!source || !Array.isArray(source.bands) || source.bands.length === 0) return empty
  const grid = gridFromRaster(source)
  if (!grid) return { ...empty, meta: 'raster sans géoréférencement — impossible de situer la zone' }

  const landClass = Math.round(Number(params.land_class) || 40)
  const opacity = Math.min(1, Math.max(0, Number(params.opacity ?? 0.5)))

  // The download depends on where and how finely, not on which class is picked,
  // so changing the class re-thresholds the cached raster without a new fetch.
  const key = JSON.stringify([grid.zone.epsg, grid.minX, grid.maxY, grid.width, grid.height, grid.resolution])
  const stateKey = `${ctx.nodeId}:landcover`
  let cache = ctx.state.get(stateKey) as LandCoverCache | undefined

  if (!cache || cache.key !== key) {
    try {
      ctx.emit('status', 'ESA WorldCover — recherche…')
      const collection = COLLECTIONS['ESA WorldCover (10m)']
      const result = await fetchStac({
        collection: collection.source,
        box: gridBounds(grid),
        dateRange: null,
        cloudMax: null,
        orbit: null,
        limit: 200,
        assetKeys: collection.assetKeys ?? collection.bands,
        resolution: grid.resolution,
        categorical: true,
        maxScenes: 1,
        method: 'first',
        onProgress: (fraction, message) => ctx.emit('status', `${Math.round(fraction * 100)} % — ${message}`),
      })
      // fetchStac builds its own grid from the box; resample onto the input's.
      const classes = result.grid.width === grid.width && result.grid.height === grid.height
        ? result.bands[0]
        : warpToGrid(
            {
              data: result.bands[0],
              width: result.grid.width,
              height: result.grid.height,
              minX: result.grid.minX,
              minY: result.grid.minY,
              maxX: result.grid.maxX,
              maxY: result.grid.maxY,
              epsg: result.grid.zone.epsg,
            },
            grid,
            true
          )
      cache = { key, raster: toRaster([classes], ['lulc_class'], grid, true), error: null }
    } catch (error) {
      // A failed fetch is cached too: without that, a graph running at video
      // rate would re-issue the same doomed request every single frame.
      cache = { key, raster: null, error: error instanceof Error ? error.message : String(error) }
    }
    ctx.state.set(stateKey, cache)
  }

  if (!cache.raster) return { ...empty, meta: cache.error }
  const classes = cache.raster.bands[0]

  const mask = ctx.track(new cv.Mat(grid.height, grid.width, cv.CV_8U))
  const maskBytes = mask.data as Uint8Array
  let hits = 0
  for (let p = 0; p < classes.length; p++) {
    const on = Math.round(classes[p]) === landClass
    maskBytes[p] = on ? 255 : 0
    if (on) hits++
  }

  // Overlay: the full palette dimmed under the selected class picked out in white.
  const overlay = ctx.track(new cv.Mat(grid.height, grid.width, cv.CV_8UC3))
  const overlayBytes = overlay.data as Uint8Array
  for (let p = 0; p < classes.length; p++) {
    const colour = WORLDCOVER_PALETTE[Math.round(classes[p])] ?? [40, 40, 40]
    const lit = maskBytes[p] === 255
    for (let c = 0; c < 3; c++) {
      const base = colour[c]
      overlayBytes[p * 3 + c] = lit
        ? Math.round(base * (1 - opacity) + 255 * opacity)
        : Math.round(base * 0.45)
    }
  }

  const percent = (100 * hits) / Math.max(1, classes.length)
  const name = LAND_CLASS_NAMES[landClass] ?? String(landClass)
  return {
    mask,
    main: overlay,
    overlay,
    meta: `ESA WorldCover ${grid.width}×${grid.height} @ ${grid.resolution} m — ` +
      `classe ${landClass} (${name}) : ${percent.toFixed(2)} % de l’emprise`,
  }
}
