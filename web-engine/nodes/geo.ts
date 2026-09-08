/**
 * The portable half of the `geo_*` family: a GeoTIFF reader and the nodes that
 * compute on a georeferenced raster.
 *
 * The desktop's data sources are not here — `geo_copernicus` authenticates
 * against Copernicus and unpacks Sentinel products with rasterio, and
 * `geo_land_cover` drives the Google Earth Engine SDK. Neither has a browser
 * equivalent. The reader takes their place: export a GeoTIFF from the desktop
 * (or drop any other one) and the rest of the chain runs unchanged.
 */
import type { NodeImpl, RunContext } from '../types'
import {
  autoStretch,
  band,
  gaussianField,
  GeoRaster,
  isGeoRaster,
  normalisedIndex,
  percentile,
  seededRandom,
  stretchToBytes,
  withBands,
} from '../geo'
import { applyColormap, hotColor, jetColor, plasmaColor, turboColor, viridisColor } from '../colormaps'

const COLORMAPS: ((v: number) => [number, number, number])[] = [
  viridisColor, plasmaColor, turboColor, jetColor, hotColor,
  // 'rdylgn' then 'gray', matching the desktop's key order.
  (v: number) => (v / 255 < 0.5 ? [220, Math.trunc(440 * (v / 255)), 30] : [Math.max(0, Math.trunc(440 * (1 - v / 255))), 200, 30]),
  (v: number) => [v, v, v],
]

function colormapAt(index: unknown): (v: number) => [number, number, number] {
  return COLORMAPS[Math.round(Number(index) || 0)] ?? viridisColor
}

/** A float band rendered through a colormap between two clamp values. */
function colorize(cv: any, ctx: RunContext, values: Float32Array, w: number, h: number, low: number, high: number, colour: unknown): any {
  const gray = new cv.Mat(h, w, cv.CV_8U)
  ;(gray.data as Uint8Array).set(stretchToBytes(values, low, high))
  const out = ctx.track(applyColormap(cv, gray, colormapAt(colour)))
  gray.delete()
  return out
}

/* ------------------------------------------------------------- bounding box */

export const geoBbox: NodeImpl = (_inputs, params) => {
  const west = Number(params.lon_min ?? -5.5)
  const south = Number(params.lat_min ?? 41)
  const east = Number(params.lon_max ?? 9.5)
  const north = Number(params.lat_max ?? 51.5)
  return { bbox: `${west},${south},${east},${north}` }
}

/* --------------------------------------------------------- GeoTIFF reader */

/** Cached per path, since decoding a scene is far too slow to redo every frame. */
interface ReaderCache { path: string; raster: GeoRaster }

export const geoGeotiffReader: NodeImpl = async (inputs, params, ctx) => {
  const cv = ctx.cv
  const path = String(params.file_path ?? '').trim()
  if (!path) return { geotiff: null, preview: null, meta: null }

  const key = `${ctx.nodeId}:geotiff`
  let cache = ctx.state.get(key) as ReaderCache | undefined
  if (!cache || cache.path !== path) {
    const { fromArrayBuffer } = await import('geotiff')
    const response = await fetch(path)
    if (!response.ok) return { geotiff: null, preview: null, meta: null }
    const tiff = await fromArrayBuffer(await response.arrayBuffer())
    const image = await tiff.getImage()
    const width = image.getWidth()
    const height = image.getHeight()
    const rasters = (await image.readRasters()) as unknown as ArrayLike<number>[]
    const nodata = image.getGDALNoData?.() ?? null

    const bands = Array.from({ length: rasters.length }, (_, i) => {
      const source = rasters[i]
      const values = new Float32Array(width * height)
      for (let p = 0; p < values.length; p++) {
        const v = source[p]
        // rasterio's reader zeroes nodata; the same is done here so the
        // downstream stretch and index maths see the desktop's values.
        values[p] = nodata !== null && v === nodata ? 0 : v
      }
      return values
    })

    let bounds: GeoRaster['bounds'] = null
    try {
      const [west, south, east, north] = image.getBoundingBox()
      bounds = { west, south, east, north }
    } catch {
      bounds = null
    }
    const origin = image.getOrigin?.()
    const resolution = image.getResolution?.()
    const transform = origin && resolution
      ? [resolution[0], 0, origin[0], 0, resolution[1], origin[1]]
      : null

    cache = {
      path,
      raster: {
        bands,
        band_names: bands.map((_, i) => `B${i + 1}`),
        count: bands.length,
        width,
        height,
        crs: null,
        transform,
        bounds,
        nodata,
        dtype: 'float32',
      },
    }
    ctx.state.set(key, cache)
  }

  const raster = cache.raster
  const pick = (value: unknown, fallback: number) =>
    Math.min(Math.max(1, Math.round(Number(value ?? fallback))), raster.count)
  const r = autoStretch(band(raster, pick(params.r_band, 1)))
  const g = autoStretch(band(raster, pick(params.g_band, Math.min(2, raster.count))))
  const b = autoStretch(band(raster, pick(params.b_band, Math.min(3, raster.count))))

  const preview = ctx.track(new cv.Mat(raster.height, raster.width, cv.CV_8UC3))
  const bytes = preview.data as Uint8Array
  for (let p = 0; p < r.length; p++) {
    bytes[p * 3] = b[p]
    bytes[p * 3 + 1] = g[p]
    bytes[p * 3 + 2] = r[p]
  }

  return {
    geotiff: raster,
    main: preview,
    preview,
    meta: {
      crs: raster.crs,
      band_count: raster.count,
      width: raster.width,
      height: raster.height,
      dtype: raster.dtype,
      bounds: raster.bounds,
      band_names: raster.band_names,
    },
  }
}

/* --------------------------------------------------------- GeoTIFF → mask */

export const geotiffToMask: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const raster = inputs.geotiff
  if (!isGeoRaster(raster)) return {}

  const requested = Math.round(Number(params.band_index ?? 0))
  const index = requested >= raster.count ? 0 : Math.max(0, requested)
  const values = raster.bands[index]
  const threshold = Number(params.threshold ?? 0.5)

  const mask = ctx.track(new cv.Mat(raster.height, raster.width, cv.CV_8U))
  const bytes = mask.data as Uint8Array
  for (let p = 0; p < values.length; p++) bytes[p] = values[p] > threshold ? 255 : 0
  return { mask, main: mask }
}

/* -------------------------------------------------------- band calculator */

/** The names the spectral-index node binds on top of B1…Bn. */
const ALIASES = ['NIR', 'RED', 'GREEN', 'BLUE', 'SWIR'] as const

/**
 * Compiles a band expression to a function over the band arrays.
 *
 * The desktop `eval`s the expression against numpy arrays, so `(B4-B3)/(B4+B3)`
 * is whole-array arithmetic. Here it is evaluated per pixel with the band values
 * as plain numbers, which reads the same and avoids materialising temporaries.
 *
 * `aliased` also binds NIR/RED/GREEN/BLUE/SWIR, which the spectral-index node
 * offers alongside the numbered bands — an expression such as
 * `BLUE + 2.5*GREEN - 1.5*(NIR + SWIR)` depends on them.
 */
function compileExpression(
  expression: string,
  count: number,
  aliased = false
): ((values: number[], alias?: number[]) => number) | null {
  const names = Array.from({ length: count }, (_, i) => `B${i + 1}`)
  try {
    const destructure = aliased ? `const [${ALIASES.join(', ')}] = alias;` : ''
    const body = `"use strict"; const [${names.join(', ')}] = v; ${destructure} return (${expression});`
    const fn = new Function('v', 'alias', 'sqrt', 'log', 'abs', 'exp', 'clip', body) as any
    const clip = (x: number, low: number, high: number) => Math.min(Math.max(x, low), high)
    return (values: number[], alias?: number[]) =>
      fn(values, alias ?? [], Math.sqrt, Math.log, Math.abs, Math.exp, clip)
  } catch {
    return null
  }
}

export const geoBandCalc: NodeImpl = (inputs, params, ctx) => {
  const raster = inputs.geotiff
  if (!isGeoRaster(raster)) return { raw: null, colormap: null }
  const expression = String(params.expression ?? '').trim()
  if (!expression) return { raw: null, colormap: null }

  const compute = compileExpression(expression, raster.count)
  if (!compute) {
    ctx.emit('error', 'Band Calc: expression invalide')
    return { raw: null, colormap: null }
  }

  const low = Number(params.clamp_min ?? -1)
  const high = Number(params.clamp_max ?? 1)
  const result = new Float32Array(raster.width * raster.height)
  const scratch = new Array(raster.count)
  try {
    for (let p = 0; p < result.length; p++) {
      for (let b = 0; b < raster.count; b++) scratch[b] = raster.bands[b][p]
      const value = compute(scratch)
      result[p] = value < low ? low : value > high ? high : value
    }
  } catch (error) {
    ctx.emit('error', `Band Calc: ${error instanceof Error ? error.message : String(error)}`)
    return { raw: null, colormap: null }
  }

  const coloured = colorize(ctx.cv, ctx, result, raster.width, raster.height, low, high, params.colormap)
  return {
    raw: withBands(raster, [result], ['result']),
    colormap: coloured,
    main: coloured,
  }
}

/* ------------------------------------------------------- spectral indices */

/** {nir, red, green, blue, swir} band numbers per sensor, as the desktop lists them. */
const SENSOR_PRESETS: [number, number, number, number, number][] = [
  [4, 1, 2, 3, 5],   // Manual
  [4, 1, 2, 3, 5],   // Sentinel-2
  [5, 4, 3, 2, 6],   // Landsat 8/9
  [4, 3, 2, 1, 5],   // SPOT-6/7
]

export const geoSpectralIndices: NodeImpl = (inputs, params, ctx) => {
  const raster = inputs.geotiff
  if (!isGeoRaster(raster)) return {}

  const sensor = Math.round(Number(params.sensor ?? 0))
  const preset = SENSOR_PRESETS[sensor]
  const [nirB, redB, greenB, blueB, swirB] = sensor > 0 && preset
    ? preset
    : [
        Math.round(Number(params.nir_band ?? 4)),
        Math.round(Number(params.red_band ?? 1)),
        Math.round(Number(params.green_band ?? 2)),
        Math.round(Number(params.blue_band ?? 3)),
        Math.round(Number(params.swir_band ?? 5)),
      ]

  const NIR = band(raster, nirB)
  const RED = band(raster, redB)
  const GREEN = band(raster, greenB)
  const BLUE = band(raster, blueB)
  const SWIR = band(raster, swirB)

  const low = Number(params.clamp_min ?? -1)
  const high = Number(params.clamp_max ?? 1)
  const guard = params.guard_invalid ? Number(params.valid_min ?? -0.002) : null
  const clamp = (values: Float32Array) => {
    for (let i = 0; i < values.length; i++) {
      values[i] = values[i] < low ? low : values[i] > high ? high : values[i]
    }
    return values
  }

  const result: Record<string, unknown> = {}
  const stack: Float32Array[] = []
  const labels: string[] = []
  const add = (key: string, values: Float32Array) => {
    stack.push(values)
    labels.push(key.toUpperCase())
    result[key] = colorize(ctx.cv, ctx, values, raster.width, raster.height, low, high, params.colormap)
  }

  if (params.ndvi !== false) add('ndvi', normalisedIndex(NIR, RED, guard))
  if (params.ndwi) add('ndwi', normalisedIndex(GREEN, NIR, guard))
  if (params.evi) {
    const evi = new Float32Array(NIR.length)
    for (let i = 0; i < evi.length; i++) {
      evi[i] = (2.5 * (NIR[i] - RED[i])) / (NIR[i] + 6 * RED[i] - 7.5 * BLUE[i] + 1 + 1e-8)
    }
    add('evi', clamp(evi))
  }
  if (params.mndwi) add('mndwi', normalisedIndex(GREEN, SWIR, guard))
  if (params.nbr) add('nbr', normalisedIndex(NIR, SWIR, guard))
  if (params.bsi) {
    const left = new Float32Array(NIR.length)
    const right = new Float32Array(NIR.length)
    for (let i = 0; i < left.length; i++) {
      left[i] = SWIR[i] + RED[i]
      right[i] = NIR[i] + BLUE[i]
    }
    add('bsi', normalisedIndex(left, right))
  }

  for (const [enable, labelKey, exprKey, outKey] of [
    ['expr1_enable', 'expr1_label', 'expr1', 'custom1'],
    ['expr2_enable', 'expr2_label', 'expr2', 'custom2'],
  ] as const) {
    if (!params[enable]) continue
    const expression = String(params[exprKey] ?? '').trim()
    if (!expression) continue
    const compute = compileExpression(expression, raster.count, true)
    if (!compute) { ctx.emit('error', `Spectral Indices: ${outKey} expression invalide`); continue }
    const values = new Float32Array(raster.width * raster.height)
    const scratch = new Array(raster.count)
    const alias = new Array(ALIASES.length)
    try {
      for (let p = 0; p < values.length; p++) {
        for (let b = 0; b < raster.count; b++) scratch[b] = raster.bands[b][p]
        alias[0] = NIR[p]; alias[1] = RED[p]; alias[2] = GREEN[p]; alias[3] = BLUE[p]; alias[4] = SWIR[p]
        values[p] = compute(scratch, alias)
      }
    } catch (error) {
      ctx.emit('error', `Spectral Indices: ${outKey} — ${error instanceof Error ? error.message : String(error)}`)
      continue
    }
    stack.push(clamp(values))
    labels.push(String(params[labelKey] ?? outKey).trim() || outKey)
    result[outKey] = colorize(ctx.cv, ctx, values, raster.width, raster.height, low, high, params.colormap)
  }

  for (const key of ['ndvi', 'ndwi', 'evi', 'mndwi', 'nbr', 'bsi', 'custom1', 'custom2']) {
    if (!(key in result)) result[key] = null
  }
  if (stack.length === 0) return result

  result.stack = withBands(raster, stack, labels)
  result.main = result[Object.keys(result).find((k) => result[k] && k !== 'stack') ?? 'ndvi']
  return result
}

/* ---------------------------------------------------- raster Gaussian noise */

interface NoiseState { tick: number; running: boolean; lastToggle: number; lastReset: number; output: Record<string, unknown> | null }

export const geoRasterNoise: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const raster = inputs.geotiff
  const key = `${ctx.nodeId}:noise`
  let state = ctx.state.get(key) as NoiseState | undefined
  if (!state) {
    state = { tick: 0, running: true, lastToggle: 0, lastReset: 0, output: null }
    ctx.state.set(key, state)
  }
  if (!isGeoRaster(raster)) return { geotiff: null, preview: null, tick: state.tick }

  // Reset and Start/Stop are triggers: act on the rising edge, as every
  // one-shot control in the port does.
  const reset = Number(params.reset ?? 0) || 0
  if (reset > 0.5 && state.lastReset <= 0.5) {
    state.tick = 0
    state.running = true
    state.output = null
  }
  state.lastReset = reset

  const toggle = Number(params.toggle_run ?? 0) || 0
  if (toggle > 0.5 && state.lastToggle <= 0.5) state.running = !state.running
  state.lastToggle = toggle

  // Once the target number of realisations is drawn the run pauses itself, and
  // the accumulator downstream has by then averaged exactly that many frames.
  const maxTicks = Math.round(Number(params.max_ticks ?? 0)) || 0
  if (maxTicks > 0 && state.tick >= maxTicks) state.running = false
  if (!state.running && state.output) return state.output

  const sigmaAbs = Number(params.sigma_abs ?? 0.01)
  const sigmaRel = Number(params.sigma_rel ?? 0.02)
  const clipNegative = params.clip_negative !== false
  const clipMin = Number(params.clip_min ?? 0)
  const clipMax = Number(params.clip_max ?? 0)
  const baseSeed = Math.round(Number(params.seed ?? -1))
  const correlation = Number(params.spatial_corr_px ?? 0)

  const random = baseSeed >= 0 ? seededRandom(baseSeed + state.tick) : Math.random
  state.tick++

  const pixels = raster.width * raster.height
  const noisy: Float32Array[] = []
  for (let b = 0; b < raster.count; b++) {
    const source = raster.bands[b]
    const noise = gaussianField(pixels, random)

    if (correlation > 0) {
      // Blurring the white-noise field makes neighbours co-vary; renormalising
      // afterwards keeps the per-pixel sigma the caller asked for. Independent
      // noise would understate the ensemble spread and leave P over-confident.
      const field = new cv.Mat(raster.height, raster.width, cv.CV_32F)
      ;(field.data32F as Float32Array).set(noise)
      cv.GaussianBlur(field, field, new cv.Size(0, 0), correlation, correlation, cv.BORDER_DEFAULT)
      const blurred = field.data32F as Float32Array
      let mean = 0
      for (let i = 0; i < pixels; i++) mean += blurred[i]
      mean /= pixels
      let variance = 0
      for (let i = 0; i < pixels; i++) variance += (blurred[i] - mean) ** 2
      const deviation = Math.sqrt(variance / pixels)
      for (let i = 0; i < pixels; i++) noise[i] = deviation > 1e-8 ? blurred[i] / deviation : blurred[i]
      field.delete()
    }

    const out = new Float32Array(pixels)
    for (let i = 0; i < pixels; i++) {
      const sigma = sigmaAbs + sigmaRel * Math.abs(source[i])
      let value = source[i] + noise[i] * sigma
      // An explicit range supersedes the legacy clip-negatives floor, and is
      // only active when max is above min so old graphs keep their behaviour.
      if (clipMax > clipMin) value = value < clipMin ? clipMin : value > clipMax ? clipMax : value
      else if (clipNegative && value < 0) value = 0
      out[i] = value
    }
    noisy.push(out)
  }

  const first = noisy[0]
  const low = percentile(first, 2)
  const high = percentile(first, 98)
  const gray = new cv.Mat(raster.height, raster.width, cv.CV_8U)
  ;(gray.data as Uint8Array).set(stretchToBytes(first, low, high === low ? low + 1 : high))
  const preview = ctx.track(applyColormap(cv, gray, (v: number) => [v, v, v]))
  gray.delete()

  state.output = { geotiff: withBands(raster, noisy, raster.band_names), preview, main: preview, tick: state.tick }
  return state.output
}
