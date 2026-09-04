import type { NodeImpl } from '../types'
import { drawPolyline, toBgr, toGray } from '../cvUtils'
import { applyColormap, hotColor, jetColor, magmaColor, oceanColor, viridisColor } from '../colormaps'

/* ------------------------------------------------------------------ shared */

/** #rrggbb → BGR. */
function hexToBgr(raw: unknown, fallback: [number, number, number]): [number, number, number] {
  let hex = String(raw ?? '').trim().replace(/^#/, '')
  if (hex.length === 3) hex = hex.split('').map((c) => c + c).join('')
  if (!/^[0-9a-fA-F]{6}$/.test(hex)) return fallback
  return [parseInt(hex.slice(4, 6), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(0, 2), 16)]
}

function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0
}

/** Population standard deviation, matching numpy's default ddof=0. */
function stdDev(values: number[]): number {
  if (!values.length) return 0
  const m = mean(values)
  return Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / values.length)
}

function median(values: number[]): number {
  if (!values.length) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

/* ----------------------------------------------------------------- plotter */

// BGR, matching the desktop's literal tuples.
const SERIES_COLOURS: [number, number, number][] = [
  [255, 100, 100],
  [100, 255, 100],
  [100, 100, 255],
  [255, 255, 100],
  [255, 100, 255],
]

/** Reduces any wired value to the single number a time series can plot. */
function toPlottable(value: unknown): number | null {
  if (value === null || value === undefined) return null
  if (typeof value === 'number') return value
  if (Array.isArray(value)) {
    if (!value.length) return 0
    const first = value[0]
    if (first && typeof first === 'object' && !Array.isArray(first)) {
      // A list of detections: average whichever measurement they carry, and
      // fall back to simply counting them.
      for (const key of ['area', 'scalar', 'value', 'confidence']) {
        if (key in (first as Record<string, unknown>)) {
          return mean(value.map((item) => Number((item as Record<string, unknown>)[key]) || 0))
        }
      }
      return value.length
    }
    const nums = value.map(Number).filter((v) => Number.isFinite(v))
    return nums.length ? mean(nums) : 0
  }
  if (typeof value === 'object') {
    for (const key of ['area', 'scalar', 'value', 'confidence']) {
      const v = (value as Record<string, unknown>)[key]
      if (typeof v === 'number') return v
    }
    return 1
  }
  return 0
}

export const sciPlotter: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const bufferSize = Math.max(10, Math.round(Number(params.buffer_size) || 200))
  const w = Math.max(100, Math.round(Number(params.width) || 640))
  const h = Math.max(100, Math.round(Number(params.height) || 360))

  let history = ctx.state.get(ctx.nodeId) as Map<string, number[]> | undefined
  if (!history) {
    history = new Map()
    ctx.state.set(ctx.nodeId, history)
  }

  const series = new Map<string, unknown>()
  for (const [port, value] of Object.entries(inputs)) {
    if (port === 'raw_frame' || value === null || value === undefined) continue
    series.set(port, value)
  }

  // Drop the history of any series whose link was removed.
  for (const key of [...history.keys()]) if (!series.has(key)) history.delete(key)

  for (const [key, value] of series) {
    const v = toPlottable(value)
    if (v === null || !Number.isFinite(v)) continue
    const track = history.get(key) ?? []
    track.push(v)
    if (track.length > bufferSize) track.splice(0, track.length - bufferSize)
    history.set(key, track)
  }

  const img = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(20, 20, 20, 255)))
  const gridColour = new cv.Scalar(40, 40, 40, 255)
  for (let i = 1; i < 4; i++) {
    const y = Math.trunc((h * i) / 4)
    cv.line(img, new cv.Point(0, y), new cv.Point(w, y), gridColour, 1)
  }

  let minY = Number(params.min_y) || 0
  let maxY = Number(params.max_y) ?? 100
  if (params.auto_scale !== false) {
    const all = [...history.values()].flat()
    if (all.length) {
      minY = Math.min(...all)
      maxY = Math.max(...all)
      if (maxY === minY) maxY += 1
    }
  }
  const range = maxY - minY || 1

  const out: Record<string, unknown> = { main: img }
  let index = 0
  for (const [key, track] of history) {
    const colour = SERIES_COLOURS[index % SERIES_COLOURS.length]
    const scalar = new cv.Scalar(colour[0], colour[1], colour[2], 255)
    if (track.length >= 2) {
      const points = track.map((value, j) => ({
        x: bufferSize > 1 ? Math.trunc((j * w) / (bufferSize - 1)) : 0,
        y: Math.max(0, Math.min(h - 1, Math.trunc(h - ((value - minY) / range) * h))),
      }))
      drawPolyline(cv, img, points, false, scalar, 2)
    }
    if (track.length) {
      cv.putText(img, `${key}: ${track[track.length - 1].toFixed(2)}`, new cv.Point(10, 20 + index * 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, scalar, 1, cv.LINE_AA)
      // The node's own chart component reads these per-port values, so the
      // latest sample of every series has to travel as an output too.
      out[key] = track[track.length - 1]
    }
    index++
  }

  return out
}

/* ------------------------------------------------------------------- stats */

export const sciStats: NodeImpl = (inputs) => {
  const zero = { mean: 0, median: 0, std: 0, min: 0, max: 0 }
  const data = inputs.data_list
  if (!Array.isArray(data) || !data.length) return zero

  const nums: number[] = []
  for (const item of data) {
    if (typeof item === 'number') nums.push(item)
    else if (item && typeof item === 'object') {
      const d = item as Record<string, unknown>
      if (typeof d.area === 'number') nums.push(d.area)
      else if (typeof d.scalar === 'number') nums.push(d.scalar)
    }
  }
  if (!nums.length) return zero

  return { mean: mean(nums), median: median(nums), std: stdDev(nums), min: Math.min(...nums), max: Math.max(...nums) }
}

/* ----------------------------------------------------------------- heatmap */

const HEATMAP_COLOURS = [jetColor, hotColor, magmaColor, viridisColor, oceanColor]

interface HeatmapState {
  buffer: Float32Array
  res: number
  lastReset: number
}

export const sciHeatmap: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = inputs.image as any
  const points = inputs.points

  const res = Math.max(16, Math.round(Number(params.res) || 64))
  const decay = Number(params.decay) ?? 0.01
  const intensity = Number(params.intensity) || 1
  const reset = Number(params.reset) ? 1 : 0

  let state = ctx.state.get(ctx.nodeId) as HeatmapState | undefined
  if (!state || state.res !== res || reset === 1) {
    state = { buffer: new Float32Array(res * res), res, lastReset: reset }
    ctx.state.set(ctx.nodeId, state)
  }
  state.lastReset = reset

  const buffer = state.buffer
  // Exponential fade, so old activity thins out instead of saturating forever.
  if (decay > 0) {
    const keep = 1 - decay
    for (let i = 0; i < buffer.length; i++) buffer[i] *= keep
  }

  if (points) {
    if (typeof (points as any).cols === 'number' && typeof (points as any).delete === 'function') {
      // A dense map: an optical-flow field contributes its magnitude, a
      // single-channel map its own values.
      const mat = points as any
      const small = new cv.Mat()
      if (mat.channels() === 2) {
        const planes = new cv.MatVector()
        cv.split(mat, planes)
        const magnitude = new cv.Mat()
        const angle = new cv.Mat()
        cv.cartToPolar(planes.get(0), planes.get(1), magnitude, angle)
        cv.resize(magnitude, small, new cv.Size(res, res), 0, 0, cv.INTER_LINEAR)
        const d = small.data32F
        for (let i = 0; i < buffer.length; i++) buffer[i] += d[i] * intensity * 0.1
        magnitude.delete()
        angle.delete()
        planes.delete()
      } else if (mat.channels() === 1) {
        cv.resize(mat, small, new cv.Size(res, res), 0, 0, cv.INTER_LINEAR)
        if (small.depth() === cv.CV_32F) {
          const d = small.data32F
          for (let i = 0; i < buffer.length; i++) buffer[i] += d[i] * intensity
        } else {
          const d = small.data
          for (let i = 0; i < buffer.length; i++) buffer[i] += (d[i] / 255) * intensity
        }
      }
      small.delete()
    } else {
      const list = Array.isArray(points) ? points : [points]
      for (const entry of list) {
        if (!entry || typeof entry !== 'object') continue
        const p = entry as Record<string, any>
        let x: number | undefined
        let y: number | undefined
        if (p.center && typeof p.center === 'object') {
          x = p.center.x
          y = p.center.y
        } else if (typeof p.x === 'number' && typeof p.y === 'number') {
          x = p.x
          y = p.y
        } else if (typeof p.xmin === 'number') {
          x = p.xmin + (Number(p.width) || 0) / 2
          y = (Number(p.ymin) || 0) + (Number(p.height) || 0) / 2
        }
        if (x === undefined || y === undefined) continue
        const ix = Math.trunc(x * res)
        const iy = Math.trunc(y * res)
        if (ix >= 0 && ix < res && iy >= 0 && iy < res) buffer[iy * res + ix] += intensity
      }
    }
  }

  let peak = 0
  for (let i = 0; i < buffer.length; i++) if (buffer[i] > peak) peak = buffer[i]
  const scaled = new cv.Mat(res, res, cv.CV_8U)
  const bytes = scaled.data
  for (let i = 0; i < buffer.length; i++) bytes[i] = peak > 0 ? Math.round((buffer[i] / peak) * 255) : 0

  let blurRadius = Math.round(Number(params.blur) ?? 5)
  let smoothed = scaled
  if (blurRadius > 0) {
    // A Gaussian kernel has to be odd-sized to stay centred.
    if (blurRadius % 2 === 0) blurRadius += 1
    smoothed = new cv.Mat()
    cv.GaussianBlur(scaled, smoothed, new cv.Size(blurRadius, blurRadius), 0, 0, cv.BORDER_DEFAULT)
    scaled.delete()
  }

  const choice = Math.min(HEATMAP_COLOURS.length - 1, Math.max(0, Math.round(Number(params.colormap) || 0)))
  const heat = applyColormap(cv, smoothed, HEATMAP_COLOURS[choice])
  smoothed.delete()
  return finishHeatmap(cv, ctx, heat, image, params)
}

/** Scales the heat map onto the frame and blends, or returns it bare. */
function finishHeatmap(cv: any, ctx: any, heat: any, image: any, params: Record<string, any>): Record<string, unknown> {
  if (!image) return { main: ctx.track(heat) }

  const resized = new cv.Mat()
  cv.resize(heat, resized, new cv.Size(image.cols, image.rows), 0, 0, cv.INTER_LINEAR)
  heat.delete()

  const background = toBgr(cv, image)
  const alpha = Number(params.blend) ?? 0.7
  const out = ctx.track(new cv.Mat())
  cv.addWeighted(background, 1 - alpha, resized, alpha, 0, out)
  background.delete()
  resized.delete()
  return { main: out }
}

/* -------------------------------------------------------- list aggregator */

export const featListAggregator: NodeImpl = (inputs, params) => {
  const items = Array.isArray(inputs.items) ? inputs.items : []
  const key = String(params.key ?? 'radius')
  const prefix = String(params.prefix ?? 'Item')
  const capitalised = key.charAt(0).toUpperCase() + key.slice(1)
  const isArea = !!params.is_area
  const factorIn = inputs.px_per_unit
  const factor = Number(factorIn) || 1

  let unit = String(params.unit ?? 'px')
  // A calibration link with the unit left at px means the user forgot to set it.
  if (factorIn !== null && factorIn !== undefined && unit === 'px') unit = 'mm'

  const values: number[] = []
  for (const item of items) {
    if (typeof item === 'number') values.push(item)
    else if (item && typeof item === 'object' && key in (item as Record<string, unknown>)) {
      const v = Number((item as Record<string, unknown>)[key])
      if (Number.isFinite(v)) values.push(v)
    }
  }

  if (!values.length) {
    return { stats: { [`${prefix} Count`]: 0, [`Avg ${capitalised}`]: 0 } }
  }

  let scaled = values
  if (factor > 0 && factor !== 1) {
    // An area scales with the square of a linear calibration factor.
    const divisor = isArea ? factor * factor : factor
    scaled = values.map((v) => v / divisor)
  } else {
    unit = 'px'
  }

  const round4 = (v: number) => Math.round(v * 1e4) / 1e4
  return {
    stats: {
      [`${prefix} Count`]: values.length,
      [`Avg ${capitalised} (${unit})`]: round4(mean(scaled)),
      [`Std ${capitalised}`]: round4(stdDev(scaled)),
      [`Min ${capitalised}`]: round4(Math.min(...scaled)),
      [`Max ${capitalised}`]: round4(Math.max(...scaled)),
      [`Total ${capitalised}`]: Math.round(scaled.reduce((a, b) => a + b, 0) * 100) / 100,
    },
  }
}

/* ---------------------------------------------------------- k-means on a key */

export const sciKmeansList: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const items = Array.isArray(inputs.items) ? inputs.items : []
  const k = Math.max(2, Math.round(Number(params.k) || 3))
  const key = String(params.key ?? 'radius')

  const availableKeys = [
    ...new Set(
      items.flatMap((item) =>
        item && typeof item === 'object'
          ? Object.entries(item as Record<string, unknown>).filter(([, v]) => typeof v === 'number').map(([k2]) => k2)
          : []
      )
    ),
  ].sort()

  if (items.length < k) return { items, stats: { error: 'Not enough items' }, _available_keys: availableKeys }

  const valid = items.filter((item) => item && typeof item === 'object' && key in (item as Record<string, unknown>))
  if (!valid.length) return { items, stats: { error: `No item has key '${key}'` }, _available_keys: availableKeys }

  const samples = cv.matFromArray(valid.length, 1, cv.CV_32F, valid.map((item) => Number((item as Record<string, unknown>)[key]) || 0))
  const labels = new cv.Mat()
  const centers = new cv.Mat()
  // EPS | MAX_ITER is 3; the named constants are not exported by this build.
  cv.kmeans(samples, k, labels, new cv.TermCriteria(3, 100, 0.1), 10, cv.KMEANS_PP_CENTERS, centers)

  const centreValues: number[] = []
  for (let i = 0; i < centers.rows; i++) centreValues.push(centers.data32F[i])

  // Re-label so cluster 0 is the smallest centroid: the raw ids depend on the
  // seeding, and downstream code needs an ordering it can rely on.
  const order = centreValues.map((_, i) => i).sort((a, b) => centreValues[a] - centreValues[b])
  const rank = new Map(order.map((original, position) => [original, position]))
  const sortedCentres = order.map((i) => centreValues[i])

  const counts = new Array<number>(k).fill(0)
  const tagged = valid.map((item, i) => {
    const cluster = rank.get(labels.data32S[i]) ?? 0
    counts[cluster] += 1
    return { ...(item as Record<string, unknown>), cluster_id: cluster }
  })

  const stats: Record<string, unknown> = {}
  for (let i = 0; i < k; i++) {
    stats[`group_${i}`] = { count: counts[i], center: Math.round(sortedCentres[i] * 1e4) / 1e4 }
  }

  samples.delete()
  labels.delete()
  centers.delete()

  return { items: tagged, stats, _available_keys: availableKeys }
}

/* ------------------------------------------------------- robust pixel stats */

export const sciRobustStats: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  if (!image) return { main: null, mean: null, median: null, std: null, mad: null, count: null }

  const iw = image.cols
  const ih = image.rows
  let rx = Math.trunc(((Number(params.x) || 0) / 100) * iw)
  let ry = Math.trunc(((Number(params.y) || 0) / 100) * ih)
  let rw = Math.max(1, Math.trunc(((Number(params.w) ?? 100) / 100) * iw))
  let rh = Math.max(1, Math.trunc(((Number(params.h) ?? 100) / 100) * ih))
  rx = Math.min(rx, iw - 1)
  ry = Math.min(ry, ih - 1)
  rw = Math.min(rw, iw - rx)
  rh = Math.min(rh, ih - ry)

  const colour = toBgr(cv, image)
  const roi = colour.roi(new cv.Rect(rx, ry, rw, rh))

  const channel = Math.round(Number(params.channel) || 0)
  const values: number[] = []
  if (image.channels() === 1 || channel === 0) {
    const gray = toGray(cv, roi)
    for (let i = 0; i < gray.data.length; i++) values.push(gray.data[i])
    gray.delete()
  } else {
    // Red is channel 2 in BGR, green 1, blue 0.
    const offset = channel === 1 ? 2 : channel === 2 ? 1 : 0
    const data = roi.data
    for (let i = offset; i < data.length; i += 3) values.push(data[i])
  }

  const meanV = mean(values)
  const stdV = stdDev(values)
  const medianV = median(values)
  // MAD: the median of the absolute deviations from the median, which a few
  // extreme pixels cannot drag around the way they drag the standard deviation.
  const madV = median(values.map((v) => Math.abs(v - medianV)))

  const [b, g, r] = hexToBgr(params.color, [160, 255, 0])
  const out = ctx.track(colour)
  const scalar = new cv.Scalar(b, g, r, 255)
  cv.rectangle(out, new cv.Point(rx, ry), new cv.Point(rx + rw, ry + rh), scalar, 1)

  if (params.show_stats !== false) {
    const lines = [`mean=${meanV.toFixed(1)}  std=${stdV.toFixed(1)}`, `median=${medianV.toFixed(1)}  MAD=${madV.toFixed(1)}`]
    lines.forEach((line, i) => {
      cv.putText(out, line, new cv.Point(rx + 3, ry + 14 + i * 14), cv.FONT_HERSHEY_SIMPLEX, 0.38, scalar, 1, cv.LINE_AA)
    })
  }

  roi.delete()
  return { main: out, mean: meanV, median: medianV, std: stdV, mad: madV, count: values.length }
}

/* -------------------------------------------------------------- compare grid */

const GRID_RESERVED = new Set(['raw_frame', 'image', 'data', 'in', 'value', 'main'])

/** Dynamic port keys look like "0_x7k2"; unindexed keys sort last. */
function portIndex(key: string): number {
  const n = Number.parseInt(key.split('_')[0], 10)
  return Number.isFinite(n) ? n : 9999
}

export const vizCompareGrid: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const panelPx = Math.max(96, Math.round(Number(params.panel_px) || 320))
  const colsParam = Math.round(Number(params.cols) || 0)
  const showLabels = params.show_labels !== false

  const keys = Object.keys(inputs)
    .filter((k) => !GRID_RESERVED.has(k) && inputs[k] && typeof (inputs[k] as any).cols === 'number')
    .sort((a, b) => portIndex(a) - portIndex(b))
  if (!keys.length) return { main: null, n_panels: 0 }

  const panels = keys.map((key, i) => {
    const src = inputs[key] as any
    const bgr = toBgr(cv, src)
    const scale = panelPx / Math.max(bgr.rows, bgr.cols)
    const pw = Math.max(1, Math.round(bgr.cols * scale))
    const ph = Math.max(1, Math.round(bgr.rows * scale))
    const small = new cv.Mat()
    cv.resize(bgr, small, new cv.Size(pw, ph), 0, 0, cv.INTER_AREA)
    bgr.delete()

    cv.rectangle(small, new cv.Point(0, 0), new cv.Point(pw - 1, ph - 1), new cv.Scalar(70, 70, 70, 255), 1)
    if (showLabels) {
      const label = String.fromCharCode(65 + i)
      const at = new cv.Point(6, 22)
      cv.putText(small, label, at, cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(0, 0, 0, 255), 3, cv.LINE_AA)
      cv.putText(small, label, at, cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(255, 255, 255, 255), 1, cv.LINE_AA)
    }
    return small
  })

  const cols = colsParam > 0 ? colsParam : Math.ceil(Math.sqrt(panels.length))
  const rows = Math.ceil(panels.length / cols)
  const cellH = Math.max(...panels.map((p) => p.rows))
  const cellW = Math.max(...panels.map((p) => p.cols))
  const gap = 6

  const canvas = ctx.track(
    new cv.Mat(rows * cellH + (rows + 1) * gap, cols * cellW + (cols + 1) * gap, cv.CV_8UC3, new cv.Scalar(18, 18, 18, 255))
  )
  panels.forEach((panel, i) => {
    const y = gap + Math.floor(i / cols) * (cellH + gap)
    const x = gap + (i % cols) * (cellW + gap)
    panel.copyTo(canvas.roi(new cv.Rect(x, y, panel.cols, panel.rows)))
    panel.delete()
  })

  return { main: canvas, n_panels: panels.length }
}

/* -------------------------------------------------------------- phase space */

/** One phase-space cell: grid, zero axes, fading trail, head glow and labels. */
function renderPhaseCell(cv: any, points: [number, number][], w: number, h: number, bgr: [number, number, number], title = '', xLabel = 'X', yLabel = 'Y'): any {
  const img = new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255))
  const font = cv.FONT_HERSHEY_SIMPLEX

  if (points.length < 2) {
    cv.putText(img, title, new cv.Point(8, 22), font, 0.46, new cv.Scalar(90, 90, 90, 255), 1, cv.LINE_AA)
    cv.putText(img, 'waiting…', new cv.Point(8, 44), font, 0.38, new cv.Scalar(60, 60, 60, 255), 1, cv.LINE_AA)
    return img
  }

  const xs = points.map((p) => p[0])
  const ys = points.map((p) => p[1])
  let xMin = Math.min(...xs)
  let xMax = Math.max(...xs)
  let yMin = Math.min(...ys)
  let yMax = Math.max(...ys)
  // A degenerate range still needs a window, or every point lands on one pixel.
  const xPad = (xMax - xMin) * 0.12 || 0.05
  const yPad = (yMax - yMin) * 0.12 || 0.05
  xMin -= xPad
  xMax += xPad
  yMin -= yPad
  yMax += yPad
  const dx = xMax - xMin
  const dy = yMax - yMin

  const M = 30
  const toPx = (xv: number, yv: number) => {
    const px = M + Math.trunc(((xv - xMin) / dx) * (w - 2 * M))
    const py = h - M - Math.trunc(((yv - yMin) / dy) * (h - 2 * M))
    return new cv.Point(Math.max(0, Math.min(w - 1, px)), Math.max(0, Math.min(h - 1, py)))
  }

  const grid = new cv.Scalar(20, 20, 20, 255)
  for (let i = 0; i < 5; i++) {
    const gx = M + Math.trunc(((w - 2 * M) * i) / 4)
    const gy = M + Math.trunc(((h - 2 * M) * i) / 4)
    cv.line(img, new cv.Point(gx, M), new cv.Point(gx, h - M), grid, 1)
    cv.line(img, new cv.Point(M, gy), new cv.Point(w - M, gy), grid, 1)
  }

  const axis = new cv.Scalar(45, 45, 45, 255)
  if (xMin <= 0 && 0 <= xMax) {
    const p = toPx(0, yMin)
    cv.line(img, new cv.Point(p.x, M), new cv.Point(p.x, h - M), axis, 1)
  }
  if (yMin <= 0 && 0 <= yMax) {
    const p = toPx(xMin, 0)
    cv.line(img, new cv.Point(M, p.y), new cv.Point(w - M, p.y), axis, 1)
  }

  // Trail fading toward the tail: brightness rises as t^1.5 along the buffer.
  const n = points.length
  for (let i = 1; i < n; i++) {
    const t = (i / n) ** 1.5
    const colour = new cv.Scalar(bgr[0] * t, bgr[1] * t, bgr[2] * t, 255)
    cv.line(img, toPx(points[i - 1][0], points[i - 1][1]), toPx(points[i][0], points[i][1]), colour, i > n - 5 ? 2 : 1, cv.LINE_AA)
  }

  const head = toPx(points[n - 1][0], points[n - 1][1])
  for (const [radius, alpha] of [[10, 0.07], [7, 0.18], [4, 0.45]] as [number, number][]) {
    cv.circle(img, head, radius, new cv.Scalar(bgr[0] * alpha, bgr[1] * alpha, bgr[2] * alpha, 255), -1, cv.LINE_AA)
  }
  cv.circle(img, head, 3, new cv.Scalar(255, 255, 255, 255), -1, cv.LINE_AA)
  cv.circle(img, head, 2, new cv.Scalar(bgr[0], bgr[1], bgr[2], 255), -1, cv.LINE_AA)

  cv.rectangle(img, new cv.Point(M, M), new cv.Point(w - M, h - M), new cv.Scalar(bgr[0] * 0.3, bgr[1] * 0.3, bgr[2] * 0.3, 255), 1)

  const labelColour = new cv.Scalar(bgr[0] * 0.72, bgr[1] * 0.72, bgr[2] * 0.72, 255)
  cv.putText(img, title, new cv.Point(M, M - 7), font, 0.46, new cv.Scalar(195, 195, 195, 255), 1, cv.LINE_AA)
  cv.putText(img, xLabel, new cv.Point(Math.trunc(w / 2) - 6, h - 4), font, 0.36, labelColour, 1, cv.LINE_AA)
  cv.putText(img, yLabel, new cv.Point(2, Math.trunc(h / 2) + 5), font, 0.36, labelColour, 1, cv.LINE_AA)

  const dim = new cv.Scalar(75, 75, 75, 255)
  cv.putText(img, `(${points[n - 1][0].toFixed(3)}, ${points[n - 1][1].toFixed(3)})`, new cv.Point(M + 3, h - M - 5), font, 0.31, dim, 1, cv.LINE_AA)

  const tick = new cv.Scalar(55, 55, 55, 255)
  cv.putText(img, xMin.toFixed(2), new cv.Point(M, h - M + 13), font, 0.28, tick, 1, cv.LINE_AA)
  cv.putText(img, xMax.toFixed(2), new cv.Point(w - M - 22, h - M + 13), font, 0.28, tick, 1, cv.LINE_AA)
  cv.putText(img, yMax.toFixed(2), new cv.Point(1, M + 8), font, 0.28, tick, 1, cv.LINE_AA)
  cv.putText(img, yMin.toFixed(2), new cv.Point(1, h - M - 1), font, 0.28, tick, 1, cv.LINE_AA)

  return img
}

export const phaseSpace: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const w = Math.max(150, Math.round(Number(params.width) || 380))
  const h = Math.max(150, Math.round(Number(params.height) || 380))

  const xv = Number(inputs.x)
  const yv = Number(inputs.y)
  if (inputs.x === null || inputs.x === undefined || inputs.y === null || inputs.y === undefined || !Number.isFinite(xv) || !Number.isFinite(yv)) {
    return { main: ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255))) }
  }

  let buffer = ctx.state.get(ctx.nodeId) as [number, number][] | undefined
  if (!buffer) {
    buffer = []
    ctx.state.set(ctx.nodeId, buffer)
  }
  buffer.push([xv, yv])
  const trailLength = Math.max(10, Math.round(Number(params.trail_len) || 300))
  if (buffer.length > trailLength) buffer.splice(0, buffer.length - trailLength)

  const bgr = hexToBgr(params.trail_color, [204, 255, 0])
  const img = ctx.track(
    renderPhaseCell(cv, buffer, w, h, bgr, String(params.title ?? 'Phase Space'), String(params.x_label ?? 'X'), String(params.y_label ?? 'Y'))
  )
  return { main: img }
}

// [xPort, yPort, title, xLabel, yLabel, BGR]
const IMU_PLOTS: [string, string, string, string, string, [number, number, number]][] = [
  ['ax', 'ay', 'Accel  XY', 'ax', 'ay', [204, 255, 0]],
  ['ax', 'az', 'Accel  XZ', 'ax', 'az', [255, 170, 0]],
  ['ay', 'az', 'Accel  YZ', 'ay', 'az', [255, 68, 170]],
  ['gx', 'gy', 'Gyro   XY', 'gx', 'gy', [68, 68, 255]],
  ['gx', 'gz', 'Gyro   XZ', 'gx', 'gz', [0, 136, 255]],
  ['gy', 'gz', 'Gyro   YZ', 'gy', 'gz', [0, 238, 255]],
]

export const imuPhaseDashboard: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const trailLength = Math.max(10, Math.round(Number(params.trail_len) || 300))
  const W = Math.max(450, Math.round(Number(params.width) || 840))
  const H = Math.max(300, Math.round(Number(params.height) || 597))
  const headerH = 36
  const cellW = Math.trunc(W / 3)
  const cellH = Math.max(1, Math.trunc((H - headerH - 1) / 2))

  let buffers = ctx.state.get(ctx.nodeId) as Map<string, [number, number][]> | undefined
  if (!buffers) {
    buffers = new Map(IMU_PLOTS.map(([x, y]) => [`${x}${y}`, [] as [number, number][]]))
    ctx.state.set(ctx.nodeId, buffers)
  }

  const values: Record<string, number> = {}
  for (const port of ['ax', 'ay', 'az', 'gx', 'gy', 'gz']) {
    const v = Number(inputs[port])
    values[port] = Number.isFinite(v) ? v : 0
  }

  for (const [xk, yk] of IMU_PLOTS) {
    const key = `${xk}${yk}`
    const buffer = buffers.get(key)!
    buffer.push([values[xk], values[yk]])
    if (buffer.length > trailLength) buffer.splice(0, buffer.length - trailLength)
  }

  // Header, then two rows of three cells separated by a single dark line.
  const out = ctx.track(new cv.Mat(headerH + cellH * 2 + 1, W, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))
  cv.line(out, new cv.Point(0, headerH - 1), new cv.Point(W, headerH - 1), new cv.Scalar(42, 42, 42, 255), 1)
  cv.putText(out, String(params.title ?? 'IMU Phase Space'), new cv.Point(12, 24), cv.FONT_HERSHEY_SIMPLEX, 0.56, new cv.Scalar(215, 215, 215, 255), 1, cv.LINE_AA)

  IMU_PLOTS.forEach(([xk, yk, title, xLabel, yLabel, bgr], i) => {
    const cell = renderPhaseCell(cv, buffers!.get(`${xk}${yk}`)!, cellW, cellH, bgr, title, xLabel, yLabel)
    const row = Math.floor(i / 3)
    const y = headerH + row * (cellH + (row > 0 ? 1 : 0))
    cell.copyTo(out.roi(new cv.Rect((i % 3) * cellW, y, cellW, cellH)))
    cell.delete()
  })

  const sepY = headerH + cellH
  cv.line(out, new cv.Point(0, sepY), new cv.Point(W, sepY), new cv.Scalar(30, 30, 30, 255), 1)

  return { main: out }
}
