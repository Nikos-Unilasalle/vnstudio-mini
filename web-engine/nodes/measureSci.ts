/**
 * The `measure` family: region analysis on a label map, curve tracing, axis
 * calibration, particle export, mean-shift segmentation and the visual size gate.
 */
import type { NodeImpl, RunContext } from '../types'
import { makeDf, pyRound } from '../dataframe'
import { pyrMeanShiftFiltering } from '../meanShift'
import { buildZip, ZipEntry } from '../zip'
import { downloadFile } from '../../shims/vfs'
import { encodeImage } from './io'

/** `#rgb` or `#rrggbb` to a BGR triple, defaulting to green as the desktop does. */
function hexToBgr(raw: unknown, fallback: [number, number, number] = [0, 255, 0]): [number, number, number] {
  let hex = String(raw ?? '').trim()
  if (!hex.startsWith('#')) hex = '#' + hex
  if (hex.length === 4) hex = '#' + hex[1] + hex[1] + hex[2] + hex[2] + hex[3] + hex[3]
  if (hex.length !== 7) return fallback
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return [r, g, b].some(Number.isNaN) ? fallback : [b, g, r]
}

/** An int32 label map's values as a plain array, whatever depth the Mat carries. */
function labelData(cv: any, markers: any): { data: Int32Array; width: number; height: number } {
  const width = markers.cols
  const height = markers.rows
  if (markers.type() === cv.CV_32SC1) return { data: new Int32Array(markers.data32S), width, height }
  const converted = new cv.Mat()
  const single = markers.channels() > 1 ? new cv.Mat() : markers
  if (markers.channels() > 1) cv.extractChannel(markers, single, 0)
  single.convertTo(converted, cv.CV_32S)
  const data = new Int32Array(converted.data32S)
  converted.delete()
  if (single !== markers) single.delete()
  return { data, width, height }
}

function toBgr8(cv: any, image: any): any {
  const out = new cv.Mat()
  const eight = new cv.Mat()
  if (image.depth() !== cv.CV_8U) {
    // Float generators emit 0..1; anything else is already in 0..255.
    const range = new cv.Mat()
    image.convertTo(range, cv.CV_8U, 255)
    range.copyTo(eight)
    range.delete()
  } else {
    image.copyTo(eight)
  }
  if (eight.channels() === 1) cv.cvtColor(eight, out, cv.COLOR_GRAY2BGR)
  else if (eight.channels() === 4) cv.cvtColor(eight, out, cv.COLOR_BGRA2BGR)
  else eight.copyTo(out)
  eight.delete()
  return out
}

/* ------------------------------------------------------------ region analysis */

export const sciMarkerAnalysis: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const markers = inputs.markers ?? inputs.labels_map
  const image = (inputs.image ?? inputs.main) as any
  if (!markers || typeof (markers as any).cols !== 'number') {
    return { main: image ?? null, data_list: [], count: 0 }
  }

  const { data, width, height } = labelData(cv, markers)
  const out = image && typeof image.cols === 'number' ? ctx.track(toBgr8(cv, image)) : ctx.track(new cv.Mat(height, width, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))

  const showLabels = Math.round(Number(params.show_labels ?? 1)) === 1
  const showPoints = Math.round(Number(params.show_points ?? 1)) === 1
  const fontScale = Number(params.font_scale) || 0.6
  const thickness = Math.max(1, Math.round(Number(params.thickness) || 1))
  const relative = Math.round(Number(params.coord_type) || 0) === 0
  const colourHex = String(params.text_color ?? '#00FF00')
  const [b, g, r] = hexToBgr(colourHex)
  const colour = new cv.Scalar(b, g, r, 255)

  // One pass over the map accumulates each label's zeroth and first moments,
  // which for a binary region are exactly its area and centroid sums.
  const area = new Map<number, number>()
  const sumX = new Map<number, number>()
  const sumY = new Map<number, number>()
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const id = data[y * width + x]
      if (id <= 0) continue
      area.set(id, (area.get(id) ?? 0) + 1)
      sumX.set(id, (sumX.get(id) ?? 0) + x)
      sumY.set(id, (sumY.get(id) ?? 0) + y)
    }
  }

  const dataList: Record<string, unknown>[] = []
  for (const id of [...area.keys()].sort((p, q) => p - q)) {
    const count = area.get(id)!
    const cx = sumX.get(id)! / count
    const cy = sumY.get(id)! / count
    dataList.push({
      id,
      label: `#${id}`,
      x: relative ? cx / width : cx,
      y: relative ? cy / height : cy,
      area: count,
      center: { x: cx / width, y: cy / height },
      _type: 'graphics',
      shape: 'point',
      pts: [[cx / width, cy / height]],
      relative: true,
      color: colourHex,
    })
    const px = Math.round(cx)
    const py = Math.round(cy)
    if (showPoints) cv.circle(out, new cv.Point(px, py), 3, colour, -1)
    if (showLabels) cv.putText(out, String(id), new cv.Point(px + 5, py - 5), cv.FONT_HERSHEY_SIMPLEX, fontScale, colour, thickness)
  }

  return { main: out, data_list: dataList, count: dataList.length, display_text: `Islands: ${dataList.length}` }
}

/* ---------------------------------------------------------------- curve trace */

export const sciCurveTrace: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const mask = (inputs.mask ?? inputs.main) as any
  if (!mask || typeof mask.cols !== 'number') return { points: [] }

  const single = mask.channels() > 1 ? new cv.Mat() : mask
  if (mask.channels() > 1) cv.cvtColor(mask, single, cv.COLOR_BGR2GRAY)
  const eight = single.depth() === cv.CV_8U ? single : new cv.Mat()
  if (single.depth() !== cv.CV_8U) single.convertTo(eight, cv.CV_8U, 255)
  const data = eight.data as Uint8Array
  const width = eight.cols
  const height = eight.rows

  const byColumn = Math.round(Number(params.axis) || 0) === 0
  const how = Math.round(Number(params.aggregate) || 0)
  const aggregate = (hits: number[]): number => {
    switch (how) {
      case 1:
        return hits.reduce((a, v) => a + v, 0) / hits.length
      case 2:
        return hits[0]
      case 3:
        return hits[hits.length - 1]
      default: {
        // numpy's median: the mean of the two middle values on an even count.
        const mid = hits.length >> 1
        return hits.length % 2 ? hits[mid] : (hits[mid - 1] + hits[mid]) / 2
      }
    }
  }

  const points: { x: number; y: number }[] = []
  const scanLength = byColumn ? width : height
  const crossLength = byColumn ? height : width
  for (let i = 0; i < scanLength; i++) {
    const hits: number[] = []
    for (let k = 0; k < crossLength; k++) {
      const value = byColumn ? data[k * width + i] : data[i * width + k]
      if (value !== 0) hits.push(k)
    }
    if (hits.length === 0) continue
    const value = aggregate(hits)
    points.push(byColumn ? { x: i, y: value } : { x: value, y: i })
  }

  if (eight !== single) eight.delete()
  if (single !== mask) single.delete()
  return { points }
}

/* ----------------------------------------------------------- axis calibration */

/** Days from 0001-01-01, the ordinal Python's `date.toordinal` returns. */
function toOrdinal(year: number, month: number, day: number): number {
  const shiftedYear = month <= 2 ? year - 1 : year
  const era = Math.floor(shiftedYear / 400)
  const yearOfEra = shiftedYear - era * 400
  const dayOfYear = Math.floor((153 * (month + (month > 2 ? -3 : 9)) + 2) / 5) + day - 1
  const dayOfEra = yearOfEra * 365 + Math.floor(yearOfEra / 4) - Math.floor(yearOfEra / 100) + dayOfYear
  // 0001-01-01 is ordinal 1; the era arithmetic above counts from 0000-03-01.
  return era * 146097 + dayOfEra - 305
}

function fromOrdinal(ordinal: number): { year: number; month: number; day: number } {
  const shifted = ordinal + 305
  const era = Math.floor(shifted / 146097)
  const dayOfEra = shifted - era * 146097
  const yearOfEra = Math.floor((dayOfEra - Math.floor(dayOfEra / 1460) + Math.floor(dayOfEra / 36524) - Math.floor(dayOfEra / 146096)) / 365)
  const dayOfYear = dayOfEra - (365 * yearOfEra + Math.floor(yearOfEra / 4) - Math.floor(yearOfEra / 100))
  const monthPrime = Math.floor((5 * dayOfYear + 2) / 153)
  const day = dayOfYear - Math.floor((153 * monthPrime + 2) / 5) + 1
  const month = monthPrime + (monthPrime < 10 ? 3 : -9)
  const year = era * 400 + yearOfEra + (month <= 2 ? 1 : 0)
  return { year, month, day }
}

const DIRECTIVES: Record<string, { width: number; pad: boolean }> = {
  Y: { width: 4, pad: true },
  y: { width: 2, pad: true },
  m: { width: 2, pad: true },
  d: { width: 2, pad: true },
  j: { width: 3, pad: true },
}

/** Parses a date string against a strftime-style format, the subset a chart axis uses. */
function parseDate(raw: string, format: string): number | null {
  let at = 0
  const parts: Record<string, number> = {}
  for (let f = 0; f < format.length; f++) {
    if (format[f] === '%' && f + 1 < format.length) {
      const directive = format[++f]
      const spec = DIRECTIVES[directive]
      if (!spec) return null
      const slice = raw.slice(at, at + spec.width)
      if (!/^\d+$/.test(slice)) return null
      parts[directive] = Number(slice)
      at += slice.length
    } else {
      if (raw[at] !== format[f]) return null
      at++
    }
  }
  const year = parts.Y ?? (parts.y !== undefined ? (parts.y < 69 ? 2000 + parts.y : 1900 + parts.y) : 1900)
  if (parts.j !== undefined && parts.m === undefined) return toOrdinal(year, 1, 1) + parts.j - 1
  return toOrdinal(year, parts.m ?? 1, parts.d ?? 1)
}

function formatDate(ordinal: number, format: string): string {
  const { year, month, day } = fromOrdinal(Math.round(ordinal))
  const pad = (value: number, width: number) => String(value).padStart(width, '0')
  let out = ''
  for (let f = 0; f < format.length; f++) {
    if (format[f] === '%' && f + 1 < format.length) {
      switch (format[++f]) {
        case 'Y': out += pad(year, 4); break
        case 'y': out += pad(year % 100, 2); break
        case 'm': out += pad(month, 2); break
        case 'd': out += pad(day, 2); break
        case 'j': out += pad(Math.round(ordinal) - toOrdinal(year, 1, 1) + 1, 3); break
        default: out += format[f]
      }
    } else {
      out += format[f]
    }
  }
  return out
}

function interpolate(pixel: number, px1: number, v1: number, px2: number, v2: number): number {
  if (px2 === px1) return v1
  return v1 + ((pixel - px1) / (px2 - px1)) * (v2 - v1)
}

export const sciAxisCalibration: NodeImpl = (inputs, params) => {
  const points = inputs.points
  if (!Array.isArray(points) || points.length === 0) return {}
  const valid = points.filter((p): p is Record<string, unknown> =>
    Boolean(p) && typeof p === 'object' && 'x' in (p as object) && 'y' in (p as object))
  if (valid.length === 0) return {}

  // Calibrating on the reference frame's edges is the point of that input: a
  // traced curve rarely touches both axis limits, so its own extent would
  // stretch the mapping.
  const reference = inputs.reference as any
  let xPixel1: number
  let xPixel2: number
  let yPixel1: number
  let yPixel2: number
  if (reference && typeof reference.cols === 'number') {
    xPixel1 = 0
    xPixel2 = reference.cols - 1
    yPixel1 = 0
    yPixel2 = reference.rows - 1
  } else {
    const xs = valid.map((p) => Number(p.x))
    const ys = valid.map((p) => Number(p.y))
    xPixel1 = Math.min(...xs)
    xPixel2 = Math.max(...xs)
    yPixel1 = Math.min(...ys)
    yPixel2 = Math.max(...ys)
  }

  const isDate = Math.round(Number(params.x_type) || 0) === 1
  const dateFormat = String(params.date_format ?? '%Y%m%d')
  const anchor = (wired: unknown, fallback: unknown): number | null => {
    const raw = String((wired ?? fallback) ?? '').trim()
    if (isDate) return parseDate(raw, dateFormat)
    const numeric = Number(raw.replace(',', '.'))
    return Number.isFinite(numeric) ? numeric : null
  }
  const xValue1 = anchor(inputs.x_value_1, params.x_value_1 ?? '0')
  const xValue2 = anchor(inputs.x_value_2, params.x_value_2 ?? '100')
  if (xValue1 === null || xValue2 === null) return {}

  const yValue1 = Number(params.y_value_1 ?? 1)
  const yValue2 = Number(params.y_value_2 ?? 0)
  const xCol = String(params.x_col ?? 'x') || 'x'
  const yCol = String(params.y_col ?? 'y') || 'y'
  const labelCol = String(params.label_col ?? 'source') || 'source'
  const label = inputs.label

  const columns = [xCol, yCol, ...(label ? [labelCol] : [])]
  const rows = valid.map((p) => {
    const x = interpolate(Number(p.x), xPixel1, xValue1, xPixel2, xValue2)
    const y = interpolate(Number(p.y), yPixel1, yValue1, yPixel2, yValue2)
    const record: Record<string, unknown> = {
      [xCol]: isDate ? formatDate(x, dateFormat) : pyRound(x, 6),
      [yCol]: pyRound(y, 6),
    }
    if (label) record[labelCol] = String(label)
    return record
  })

  return { data: makeDf(columns, rows) }
}

/* -------------------------------------------------------------- mean shift */

export const cvMeanShift: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  if (!image || typeof image.cols === 'undefined') return { main: null }

  const sp = Math.max(1, Math.round(Number(params.spatial_radius) || 10))
  const sr = Math.max(1, Math.round(Number(params.color_radius) || 30))
  const maxLevel = Math.max(0, Math.round(Number(params.max_level ?? 1)))

  const source = toBgr8(cv, image)
  const width = source.cols
  const height = source.rows

  // The filter costs O(sp²) per pixel per iteration, so large frames are
  // filtered small and scaled back, as the desktop node does.
  const MAX_SIDE = 640
  const longest = Math.max(width, height)
  let work = source
  if (longest > MAX_SIDE) {
    const scale = MAX_SIDE / longest
    work = new cv.Mat()
    cv.resize(source, work, new cv.Size(Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))), 0, 0, cv.INTER_AREA)
  }

  const filtered = pyrMeanShiftFiltering(cv, work, sp, sr, maxLevel)
  let result = filtered
  if (work !== source) {
    result = new cv.Mat()
    cv.resize(filtered, result, new cv.Size(width, height), 0, 0, cv.INTER_NEAREST)
    filtered.delete()
    work.delete()
  }
  source.delete()
  ctx.track(result)

  const bytes = result.data as Uint8Array
  const distinct = new Set<number>()
  for (let p = 0; p < bytes.length; p += 3) distinct.add((bytes[p] << 16) | (bytes[p + 1] << 8) | bytes[p + 2])

  return { main: result, data: { sp, sr, n_unique_colors: distinct.size } }
}

/* ---------------------------------------------------------- visual size gate */

const SHAPE_AREA = [
  (d: number) => Math.PI * (d / 2) ** 2,   // circle
  (d: number) => d * d,                    // square
  (d: number) => Math.PI * (d / 2) * (d / 4), // thin rod
]

export const featVisualSizeGate: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const markers = inputs.markers ?? inputs.labels_map
  const image = (inputs.image ?? inputs.main) as any
  if (!markers || typeof (markers as any).cols !== 'number') {
    return { markers_out: null, markers_rej: null, mask_kept: null, mask_rej: null, main: image ?? null, count: 0, ref_area: 0 }
  }

  const source = labelData(cv, markers)
  const width = source.width
  const height = source.height
  let labels = source.data

  const distinct = new Set<number>()
  for (const value of labels) if (value > 0) distinct.add(value)

  // A plain binary mask carries a single label; split it into instances first,
  // otherwise every touching object counts as one enormous region.
  if (distinct.size === 1) {
    const binary = new cv.Mat(height, width, cv.CV_8U)
    const bits = binary.data as Uint8Array
    for (let p = 0; p < labels.length; p++) bits[p] = labels[p] > 0 ? 1 : 0

    const separate = params.auto_separate !== false
    let done = false
    if (separate) {
      const distance = new cv.Mat()
      cv.distanceTransform(binary, distance, cv.DIST_L2, 5)
      const range = cv.minMaxLoc(distance)
      if (range.maxVal > 0) {
        // Distance-transform watershed: the peaks become seeds and the ridges
        // between touching objects become the cuts.
        const percent = (Number(params.separation) || 40) / 100
        const sure = new cv.Mat()
        cv.threshold(distance, sure, range.maxVal * percent, 255, cv.THRESH_BINARY)
        const sure8 = new cv.Mat()
        sure.convertTo(sure8, cv.CV_8U)
        const seeds = new cv.Mat()
        cv.connectedComponents(sure8, seeds)
        const seedData = seeds.data32S as Int32Array
        const sureBits = sure8.data as Uint8Array
        const ws = new cv.Mat(height, width, cv.CV_32S)
        const wsData = ws.data32S as Int32Array
        for (let p = 0; p < wsData.length; p++) {
          wsData[p] = bits[p] > 0 && sureBits[p] === 0 ? 0 : seedData[p] + 1
        }
        const colour = image && typeof image.cols === 'number' ? toBgr8(cv, image) : new cv.Mat(height, width, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255))
        if (colour.rows !== height || colour.cols !== width) cv.resize(colour, colour, new cv.Size(width, height))
        cv.watershed(colour, ws)
        labels = new Int32Array(ws.data32S)
        // Background (1) and the watershed lines (-1) are not objects.
        for (let p = 0; p < labels.length; p++) if (labels[p] <= 1) labels[p] = 0
        colour.delete(); ws.delete(); seeds.delete(); sure8.delete(); sure.delete()
        done = true
      }
      distance.delete()
    }
    if (!done) {
      const components = new cv.Mat()
      cv.connectedComponents(binary, components, 8)
      labels = new Int32Array(components.data32S)
      components.delete()
    }
    binary.delete()
  }

  // The reference line, drawn on the node, gives the expected object size.
  let reference = 0
  let p1: [number, number] | null = null
  let p2: [number, number] | null = null
  let drawn: unknown[] = []
  try {
    const parsed = JSON.parse(String(params.points ?? '[]'))
    if (Array.isArray(parsed)) drawn = parsed
  } catch {
    drawn = []
  }
  if (drawn.length >= 2) {
    const a = drawn[0] as Record<string, unknown>
    const b = drawn[1] as Record<string, unknown>
    p1 = [Number(a.x) * width, Number(a.y) * height]
    p2 = [Number(b.x) * width, Number(b.y) * height]
    const length = Math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if (length > 0) reference = SHAPE_AREA[Math.round(Number(params.shape) || 0)]?.(length) ?? 0
  }

  const tolerance = Number(params.tolerance ?? 20)
  const minSize = Math.max(1, Math.round(Number(params.min_size) || 20))
  const remap = params.remap_ids !== false

  let minArea = 0
  let maxArea = Infinity
  if (reference > 0 && tolerance < 100) {
    // Symmetric in log space: 50% gives [ref/2, 2*ref], 90% gives [ref/10, 10*ref].
    const factor = 1 / (1 - tolerance / 100)
    minArea = reference / factor
    maxArea = reference * factor
  }
  minArea = Math.max(minArea, minSize)

  const areas = new Map<number, number>()
  for (const value of labels) if (value > 0) areas.set(value, (areas.get(value) ?? 0) + 1)

  const keptId = new Map<number, number>()
  const rejectedId = new Map<number, number>()
  const surviving: number[] = []
  let nextKept = 1
  let nextRejected = 1
  for (const id of [...areas.keys()].sort((a, b) => a - b)) {
    const area = areas.get(id)!
    if (area < minSize) continue   // silent noise discard
    surviving.push(area)
    if (area >= minArea && area <= maxArea) keptId.set(id, remap ? nextKept++ : id)
    else rejectedId.set(id, remap ? nextRejected++ : id)
  }

  const keptLabels = ctx.track(new cv.Mat(height, width, cv.CV_32S))
  const rejectedLabels = ctx.track(new cv.Mat(height, width, cv.CV_32S))
  const maskKept = ctx.track(new cv.Mat(height, width, cv.CV_8U, new cv.Scalar(0)))
  const maskRejected = ctx.track(new cv.Mat(height, width, cv.CV_8U, new cv.Scalar(0)))
  const keptData = keptLabels.data32S as Int32Array
  const rejectedData = rejectedLabels.data32S as Int32Array
  const keptBits = maskKept.data as Uint8Array
  const rejectedBits = maskRejected.data as Uint8Array
  keptData.fill(0)
  rejectedData.fill(0)
  for (let p = 0; p < labels.length; p++) {
    const id = labels[p]
    if (id <= 0) continue
    const kept = keptId.get(id)
    if (kept !== undefined) { keptData[p] = kept; keptBits[p] = 255; continue }
    const rejected = rejectedId.get(id)
    if (rejected !== undefined) { rejectedData[p] = rejected; rejectedBits[p] = 255 }
  }

  const sortedAreas = [...surviving].sort((a, b) => a - b)
  const median = sortedAreas.length === 0
    ? 0
    : sortedAreas.length % 2
      ? sortedAreas[sortedAreas.length >> 1]
      : (sortedAreas[(sortedAreas.length >> 1) - 1] + sortedAreas[sortedAreas.length >> 1]) / 2

  const preview = ctx.track(image && typeof image.cols === 'number' ? toBgr8(cv, image) : new cv.Mat(height, width, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))
  if (preview.rows !== height || preview.cols !== width) cv.resize(preview, preview, new cv.Size(width, height))
  // Rejected regions are dimmed so the accepted ones stand out on their own.
  const previewBytes = preview.data as Uint8Array
  for (let p = 0; p < rejectedBits.length; p++) {
    if (!rejectedBits[p]) continue
    previewBytes[p * 3] = previewBytes[p * 3] * 0.25
    previewBytes[p * 3 + 1] = previewBytes[p * 3 + 1] * 0.25
    previewBytes[p * 3 + 2] = previewBytes[p * 3 + 2] * 0.25
  }

  const cyan = new cv.Scalar(255, 230, 0, 255)
  const font = cv.FONT_HERSHEY_SIMPLEX
  if (p1 && p2 && reference > 0) {
    const a = new cv.Point(Math.round(p1[0]), Math.round(p1[1]))
    const b = new cv.Point(Math.round(p2[0]), Math.round(p2[1]))
    cv.line(preview, a, b, cyan, 2, cv.LINE_AA)
    cv.circle(preview, a, 4, cyan, -1)
    cv.circle(preview, b, 4, cyan, -1)
    const mx = Math.round((p1[0] + p2[0]) / 2)
    const my = Math.round((p1[1] + p2[1]) / 2)
    const length = Math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    const shape = Math.round(Number(params.shape) || 0)
    if (shape === 0) cv.circle(preview, new cv.Point(mx, my), Math.round(length / 2), cyan, 1, cv.LINE_AA)
    else if (shape === 1) {
      const half = Math.round(length / 2)
      cv.rectangle(preview, new cv.Point(mx - half, my - half), new cv.Point(mx + half, my + half), cyan, 1)
    }
    const labelY = Math.max(18, my - Math.round(length / 2) - 6)
    cv.putText(preview, `ref=${Math.trunc(reference)}  med=${Math.trunc(median)}  n=${keptId.size}`,
      new cv.Point(10, labelY), font, 0.42, cyan, 1, cv.LINE_AA)
  } else {
    cv.putText(preview, 'Draw a line on a reference object', new cv.Point(10, 22), font, 0.45, new cv.Scalar(255, 180, 80, 255), 1, cv.LINE_AA)
    cv.putText(preview, `n=${keptId.size}  (no ref - passing all)`, new cv.Point(10, 42), font, 0.4, new cv.Scalar(130, 130, 130, 255), 1, cv.LINE_AA)
  }

  return {
    markers_out: keptLabels,
    markers_rej: rejectedLabels,
    mask_kept: maskKept,
    mask_rej: maskRejected,
    main: preview,
    count: keptId.size,
    ref_area: pyRound(reference, 1),
    median_area: pyRound(median, 1),
  }
}

/* ------------------------------------------------------------ export particles */

export const sciExportParticles: NodeImpl = async (inputs, params, ctx) => {
  const cv = ctx.cv
  const trigger = Number(params.export_trigger) ? 1 : 0
  const state = (ctx.state.get(ctx.nodeId) as { last: number } | undefined) ?? { last: 0 }
  const rising = trigger === 1 && state.last === 0
  state.last = trigger
  ctx.state.set(ctx.nodeId, state)

  const image = (inputs.image ?? inputs.main) as any
  const markers = inputs.labels_map ?? inputs.markers
  if (!rising || !image || !markers || typeof (markers as any).cols !== 'number') return { count: 0 }

  const { data, width, height } = labelData(cv, markers)
  const bgr = toBgr8(cv, image)
  const bgra = new cv.Mat()
  cv.cvtColor(bgr, bgra, cv.COLOR_BGR2BGRA)
  bgr.delete()

  // Tight bounding box per label, in one pass over the map.
  const boxes = new Map<number, [number, number, number, number]>()
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const id = data[y * width + x]
      if (id <= 0) continue
      const box = boxes.get(id)
      if (!box) boxes.set(id, [x, y, x, y])
      else {
        if (x < box[0]) box[0] = x
        if (y < box[1]) box[1] = y
        if (x > box[2]) box[2] = x
        if (y > box[3]) box[3] = y
      }
    }
  }

  const pad = Math.max(0, Math.round(Number(params.pad ?? 5)))
  const prefix = String(params.prefix ?? 'stone') || 'stone'
  const stamp = Math.floor(Date.now() / 1000)
  const entries: ZipEntry[] = []

  for (const id of [...boxes.keys()].sort((a, b) => a - b)) {
    const [x0, y0, x1, y1] = boxes.get(id)!
    const left = Math.max(0, x0 - pad)
    const top = Math.max(0, y0 - pad)
    const right = Math.min(width, x1 + 1 + pad)
    const bottom = Math.min(height, y1 + 1 + pad)
    if (right <= left || bottom <= top) continue

    const crop = bgra.roi(new cv.Rect(left, top, right - left, bottom - top)).clone()
    // Only this particle stays opaque; the padding and its neighbours vanish.
    const bytes = crop.data as Uint8Array
    for (let y = top; y < bottom; y++) {
      for (let x = left; x < right; x++) {
        const at = ((y - top) * crop.cols + (x - left)) * 4
        bytes[at + 3] = data[y * width + x] === id ? 255 : 0
      }
    }
    entries.push({ name: `${prefix}_${stamp}_${String(id).padStart(4, '0')}.png`, bytes: await encodeImage(cv, crop, 'png') })
    crop.delete()
  }
  bgra.delete()

  if (entries.length > 0) {
    const zip = buildZip(entries)
    downloadFile(`${prefix}_${stamp}.zip`, zip.buffer as ArrayBuffer, 'application/zip')
  }
  return { count: entries.length }
}
