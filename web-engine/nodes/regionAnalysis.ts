/**
 * Mask / region nodes ported from the desktop plugins mask_keep_largest.py,
 * mask_select_by_marker.py, mask_depth_bands.py, feat_peak_markers.py and
 * sci_zone_stats.py. Each mirrors its Python counterpart step for step.
 *
 * Typed-array views of a Mat (`mat.data`, `data32S`…) point into the WASM heap and
 * become stale when the heap grows, which any new Mat — or even a drawing call —
 * can trigger. Pixels read after another allocation are copied (`.slice()`), and
 * write views are only taken once every output Mat exists.
 */
import type { NodeImpl } from '../types'
import { inputLabels32S, resizeLabels32S, toGray } from '../cvUtils'
import { viridisColor } from '../colormaps'

/** 0/1 CV_8U Mat of any mask, resized to (w, h) when given. */
function binaryMat(cv: any, ctx: any, src: any, w?: number, h?: number): any {
  let g = ctx.track(toGray(cv, src))
  if (w !== undefined && h !== undefined && (g.cols !== w || g.rows !== h)) {
    const r = ctx.track(new cv.Mat())
    cv.resize(g, r, new cv.Size(w, h), 0, 0, cv.INTER_NEAREST)
    g = r
  }
  const out = ctx.track(new cv.Mat(g.rows, g.cols, cv.CV_8U))
  const s = g.data as Uint8Array
  const o = out.data as Uint8Array
  for (let i = 0; i < s.length; i++) o[i] = s[i] > 0 ? 1 : 0
  return out
}

/** New 0/255 mask whose pixel i is set when keep(i). `keep` must only read copied arrays. */
function maskFrom(cv: any, ctx: any, rows: number, cols: number, keep: (i: number) => boolean): any {
  const out = ctx.track(new cv.Mat(rows, cols, cv.CV_8U))
  const o = out.data as Uint8Array
  for (let i = 0; i < o.length; i++) o[i] = keep(i) ? 255 : 0
  return out
}

// ── Keep Largest Blobs ───────────────────────────────────────────────────────

export const maskKeepLargest: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.mask as any
  if (!src) return { mask: null, count: 0, area_px: 0 }
  const cv = ctx.cv
  const bin = binaryMat(cv, ctx, src)
  const labels = ctx.track(new cv.Mat())
  const stats = ctx.track(new cv.Mat())
  const centroids = ctx.track(new cv.Mat())
  const conn = Number(params.connectivity) === 1 ? 4 : 8
  const n = cv.connectedComponentsWithStats(bin, labels, stats, centroids, conn, cv.CV_32S)
  if (n < 2) return { mask: maskFrom(cv, ctx, bin.rows, bin.cols, () => false), count: 0, area_px: 0 }

  const st = (stats.data32S as Int32Array).slice()
  const l = (labels.data32S as Int32Array).slice()
  const keepN = Math.max(1, Math.round(Number(params.count) || 1))
  const order = Array.from({ length: n - 1 }, (_, k) => k + 1).sort((a, b) => st[b * 5 + 4] - st[a * 5 + 4])
  const kept = new Uint8Array(n)
  let area = 0
  for (const id of order.slice(0, keepN)) {
    kept[id] = 1
    area += st[id * 5 + 4]
  }
  return { mask: maskFrom(cv, ctx, bin.rows, bin.cols, (i) => kept[l[i]] === 1), count: Math.min(keepN, n - 1), area_px: area }
}

// ── Select Blobs by Marker ───────────────────────────────────────────────────

export const maskSelectByMarker: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.mask as any
  if (!src) return { mask: null, count: 0 }
  const cv = ctx.cv
  const bin = binaryMat(cv, ctx, src)
  const conn = Number(params.connectivity) === 1 ? 8 : 4
  const labels = ctx.track(new cv.Mat())
  const n = cv.connectedComponents(bin, labels, conn, cv.CV_32S)
  const marker = inputs.marker as any
  const mk = marker ? (binaryMat(cv, ctx, marker, bin.cols, bin.rows).data as Uint8Array).slice() : null
  const l = (labels.data32S as Int32Array).slice()
  if (!mk) return { mask: maskFrom(cv, ctx, bin.rows, bin.cols, (i) => l[i] > 0), count: n - 1 }

  const touched = new Uint8Array(n)
  for (let i = 0; i < l.length; i++) if (mk[i] && l[i] > 0) touched[l[i]] = 1
  const removeMode = Number(params.mode) === 1
  const keep = new Uint8Array(n)
  let count = 0
  for (let id = 1; id < n; id++) {
    keep[id] = removeMode ? 1 - touched[id] : touched[id]
    count += keep[id]
  }
  return { mask: maskFrom(cv, ctx, bin.rows, bin.cols, (i) => keep[l[i]] === 1), count }
}

// ── Mask Depth Bands ─────────────────────────────────────────────────────────

const MAX_BANDS = 10

/** Distance to the outer mask's edge (copied out); with ignoreBorder the frame is not an edge. */
function distanceToOuterEdge(cv: any, ctx: any, outer: any, ignoreBorder: boolean): Float32Array {
  const h = outer.rows
  const w = outer.cols
  if (!ignoreBorder) {
    const d = ctx.track(new cv.Mat())
    cv.distanceTransform(outer, d, cv.DIST_L2, 5)
    return (d.data32F as Float32Array).slice()
  }
  const pad = Math.floor(Math.max(h, w) / 2)
  const padded = ctx.track(new cv.Mat())
  cv.copyMakeBorder(outer, padded, pad, pad, pad, pad, cv.BORDER_REPLICATE)
  const d = ctx.track(new cv.Mat())
  cv.distanceTransform(padded, d, cv.DIST_L2, 5)
  const out = new Float32Array(h * w)
  const pd = d.data32F as Float32Array
  const pw = padded.cols
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = pd[(y + pad) * pw + x + pad]
  return out
}

export const maskDepthBands: NodeImpl = (inputs, params, ctx) => {
  const outerIn = inputs.outer as any
  if (!outerIn) return { bands: null, range_mask: null, band_1: null, band_2: null, band_3: null, depth: null, main: null }
  const cv = ctx.cv
  const outer = binaryMat(cv, ctx, outerIn)
  const h = outer.rows
  const w = outer.cols
  const inner = inputs.inner ? binaryMat(cv, ctx, inputs.inner, w, h) : ctx.track(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0)))
  const o = (outer.data as Uint8Array).slice()
  const inn = (inner.data as Uint8Array).slice()

  let dIn = new Float32Array(h * w)
  if (inn.some((v) => v !== 0)) {
    const notInner = maskFrom(cv, ctx, h, w, (i) => inn[i] === 0)
    const d = ctx.track(new cv.Mat())
    cv.distanceTransform(notInner, d, cv.DIST_L2, 5)
    dIn = (d.data32F as Float32Array).slice()
  }
  const dOut = distanceToOuterEdge(cv, ctx, outer, params.ignore_border !== false)

  const n = Math.min(MAX_BANDS, Math.max(1, Math.round(Number(params.bands) || 3)))
  const lo = Number(params.range_low ?? 0.2)
  const hi = Number(params.range_high ?? 0.8)
  const depthArr = new Float32Array(h * w)
  const bandArr = new Int32Array(h * w)
  for (let i = 0; i < depthArr.length; i++) {
    if (!o[i] || inn[i]) continue
    const v = dIn[i] / (dIn[i] + dOut[i] + 1e-6)
    depthArr[i] = v
    bandArr[i] = Math.min(Math.floor(v * n), n - 1) + 1
  }

  // Every output Mat is allocated before any write view is taken.
  const depth = ctx.track(new cv.Mat(h, w, cv.CV_32F))
  const bands = ctx.track(new cv.Mat(h, w, cv.CV_32S))
  const vis = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0)))
  ;(depth.data32F as Float32Array).set(depthArr)
  ;(bands.data32S as Int32Array).set(bandArr)
  const vp = vis.data as Uint8Array
  for (let i = 0; i < depthArr.length; i++) {
    if (!bandArr[i]) continue
    const [r, g, b] = viridisColor(depthArr[i] * 255)
    vp[i * 3] = b
    vp[i * 3 + 1] = g
    vp[i * 3 + 2] = r
  }
  const out: Record<string, unknown> = {
    bands,
    depth,
    main: vis,
    range_mask: maskFrom(cv, ctx, h, w, (i) => bandArr[i] > 0 && depthArr[i] >= lo && depthArr[i] <= hi),
  }
  for (const k of [1, 2, 3]) out[`band_${k}`] = maskFrom(cv, ctx, h, w, (i) => bandArr[i] === k)
  return out
}

// ── Local Maxima Markers ─────────────────────────────────────────────────────

const FLAT_EPS = 0.02

export const featPeakMarkers: NodeImpl = (inputs, params, ctx) => {
  const img = inputs.image as any
  if (!img) return { markers: null, points: null, count: 0 }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, img))
  const g = ctx.track(new cv.Mat())
  gray.convertTo(g, cv.CV_32F, Number(params.polarity) === 1 ? -1 : 1, 0)
  const sigma = Math.max(0, Number(params.sigma) || 0)
  let smooth = g
  if (sigma > 0) {
    smooth = ctx.track(new cv.Mat())
    cv.GaussianBlur(g, smooth, new cv.Size(0, 0), sigma, sigma, cv.BORDER_DEFAULT)
  }
  const minDist = Math.max(1, Number(params.min_distance) || 7)
  const h = g.rows
  const w = g.cols

  const size = 2 * Math.floor(minDist) + 1
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size))
  const dil = ctx.track(new cv.Mat())
  const ero = ctx.track(new cv.Mat())
  cv.dilate(smooth, dil, kernel)
  cv.erode(smooth, ero, kernel)
  kernel.delete()
  const allowed = inputs.mask ? (binaryMat(cv, ctx, inputs.mask, w, h).data as Uint8Array).slice() : null
  const s = (smooth.data32F as Float32Array).slice()
  const dl = (dil.data32F as Float32Array).slice()
  const er = (ero.data32F as Float32Array).slice()
  let lo = Infinity, hi = -Infinity
  for (let i = 0; i < s.length; i++) { if (s[i] < lo) lo = s[i]; if (s[i] > hi) hi = s[i] }
  const span = hi - lo || 1

  const isPeak = maskFrom(cv, ctx, h, w, (i) => s[i] >= dl[i] && s[i] - er[i] > FLAT_EPS * span && (!allowed || allowed[i] === 1))

  // Plateaus: one point per connected flat peak (its centroid).
  const labels = ctx.track(new cv.Mat())
  const stats = ctx.track(new cv.Mat())
  const centroids = ctx.track(new cv.Mat())
  const n = cv.connectedComponentsWithStats(isPeak, labels, stats, centroids, 8, cv.CV_32S)
  const c = (centroids.data64F as Float64Array).slice()
  const pts: [number, number][] = []
  for (let k = 1; k < n; k++) {
    const x = Math.min(w - 1, Math.max(0, Math.round(c[k * 2])))
    const y = Math.min(h - 1, Math.max(0, Math.round(c[k * 2 + 1])))
    pts.push([y, x])
  }

  // Greedy non-maximum suppression, strongest peaks first (pure JS: no heap views).
  pts.sort((a, b) => s[b[0] * w + b[1]] - s[a[0] * w + a[1]])
  const taken = new Uint8Array(h * w)
  const r = Math.max(1, Math.floor(minDist))
  const kept: [number, number][] = []
  for (const [y, x] of pts) {
    if (taken[y * w + x]) continue
    kept.push([y, x])
    for (let dy = -r; dy <= r; dy++) {
      const yy = y + dy
      if (yy < 0 || yy >= h) continue
      const half = Math.floor(Math.sqrt(r * r - dy * dy))
      for (let xx = Math.max(0, x - half); xx <= Math.min(w - 1, x + half); xx++) taken[yy * w + xx] = 1
    }
  }

  const markers = ctx.track(new cv.Mat(h, w, cv.CV_32S, new cv.Scalar(0)))
  const dots = ctx.track(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0)))
  const mp = markers.data32S as Int32Array
  kept.forEach(([y, x], k) => { mp[y * w + x] = k + 1 })
  // Drawing after the marker writes: cv.circle may allocate and move the heap.
  for (const [y, x] of kept) cv.circle(dots, new cv.Point(x, y), 2, new cv.Scalar(255), -1)
  return { markers, points: dots, count: kept.length }
}

// ── Zone Region Stats ────────────────────────────────────────────────────────

export const sciZoneStats: NodeImpl = (inputs, params, ctx) => {
  const regions = inputs.regions as any
  if (!regions) {
    return { stats: {}, total_count: 0, mean_area: 0, counts: [], mean_areas: [], areas: [], regions_mask: null }
  }
  const cv = ctx.cv
  const h = regions.rows
  const w = regions.cols
  const lab = (inputLabels32S(ctx, regions).data32S as Int32Array).slice()
  let zones: Int32Array | null = null
  let nZones = 1
  const zonesIn = inputs.zones as any
  if (zonesIn) {
    let z = inputLabels32S(ctx, zonesIn)
    if (z.rows !== h || z.cols !== w) z = resizeLabels32S(ctx, z, w, h)
    zones = (z.data32S as Int32Array).slice()
    for (let i = 0; i < zones.length; i++) if (zones[i] > nZones) nZones = zones[i]
  }

  const acc = new Map<number, { a: number; sy: number; sx: number }>()
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const id = lab[y * w + x]
      if (id <= 0) continue
      let e = acc.get(id)
      if (!e) acc.set(id, (e = { a: 0, sy: 0, sx: 0 }))
      e.a++
      e.sy += y
      e.sx += x
    }
  }

  const minA = Number(params.min_area ?? 20)
  const maxA = Number(params.max_area) || 0
  const ignore = Number(params.ignore_label ?? -99)
  const zoneOf = new Map<number, number>()
  for (const [id, e] of acc) {
    if (id === ignore || e.a < minA || (maxA > 0 && e.a > maxA)) continue
    let z = 1
    if (zones) {
      const cy = Math.min(h - 1, Math.max(0, Math.round(e.sy / e.a)))
      const cx = Math.min(w - 1, Math.max(0, Math.round(e.sx / e.a)))
      z = zones[cy * w + cx]
    }
    if (z > 0) zoneOf.set(id, z)
  }

  const counts: number[] = []
  const means: number[] = []
  const stats: Record<string, number> = {}
  for (let z = 1; z <= nZones; z++) {
    const areas = [...zoneOf.entries()].filter(([, zz]) => zz === z).map(([id]) => acc.get(id)!.a)
    const total = areas.reduce((a, b) => a + b, 0)
    const mean = areas.length ? total / areas.length : 0
    counts.push(areas.length)
    means.push(mean)
    stats[`zone_${z}_count`] = areas.length
    stats[`zone_${z}_mean_area`] = mean
    stats[`zone_${z}_total_area`] = total
  }
  const allAreas = [...zoneOf.keys()].map((id) => acc.get(id)!.a)
  const total = allAreas.length
  const meanAll = total ? allAreas.reduce((a, b) => a + b, 0) / total : 0
  stats.total_count = total
  stats.mean_area = meanAll

  const regionsMask = maskFrom(cv, ctx, h, w, (i) => zoneOf.has(lab[i]))
  return { stats, total_count: total, mean_area: meanAll, counts, mean_areas: means, areas: allAreas, regions_mask: regionsMask }
}
