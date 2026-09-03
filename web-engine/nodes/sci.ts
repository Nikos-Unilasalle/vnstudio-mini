import type { NodeImpl } from '../types'
import { drawPolyline, toBgr, toGray } from '../cvUtils'
import { applyColormap, COLORMAPS } from '../colormaps'

function hexToBgr(hex: string, fallback: [number, number, number] = [255, 255, 255]): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex ?? ''))
  if (!m) return fallback
  const int = parseInt(m[1], 16)
  return [int & 255, (int >> 8) & 255, (int >> 16) & 255]
}

/** Enum params arrive as either the option's index or its literal string label. */
function resolveEnum(val: unknown, options: string[], fallback: string): string {
  if (typeof val === 'number') return options[Math.round(val)] ?? fallback
  if (typeof val === 'string' && options.includes(val)) return val
  return fallback
}

function putLines(cv: any, mat: any, lines: string[], x: number, y0: number, dy: number, scale: number, color: [number, number, number]): void {
  const c = new cv.Scalar(color[0], color[1], color[2], 255)
  lines.forEach((line, i) => {
    cv.putText(mat, line, new cv.Point(x, y0 + i * dy), cv.FONT_HERSHEY_SIMPLEX, scale, c, 1, cv.LINE_AA)
  })
}

function darkCanvas(cv: any, w: number, h: number, shade = 18): any {
  return new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(shade, shade, shade))
}

// ---------------------------------------------------------------------------
// First Order Statistics
// ---------------------------------------------------------------------------
export const sciFirstOrderStats: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, mean: 0, variance: 0, entropy: 0, uniformity: 0 }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, src))
  const w = gray.cols
  const h = gray.rows

  const region = Math.round(Number(params.region_size) || 0)
  let roiData: Uint8Array = gray.data as Uint8Array
  let roiRect: [number, number, number, number] | null = null
  if (region > 0) {
    const cy = Math.floor(h / 2)
    const cx = Math.floor(w / 2)
    const r = Math.floor(region / 2)
    const y0 = Math.max(0, cy - r)
    const x0 = Math.max(0, cx - r)
    const roi = ctx.track(gray.roi(new cv.Rect(x0, y0, Math.min(region, w - x0), Math.min(region, h - y0))))
    roiData = roi.data as Uint8Array
    roiRect = [x0, y0, roi.cols, roi.rows]
  }

  const hist = new Float64Array(256)
  for (let i = 0; i < roiData.length; i++) hist[roiData[i]]++
  const total = roiData.length
  for (let i = 0; i < 256; i++) hist[i] /= total

  let mean = 0
  for (let i = 0; i < 256; i++) mean += i * hist[i]
  let variance = 0
  let uniformity = 0
  let entropy = 0
  for (let i = 0; i < 256; i++) {
    variance += (i - mean) ** 2 * hist[i]
    uniformity += hist[i] * hist[i]
    if (hist[i] > 0) entropy -= hist[i] * Math.log2(hist[i])
  }

  const vis = ctx.track(new cv.Mat())
  cv.cvtColor(gray, vis, cv.COLOR_GRAY2BGR)
  if (roiRect) {
    const [x0, y0, rw, rh] = roiRect
    cv.rectangle(vis, new cv.Point(x0, y0), new cv.Point(x0 + rw, y0 + rh), new cv.Scalar(0, 200, 255, 255), 1)
  }
  putLines(cv, vis, [`mean=${mean.toFixed(1)}`, `var=${variance.toFixed(1)}`, `H=${entropy.toFixed(2)}`, `U=${uniformity.toFixed(3)}`], 6, 16, 16, 0.4, [255, 255, 255])

  return {
    main: vis,
    mean: Number(mean.toFixed(2)),
    variance: Number(variance.toFixed(2)),
    entropy: Number(entropy.toFixed(4)),
    uniformity: Number(uniformity.toFixed(6)),
  }
}

// ---------------------------------------------------------------------------
// Image Normalizer
// ---------------------------------------------------------------------------
export const sciNormalizer: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, data: null }
  const cv = ctx.cv
  const mode = Number(params.mode) || 0

  const bgr = ctx.track(toBgr(cv, src))
  const data = bgr.data as Uint8Array
  const norm = new Float32Array(data.length)

  if (mode === 0 || mode === 1) {
    let lo = Infinity
    let hi = -Infinity
    for (let i = 0; i < data.length; i++) {
      if (data[i] < lo) lo = data[i]
      if (data[i] > hi) hi = data[i]
    }
    for (let i = 0; i < data.length; i++) norm[i] = hi === lo ? 0 : (data[i] - lo) / (hi - lo)
  } else if (mode === 2) {
    let sum = 0
    for (let i = 0; i < data.length; i++) sum += data[i]
    const mean = sum / data.length
    let sqsum = 0
    for (let i = 0; i < data.length; i++) sqsum += (data[i] - mean) ** 2
    const std = Math.sqrt(sqsum / data.length)
    for (let i = 0; i < data.length; i++) {
      norm[i] = std < 1e-8 ? 0 : Math.max(0, Math.min(1, (data[i] - mean) / std / 6 + 0.5))
    }
  } else {
    const sorted = Array.from(data).sort((a, b) => a - b)
    const pLow = Number(params.p_low) || 2.0
    const pHigh = Number(params.p_high) || 98.0
    const lo = sorted[Math.min(sorted.length - 1, Math.floor((pLow / 100) * sorted.length))]
    const hi = sorted[Math.min(sorted.length - 1, Math.floor((pHigh / 100) * sorted.length))]
    for (let i = 0; i < data.length; i++) norm[i] = hi <= lo ? 0 : Math.max(0, Math.min(1, (data[i] - lo) / (hi - lo)))
  }

  const out = ctx.track(new cv.Mat(bgr.rows, bgr.cols, cv.CV_8UC3))
  const outData = out.data as Uint8Array
  for (let i = 0; i < norm.length; i++) outData[i] = Math.max(0, Math.min(255, Math.round(norm[i] * 255)))

  return { main: out, data: norm }
}

// ---------------------------------------------------------------------------
// Area (ROI) Statistics
// ---------------------------------------------------------------------------
export const sciRoiStats: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, mean: null, std: null, min: null, max: null, count: null }
  const cv = ctx.cv
  const bgr = ctx.track(toBgr(cv, src))
  const iw = bgr.cols
  const ih = bgr.rows

  let rx = Math.round(((Number(params.x) || 25) / 100) * iw)
  let ry = Math.round(((Number(params.y) || 25) / 100) * ih)
  let rw = Math.max(1, Math.round(((Number(params.w) || 50) / 100) * iw))
  let rh = Math.max(1, Math.round(((Number(params.h) || 50) / 100) * ih))
  rx = Math.min(rx, iw - 1)
  ry = Math.min(ry, ih - 1)
  rw = Math.min(rw, iw - rx)
  rh = Math.min(rh, ih - ry)

  const chan = Number(params.channel) || 0
  const roi = ctx.track(bgr.roi(new cv.Rect(rx, ry, rw, rh)))
  let data: Float32Array
  if (chan === 0) {
    const gray = ctx.track(toGray(cv, roi))
    data = Float32Array.from(gray.data as Uint8Array)
  } else {
    const bytes = roi.data as Uint8Array
    const channelOffset = chan === 1 ? 2 : chan === 2 ? 1 : 0
    data = new Float32Array(rw * rh)
    for (let i = 0, px = 0; i < data.length; i++, px += 3) data[i] = bytes[px + channelOffset]
  }

  let sum = 0
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < data.length; i++) {
    sum += data[i]
    if (data[i] < min) min = data[i]
    if (data[i] > max) max = data[i]
  }
  const mean = sum / data.length
  let sqsum = 0
  for (let i = 0; i < data.length; i++) sqsum += (data[i] - mean) ** 2
  const std = Math.sqrt(sqsum / data.length)

  const [b, g, r] = hexToBgr(String(params.color ?? '#00FFA0'), [160, 255, 0])
  const color = new cv.Scalar(b, g, r, 255)
  const out = ctx.track(bgr.clone())
  cv.rectangle(out, new cv.Point(rx, ry), new cv.Point(rx + rw, ry + rh), color, 1)
  if (params.show_stats !== false) {
    putLines(cv, out, [`mean=${mean.toFixed(1)}`, `std=${std.toFixed(1)}`, `[${min.toFixed(0)},${max.toFixed(0)}]`], rx + 3, ry + 14, 14, 0.38, [b, g, r])
  }

  return { main: out, mean, std, min, max, count: data.length }
}

// ---------------------------------------------------------------------------
// Color Distance
// ---------------------------------------------------------------------------
const COLOR_DIST_CMAPS: Record<string, ((v: number) => [number, number, number]) | null> = {
  Viridis: COLORMAPS.Viridis,
  Jet: COLORMAPS.Jet,
  Plasma: COLORMAPS.Plasma,
  Grayscale: null,
}

export const sciColorDistance: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { dist_map: null, mask: null, min_dist: 0 }
  const cv = ctx.cv
  const bgr = ctx.track(toBgr(cv, src))
  const w = bgr.cols
  const h = bgr.rows
  const data = bgr.data as Uint8Array

  const refMask = inputs.ref_mask as any
  let ref: [number, number, number]
  if (refMask) {
    const grayMask = ctx.track(toGray(cv, refMask))
    let mData = grayMask.data as Uint8Array
    if (grayMask.cols !== w || grayMask.rows !== h) {
      const resized = ctx.track(new cv.Mat())
      cv.resize(grayMask, resized, new cv.Size(w, h), 0, 0, cv.INTER_NEAREST)
      mData = resized.data as Uint8Array
    }
    let sb = 0, sg = 0, sr = 0, n = 0
    for (let i = 0, px = 0; i < mData.length; i++, px += 3) {
      if (mData[i] > 127) {
        sb += data[px]
        sg += data[px + 1]
        sr += data[px + 2]
        n++
      }
    }
    ref = n > 0 ? [sb / n, sg / n, sr / n] : [0, 0, 0]
  } else {
    const [r, g, b] = hexToBgr(String(params.ref_color ?? '#ff0000'), [255, 0, 0]).reverse() as [number, number, number]
    ref = [b, g, r]
  }

  const metric = resolveEnum(params.metric, ['L2', 'L1', 'L∞', 'Cosine', 'Mahalanobis'], 'L2')
  const threshold = Number(params.threshold) ?? 0.2

  let meanC = [0, 0, 0]
  let stdC = [1, 1, 1]
  if (metric === 'Mahalanobis') {
    const n = w * h
    for (let i = 0, px = 0; i < n; i++, px += 3) {
      meanC[0] += data[px]
      meanC[1] += data[px + 1]
      meanC[2] += data[px + 2]
    }
    meanC = meanC.map((v) => v / n)
    const sq = [0, 0, 0]
    for (let i = 0, px = 0; i < n; i++, px += 3) {
      sq[0] += (data[px] - meanC[0]) ** 2
      sq[1] += (data[px + 1] - meanC[1]) ** 2
      sq[2] += (data[px + 2] - meanC[2]) ** 2
    }
    stdC = sq.map((v) => Math.sqrt(v / n) + 1e-8)
  }

  const dist = new Float32Array(w * h)
  const refNorm = Math.sqrt(ref[0] ** 2 + ref[1] ** 2 + ref[2] ** 2) + 1e-8
  let minDist = Infinity
  for (let i = 0, px = 0; i < dist.length; i++, px += 3) {
    const b = data[px], g = data[px + 1], r = data[px + 2]
    let d: number
    if (metric === 'L1') {
      d = (Math.abs(b - ref[0]) + Math.abs(g - ref[1]) + Math.abs(r - ref[2])) / (3 * 255)
    } else if (metric === 'L∞') {
      d = Math.max(Math.abs(b - ref[0]), Math.abs(g - ref[1]), Math.abs(r - ref[2])) / 255
    } else if (metric === 'Cosine') {
      const norm = Math.sqrt(b * b + g * g + r * r) + 1e-8
      const cos = (b * ref[0] + g * ref[1] + r * ref[2]) / (norm * refNorm)
      d = (1 - cos) / 2
    } else if (metric === 'Mahalanobis') {
      const db = (b - ref[0]) / stdC[0], dg = (g - ref[1]) / stdC[1], dr = (r - ref[2]) / stdC[2]
      d = Math.min(1, Math.sqrt(db * db + dg * dg + dr * dr) / Math.sqrt(3))
    } else {
      const db = b - ref[0], dg = g - ref[1], dr = r - ref[2]
      d = Math.sqrt(db * db + dg * dg + dr * dr) / (Math.sqrt(3) * 255)
    }
    d = Math.max(0, Math.min(1, d))
    dist[i] = d
    if (d < minDist) minDist = d
  }

  const cmapName = resolveEnum(params.colormap, ['Viridis', 'Jet', 'Plasma', 'Grayscale'], 'Viridis')
  const gray8 = ctx.track(new cv.Mat(h, w, cv.CV_8U))
  const gray8Data = gray8.data as Uint8Array
  const mask = ctx.track(new cv.Mat(h, w, cv.CV_8U))
  const maskData = mask.data as Uint8Array
  for (let i = 0; i < dist.length; i++) {
    gray8Data[i] = Math.round(dist[i] * 255)
    maskData[i] = dist[i] <= threshold ? 255 : 0
  }

  const cmapFn = COLOR_DIST_CMAPS[cmapName]
  const distMap = cmapFn ? ctx.track(applyColormap(cv, gray8, cmapFn)) : ctx.track((() => { const o = new cv.Mat(); cv.cvtColor(gray8, o, cv.COLOR_GRAY2BGR); return o })())

  return { dist_map: distMap, mask, min_dist: Number(minDist.toFixed(4)) }
}

// ---------------------------------------------------------------------------
// Delta E
// ---------------------------------------------------------------------------
function toLab(cv: any, ctx: any, img: any): Float32Array {
  const bgr = ctx.track(toBgr(cv, img))
  const lab = ctx.track(new cv.Mat())
  cv.cvtColor(bgr, lab, cv.COLOR_BGR2Lab)
  const src = lab.data as Uint8Array
  const out = new Float32Array(src.length)
  for (let i = 0; i < src.length; i += 3) {
    out[i] = src[i] * (100 / 255)
    out[i + 1] = src[i + 1] - 128
    out[i + 2] = src[i + 2] - 128
  }
  return out
}

function cie2000(L1: number, a1: number, b1: number, L2: number, a2: number, b2: number): number {
  const C1 = Math.hypot(a1, b1)
  const C2 = Math.hypot(a2, b2)
  const Cavg = (C1 + C2) / 2
  const Cavg7 = Cavg ** 7
  const G = 0.5 * (1 - Math.sqrt(Cavg7 / (Cavg7 + 25 ** 7)))
  const a1p = a1 * (1 + G)
  const a2p = a2 * (1 + G)
  const C1p = Math.hypot(a1p, b1)
  const C2p = Math.hypot(a2p, b2)
  const h1p = (((Math.atan2(b1, a1p) * 180) / Math.PI) + 360) % 360
  const h2p = (((Math.atan2(b2, a2p) * 180) / Math.PI) + 360) % 360

  const dLp = L2 - L1
  const dCp = C2p - C1p
  let dhRaw = h2p - h1p
  if (dhRaw > 180) dhRaw -= 360
  if (dhRaw < -180) dhRaw += 360
  const zeroC = C1p * C2p < 1e-8
  if (zeroC) dhRaw = 0
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhRaw / 2) * (Math.PI / 180))

  const LpAvg = (L1 + L2) / 2
  const CpAvg = (C1p + C2p) / 2
  const hSum = h1p + h2p
  let hAvg: number
  if (zeroC) hAvg = hSum
  else if (Math.abs(h1p - h2p) <= 180) hAvg = hSum / 2
  else hAvg = hSum < 360 ? (hSum + 360) / 2 : (hSum - 360) / 2

  const hAvgR = (hAvg * Math.PI) / 180
  const T =
    1 -
    0.17 * Math.cos(hAvgR - (30 * Math.PI) / 180) +
    0.24 * Math.cos(2 * hAvgR) +
    0.32 * Math.cos(3 * hAvgR + (6 * Math.PI) / 180) -
    0.2 * Math.cos(4 * hAvgR - (63 * Math.PI) / 180)

  const SL = 1 + (0.015 * (LpAvg - 50) ** 2) / Math.sqrt(20 + (LpAvg - 50) ** 2)
  const SC = 1 + 0.045 * CpAvg
  const SH = 1 + 0.015 * CpAvg * T

  const CpAvg7 = CpAvg ** 7
  const RC = 2 * Math.sqrt(CpAvg7 / (CpAvg7 + 25 ** 7))
  const dTheta = 30 * Math.exp(-(((hAvg - 275) / 25) ** 2))
  const RT = -Math.sin(2 * dTheta * (Math.PI / 180)) * RC

  return Math.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))
}

export const sciDeltaE: NodeImpl = (inputs, params, ctx) => {
  const imgA = inputs.image_a as any
  const imgB = inputs.image_b as any
  if (!imgA || !imgB) return { main: null, delta_e: 0, de_max: 0, de_p95: 0 }
  const cv = ctx.cv

  const H = Math.max(imgA.rows, imgB.rows)
  const W = Math.max(imgA.cols, imgB.cols)
  const fit = (img: any) => {
    if (img.rows === H && img.cols === W) return img
    const resized = new cv.Mat()
    cv.resize(img, resized, new cv.Size(W, H), 0, 0, cv.INTER_LINEAR)
    return resized
  }
  const a = fit(imgA)
  const b = fit(imgB)

  const labA = toLab(cv, ctx, a)
  const labB = toLab(cv, ctx, b)
  const formula = resolveEnum(params.formula, ['CIE76', 'CIE2000'], 'CIE2000')

  const deMap = new Float32Array(W * H)
  for (let i = 0, p = 0; i < deMap.length; i++, p += 3) {
    if (formula === 'CIE76') {
      const dL = labB[p] - labA[p]
      const da = labB[p + 1] - labA[p + 1]
      const db = labB[p + 2] - labA[p + 2]
      deMap[i] = Math.sqrt(dL * dL + da * da + db * db)
    } else {
      deMap[i] = cie2000(labA[p], labA[p + 1], labA[p + 2], labB[p], labB[p + 1], labB[p + 2])
    }
  }

  let sum = 0
  let max = -Infinity
  for (let i = 0; i < deMap.length; i++) {
    sum += deMap[i]
    if (deMap[i] > max) max = deMap[i]
  }
  const mean = sum / deMap.length
  const sorted = Float32Array.from(deMap).sort()
  const p95 = sorted[Math.min(sorted.length - 1, Math.floor(0.95 * sorted.length))]

  const deMaxDisp = Number(params.de_max_disp) || 10.0
  const cmapName = resolveEnum(params.colormap, ['Viridis', 'Plasma', 'Hot', 'Jet'], 'Viridis')
  const cmapFn = COLORMAPS[cmapName] ?? COLORMAPS.Viridis
  const gray8 = ctx.track(new cv.Mat(H, W, cv.CV_8U))
  const gray8Data = gray8.data as Uint8Array
  for (let i = 0; i < deMap.length; i++) gray8Data[i] = Math.max(0, Math.min(255, Math.round((deMap[i] / Math.max(deMaxDisp, 1e-6)) * 255)))
  const vis = ctx.track(applyColormap(cv, gray8, cmapFn))

  putLines(cv, vis, [`${formula}  mean=${mean.toFixed(2)}  max=${max.toFixed(2)}  p95=${p95.toFixed(2)}`], 8, 20, 0, 0.42, [255, 255, 255])
  putLines(cv, vis, [`scale: 0-${deMaxDisp.toFixed(0)}`], 8, H - 8, 0, 0.38, [200, 200, 200])

  return { main: vis, delta_e: Number(mean.toFixed(4)), de_max: Number(max.toFixed(4)), de_p95: Number(p95.toFixed(4)) }
}

// ---------------------------------------------------------------------------
// Channel Formula
// ---------------------------------------------------------------------------
const CHANNEL_EXPR_CMAPS = [null, COLORMAPS.Viridis, COLORMAPS.Plasma, COLORMAPS.Hot, COLORMAPS.Jet, COLORMAPS.Turbo]

export const sciChannelExpr: NodeImpl = (inputs, params, ctx) => {
  const img = inputs.image as any
  if (!img) return { result: null, visualized: null }
  const cv = ctx.cv
  const bgr = ctx.track(toBgr(cv, img))
  const w = bgr.cols
  const h = bgr.rows
  const n = w * h

  const bgrData = bgr.data as Uint8Array
  const hsv = ctx.track(new cv.Mat())
  cv.cvtColor(bgr, hsv, cv.COLOR_BGR2HSV)
  const hsvData = hsv.data as Uint8Array
  const lab = ctx.track(new cv.Mat())
  cv.cvtColor(bgr, lab, cv.COLOR_BGR2Lab)
  const labData = lab.data as Uint8Array

  const expr = String(params.expression ?? 'B * 0.5 + R * 0.3 - G * 0.8').trim()
  const argNames = ['R', 'G', 'B', 'H', 'S', 'V', 'L', 'A_lab', 'B_lab', 'abs', 'sqrt', 'clip', 'log', 'exp', 'sin', 'cos', 'min', 'max', 'pi']
  let fn: Function | null = null
  try {
    fn = new Function(...argNames, `"use strict"; return (${expr});`)
  } catch {
    fn = null
  }

  const result = new Float32Array(n)
  const clip = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
  if (fn) {
    try {
      for (let i = 0, px = 0; i < n; i++, px += 3) {
        const B = bgrData[px], G = bgrData[px + 1], R = bgrData[px + 2]
        const Hh = hsvData[px], S = hsvData[px + 1], V = hsvData[px + 2]
        const L = labData[px], Alab = labData[px + 1], Blab = labData[px + 2]
        result[i] = fn(R, G, B, Hh, S, V, L, Alab, Blab, Math.abs, Math.sqrt, clip, Math.log, Math.exp, Math.sin, Math.cos, Math.min, Math.max, Math.PI)
      }
    } catch {
      result.fill(0)
    }
  }

  const cmin = Number(params.clamp_min) ?? -1e9
  const cmax = Number(params.clamp_max) ?? 1e9
  if (cmin > -1e8 || cmax < 1e8) {
    for (let i = 0; i < result.length; i++) result[i] = clip(result[i], cmin, cmax)
  }

  const doNorm = params.normalize !== false
  const cmapIdx = Number(params.colormap) || 0
  const visU8 = ctx.track(new cv.Mat(h, w, cv.CV_8U))
  const visU8Data = visU8.data as Uint8Array
  if (doNorm) {
    let lo = Infinity
    let hi = -Infinity
    for (let i = 0; i < result.length; i++) {
      if (result[i] < lo) lo = result[i]
      if (result[i] > hi) hi = result[i]
    }
    for (let i = 0; i < result.length; i++) visU8Data[i] = hi > lo ? Math.round(((result[i] - lo) / (hi - lo + 1e-10)) * 255) : 0
  } else {
    for (let i = 0; i < result.length; i++) visU8Data[i] = Math.max(0, Math.min(255, Math.round(result[i])))
  }

  const cmapFn = CHANNEL_EXPR_CMAPS[cmapIdx] ?? null
  const visualized = cmapFn
    ? ctx.track(applyColormap(cv, visU8, cmapFn))
    : ctx.track((() => { const o = new cv.Mat(); cv.cvtColor(visU8, o, cv.COLOR_GRAY2BGR); return o })())

  return { result, visualized }
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------
function channelStats(data: Uint8Array, stride: number, offset: number): { mean: number; std: number; min: number; max: number; median: number; hist: number[] } {
  const counts = new Array(256).fill(0)
  let sum = 0
  const n = data.length / stride
  for (let i = 0, p = offset; i < n; i++, p += stride) {
    counts[data[p]]++
    sum += data[p]
  }
  const mean = sum / n
  let sqsum = 0
  for (let i = 0, p = offset; i < n; i++, p += stride) sqsum += (data[p] - mean) ** 2
  const std = Math.sqrt(sqsum / n)
  let min = 0
  let max = 0
  for (let i = 0; i < 256; i++) if (counts[i] > 0) { min = i; break }
  for (let i = 255; i >= 0; i--) if (counts[i] > 0) { max = i; break }
  let cum = 0
  let median = 0
  const half = n / 2
  for (let i = 0; i < 256; i++) {
    cum += counts[i]
    if (cum >= half) { median = i; break }
  }
  return { mean, std, min, max, median, hist: counts }
}

export const sciHistogram: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, mean: null, std: null, data: null }
  const cv = ctx.cv
  const bgr = ctx.track(toBgr(cv, src))
  const w = 512
  const h = 300
  const mode = Number(params.mode) || 0
  const bins = Math.max(1, Math.round(Number(params.bins) || 256))
  const logScale = !!params.log_scale
  const showStats = params.show_stats !== false

  const out = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(18, 18, 18)))
  for (let i = 1; i < 4; i++) {
    const x = Math.round((w * i) / 4)
    const y = Math.round((h * i) / 4)
    cv.line(out, new cv.Point(x, 0), new cv.Point(x, h), new cv.Scalar(45, 45, 45, 255), 1)
    cv.line(out, new cv.Point(0, y), new cv.Point(w, y), new cv.Scalar(45, 45, 45, 255), 1)
  }

  const data = bgr.data as Uint8Array
  const isColor = bgr.channels() === 3
  let channels: { idx: number; color: [number, number, number]; name: string }[]
  if (mode === 1 && isColor) {
    const gray = ctx.track(toGray(cv, bgr))
    channels = [{ idx: 0, color: [220, 220, 220], name: 'Luminance' }]
    var statsData: Uint8Array = gray.data as Uint8Array
    var statsStride = 1
  } else if (isColor) {
    channels = [
      { idx: 0, color: [255, 120, 100], name: 'Blue channel' },
      { idx: 1, color: [100, 255, 120], name: 'Green channel' },
      { idx: 2, color: [100, 120, 255], name: 'Red channel' },
    ]
    statsData = data
    statsStride = 3
  } else {
    channels = [{ idx: 0, color: [220, 220, 220], name: 'Intensity' }]
    statsData = data
    statsStride = 1
  }

  const perChannel: ReturnType<typeof channelStats>[] = []
  const histData: number[][] = []
  let maxVal = 0
  for (const ch of channels) {
    const stats = channelStats(statsData!, statsStride!, ch.idx)
    perChannel.push(stats)
    // Rebin into `bins` buckets.
    const rebinned = new Array(bins).fill(0)
    for (let i = 0; i < 256; i++) rebinned[Math.min(bins - 1, Math.floor((i / 256) * bins))] += stats.hist[i]
    const display = logScale ? rebinned.map((v) => Math.log10(v + 1)) : rebinned
    histData.push(display)
    for (const v of display) if (v > maxVal) maxVal = v
  }

  if (maxVal > 0) {
    for (let ci = 0; ci < channels.length; ci++) {
      const color = new cv.Scalar(channels[ci].color[0], channels[ci].color[1], channels[ci].color[2], 255)
      const hd = histData[ci]
      const pts: { x: number; y: number }[] = []
      for (let b = 0; b < bins; b++) {
        const normVal = (hd[b] / maxVal) * (h - 60)
        const px = bins > 1 ? Math.round((b * w) / (bins - 1)) : 0
        const py = Math.round(h - 20 - normVal)
        pts.push({ x: px, y: py })
      }
      const flat = pts.flatMap((p) => [p.x, p.y])
      flat.push(w - 1, h - 21, 0, h - 21)
      const fillMat = cv.matFromArray(pts.length + 2, 1, cv.CV_32SC2, flat)
      const fillVec = new cv.MatVector()
      fillVec.push_back(fillMat)
      const overlay = ctx.track(out.clone())
      cv.fillPoly(overlay, fillVec, color)
      const blended = ctx.track(new cv.Mat())
      cv.addWeighted(overlay, 0.15, out, 0.85, 0, blended)
      blended.copyTo(out)
      fillMat.delete()
      fillVec.delete()

      drawPolyline(cv, out, pts, false, color, 2)

      if (showStats) {
        const s = perChannel[ci]
        putLines(cv, out, [`${channels[ci].name}: Mean=${s.mean.toFixed(1)} Std=${s.std.toFixed(1)}`], 10, 25 + ci * 20, 0, 0.5, channels[ci].color)
      }
    }
  }

  const luma = isColor ? ctx.track(toGray(cv, bgr)) : bgr
  const lumaStats = channelStats(luma.data as Uint8Array, 1, 0)

  return {
    main: out,
    mean: Number(lumaStats.mean.toFixed(2)),
    std: Number(lumaStats.std.toFixed(2)),
    data: {
      channels: channels.map((c) => c.name),
      mean: perChannel.map((s) => s.mean),
      std: perChannel.map((s) => s.std),
      min: perChannel.map((s) => s.min),
      max: perChannel.map((s) => s.max),
      median: perChannel.map((s) => s.median),
      bins,
      pixels: luma.rows * luma.cols,
      log_scale: logScale,
      luma_mean: lumaStats.mean,
      luma_std: lumaStats.std,
    },
  }
}

// ---------------------------------------------------------------------------
// Histogram Compare
// ---------------------------------------------------------------------------
function extractChannel(cv: any, ctx: any, img: any, channel: string): Uint8Array {
  const bgr = ctx.track(toBgr(cv, img))
  if (channel === 'Luma') {
    const yuv = ctx.track(new cv.Mat())
    cv.cvtColor(bgr, yuv, cv.COLOR_BGR2YUV)
    const out = new Uint8Array(bgr.rows * bgr.cols)
    const src = yuv.data as Uint8Array
    for (let i = 0, p = 0; i < out.length; i++, p += 3) out[i] = src[p]
    return out
  }
  if (channel === 'Hue') {
    const hsv = ctx.track(new cv.Mat())
    cv.cvtColor(bgr, hsv, cv.COLOR_BGR2HSV)
    const out = new Uint8Array(bgr.rows * bgr.cols)
    const src = hsv.data as Uint8Array
    for (let i = 0, p = 0; i < out.length; i++, p += 3) out[i] = src[p]
    return out
  }
  const offset = channel === 'R' ? 2 : channel === 'G' ? 1 : 0
  const src = bgr.data as Uint8Array
  const out = new Uint8Array(bgr.rows * bgr.cols)
  for (let i = 0, p = 0; i < out.length; i++, p += 3) out[i] = src[p + offset]
  return out
}

function computeMaskedHist(channelData: Uint8Array, mask: Uint8Array | null, bins: number, maxVal: number): Float32Array {
  const hist = new Float32Array(bins)
  let total = 0
  for (let i = 0; i < channelData.length; i++) {
    if (mask && mask[i] === 0) continue
    const bin = Math.min(bins - 1, Math.floor((channelData[i] / maxVal) * bins))
    hist[bin]++
    total++
  }
  if (total > 0) for (let i = 0; i < bins; i++) hist[i] /= total
  return hist
}

export const sciHistCompare: NodeImpl = (inputs, params, ctx) => {
  const imgA = inputs.image_a as any
  const imgB = inputs.image_b as any
  if (!imgA || !imgB) return { main: null, distance: 0 }
  const cv = ctx.cv

  const metric = resolveEnum(params.metric, ['Bhattacharyya', 'Chi-squared', 'Wasserstein', 'Correlation'], 'Bhattacharyya')
  const channel = resolveEnum(params.channel, ['Luma', 'R', 'G', 'B', 'Hue'], 'Luma')
  const bins = Math.max(8, Math.round(Number(params.bins) || 64))
  const isHue = channel === 'Hue'
  const maxVal = isHue ? 180 : 256

  const chA = extractChannel(cv, ctx, imgA, channel)
  const chB = extractChannel(cv, ctx, imgB, channel)

  const prepMask = (m: any, refShape: [number, number]) => {
    if (!m) return null
    const gray = ctx.track(toGray(cv, m))
    let d = gray.data as Uint8Array
    if (gray.cols !== refShape[1] || gray.rows !== refShape[0]) {
      const resized = ctx.track(new cv.Mat())
      cv.resize(gray, resized, new cv.Size(refShape[1], refShape[0]), 0, 0, cv.INTER_NEAREST)
      d = resized.data as Uint8Array
    }
    return d
  }
  const maskA = prepMask(inputs.mask_a, [imgA.rows, imgA.cols])
  const maskB = prepMask(inputs.mask_b, [imgB.rows, imgB.cols])

  const hA = computeMaskedHist(chA, maskA, bins, maxVal)
  const hB = computeMaskedHist(chB, maskB, bins, maxVal)

  let dist: number
  if (metric === 'Chi-squared') {
    let s = 0
    for (let i = 0; i < bins; i++) s += (hA[i] - hB[i]) ** 2 / (hA[i] + hB[i] + 1e-8)
    dist = s
  } else if (metric === 'Bhattacharyya') {
    let bc = 0
    for (let i = 0; i < bins; i++) bc += Math.sqrt(hA[i] * hB[i] + 1e-8)
    dist = Math.min(-Math.log(bc + 1e-8), 5.0)
  } else if (metric === 'Wasserstein') {
    let cdfA = 0, cdfB = 0, s = 0
    for (let i = 0; i < bins; i++) {
      cdfA += hA[i]
      cdfB += hB[i]
      s += Math.abs(cdfA - cdfB)
    }
    dist = s / bins
  } else {
    let meanA = 0, meanB = 0
    for (let i = 0; i < bins; i++) { meanA += hA[i]; meanB += hB[i] }
    meanA /= bins
    meanB /= bins
    let num = 0, denA = 0, denB = 0
    for (let i = 0; i < bins; i++) {
      num += (hA[i] - meanA) * (hB[i] - meanB)
      denA += (hA[i] - meanA) ** 2
      denB += (hB[i] - meanB) ** 2
    }
    const corr = num / (Math.sqrt(denA * denB) + 1e-8)
    dist = (1 - corr) / 2
  }

  const W = 512
  const H = 256
  const canvas = ctx.track(new cv.Mat(H + 50, W, cv.CV_8UC3, new cv.Scalar(0, 0, 0)))
  const barW = Math.max(1, Math.floor(W / bins))
  let maxHist = 1e-8
  for (let i = 0; i < bins; i++) maxHist = Math.max(maxHist, hA[i], hB[i])
  for (let i = 0; i < bins; i++) {
    const x = i * barW
    const hValA = Math.round((hA[i] / maxHist) * H)
    cv.rectangle(canvas, new cv.Point(x, H - hValA), new cv.Point(x + barW - 1, H), new cv.Scalar(0, 180, 200, 255), -1)
    const hValB = Math.round((hB[i] / maxHist) * H)
    cv.rectangle(canvas, new cv.Point(x, H - hValB), new cv.Point(x + barW - 1, H), new cv.Scalar(255, 140, 0, 255), 1)
  }
  putLines(cv, canvas, [`${metric} [${channel}]: ${dist.toFixed(4)}`], 8, H + 30, 0, 0.5, [200, 200, 200])
  putLines(cv, canvas, ['A'], 8, 20, 0, 0.5, [0, 180, 200])
  putLines(cv, canvas, ['B'], 30, 20, 0, 0.5, [255, 140, 0])

  return { main: canvas, distance: Number(dist.toFixed(4)) }
}

// ---------------------------------------------------------------------------
// Mask Metrics
// ---------------------------------------------------------------------------
export const sciMaskMetrics: NodeImpl = (inputs, params, ctx) => {
  const pred = inputs.pred as any
  const truth = inputs.truth as any
  if (!pred || !truth) return { main: null, data: null, iou: 0, dice: 0 }
  const cv = ctx.cv

  let predGray = ctx.track(toGray(cv, pred))
  let truthGray = ctx.track(toGray(cv, truth))
  if (predGray.cols !== truthGray.cols || predGray.rows !== truthGray.rows) {
    const H = Math.max(predGray.rows, truthGray.rows)
    const W = Math.max(predGray.cols, truthGray.cols)
    const rp = ctx.track(new cv.Mat())
    cv.resize(predGray, rp, new cv.Size(W, H), 0, 0, cv.INTER_NEAREST)
    predGray = rp
    const rt = ctx.track(new cv.Mat())
    cv.resize(truthGray, rt, new cv.Size(W, H), 0, 0, cv.INTER_NEAREST)
    truthGray = rt
  }

  const H = predGray.rows
  const W = predGray.cols
  const pData = predGray.data as Uint8Array
  const tData = truthGray.data as Uint8Array

  let vp = 0, fp = 0, fn = 0
  for (let i = 0; i < pData.length; i++) {
    const p = pData[i] > 127
    const t = tData[i] > 127
    if (p && t) vp++
    else if (p && !t) fp++
    else if (!p && t) fn++
  }

  const iou = vp + fp + fn > 0 ? vp / (vp + fp + fn) : 0
  const dice = 2 * vp + fp + fn > 0 ? (2 * vp) / (2 * vp + fp + fn) : 0
  const precision = vp + fp > 0 ? vp / (vp + fp) : 0
  const recall = vp + fn > 0 ? vp / (vp + fn) : 0
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0

  const show = params.show_overlay !== false
  const alpha = Number(params.alpha) ?? 0.45
  const bgImg = inputs.image as any

  let base: any
  if (bgImg) {
    base = ctx.track(toBgr(cv, bgImg))
    if (base.cols !== W || base.rows !== H) {
      const resized = ctx.track(new cv.Mat())
      cv.resize(base, resized, new cv.Size(W, H), 0, 0, cv.INTER_LINEAR)
      base = resized
    }
  } else {
    base = ctx.track(new cv.Mat(H, W, cv.CV_8UC3, new cv.Scalar(0, 0, 0)))
  }

  const overlay = ctx.track(base.clone())
  if (show) {
    const baseData = base.data as Uint8Array
    const overlayData = overlay.data as Uint8Array
    for (let i = 0, px = 0; i < pData.length; i++, px += 3) {
      const p = pData[i] > 127
      const t = tData[i] > 127
      let color: [number, number, number] | null = null
      if (p && t) color = [0, 200, 0]
      else if (p && !t) color = [0, 0, 220]
      else if (!p && t) color = [200, 0, 0]
      if (color) {
        overlayData[px] = Math.round(baseData[px] * (1 - alpha) + color[0] * alpha)
        overlayData[px + 1] = Math.round(baseData[px + 1] * (1 - alpha) + color[1] * alpha)
        overlayData[px + 2] = Math.round(baseData[px + 2] * (1 - alpha) + color[2] * alpha)
      }
    }
  }

  putLines(cv, overlay, [`IoU=${iou.toFixed(3)}  Dice=${dice.toFixed(3)}`, `P=${precision.toFixed(3)}  R=${recall.toFixed(3)}  F1=${f1.toFixed(3)}`, `VP=${vp}  FP=${fp}  FN=${fn}`], 8, 22, 18, 0.45, [255, 255, 255])
  putLines(cv, overlay, ['VP'], 8, H - 38, 0, 0.4, [0, 200, 0])
  putLines(cv, overlay, ['FP'], 32, H - 38, 0, 0.4, [0, 0, 220])
  putLines(cv, overlay, ['FN'], 56, H - 38, 0, 0.4, [200, 0, 0])

  const data = { iou: round4(iou), dice: round4(dice), precision: round4(precision), recall: round4(recall), f1: round4(f1), vp, fp, fn }
  return { main: overlay, data, iou: round4(iou), dice: round4(dice) }
}

function round4(v: number): number {
  return Number(v.toFixed(4))
}

// ---------------------------------------------------------------------------
// Line Profile
// ---------------------------------------------------------------------------
function bilinear(data: Uint8Array, w: number, h: number, x: number, y: number): number {
  const x0 = Math.max(0, Math.min(w - 1, Math.floor(x)))
  const y0 = Math.max(0, Math.min(h - 1, Math.floor(y)))
  const x1 = Math.min(w - 1, x0 + 1)
  const y1 = Math.min(h - 1, y0 + 1)
  const fx = Math.max(0, Math.min(1, x - x0))
  const fy = Math.max(0, Math.min(1, y - y0))
  const v00 = data[y0 * w + x0]
  const v10 = data[y0 * w + x1]
  const v01 = data[y1 * w + x0]
  const v11 = data[y1 * w + x1]
  return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy
}

export const sciLineProfile: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, chart: null, profile: null }
  const cv = ctx.cv
  const bgr = ctx.track(toBgr(cv, src))
  const iw = bgr.cols
  const ih = bgr.rows

  const px1 = Math.round(((Number(params.x1) || 10) / 100) * iw)
  const py1 = Math.round(((Number(params.y1) || 50) / 100) * ih)
  const px2 = Math.round(((Number(params.x2) || 90) / 100) * iw)
  const py2 = Math.round(((Number(params.y2) || 50) / 100) * ih)
  const n = Math.max(2, Math.round(Number(params.samples) || 256))
  const lw = Math.max(1, Math.round(Number(params.line_width) || 1))
  const chan = Number(params.channel) || 0

  let gray: Uint8Array
  if (chan === 0) {
    const g = ctx.track(toGray(cv, bgr))
    gray = g.data as Uint8Array
  } else {
    const bytes = bgr.data as Uint8Array
    const off = chan === 1 ? 2 : chan === 2 ? 1 : 0
    gray = new Uint8Array(iw * ih)
    for (let i = 0, p = 0; i < gray.length; i++, p += 3) gray[i] = bytes[p + off]
  }

  const profile = new Float64Array(n)
  const dx = px2 - px1
  const dy = py2 - py1
  const length = Math.max(Math.hypot(dx, dy), 1e-6)
  const nx = -dy / length
  const ny = dx / length
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1)
    const x = px1 + dx * t
    const y = py1 + dy * t
    if (lw <= 1) {
      profile[i] = bilinear(gray, iw, ih, x, y)
    } else {
      let sum = 0
      for (let k = 0; k < lw; k++) {
        const off = -(lw - 1) / 2 + k
        sum += bilinear(gray, iw, ih, x + nx * off, y + ny * off)
      }
      profile[i] = sum / lw
    }
  }

  const out = ctx.track(bgr.clone())
  cv.line(out, new cv.Point(px1, py1), new cv.Point(px2, py2), new cv.Scalar(100, 255, 0, 255), 1, cv.LINE_AA)
  cv.circle(out, new cv.Point(px1, py1), 4, new cv.Scalar(255, 200, 0, 255), -1)
  cv.circle(out, new cv.Point(px2, py2), 4, new cv.Scalar(0, 100, 255, 255), -1)

  const cw = 400
  const ch = 200
  const chart = ctx.track(new cv.Mat(ch, cw, cv.CV_8UC3, new cv.Scalar(18, 18, 18)))
  for (let i = 1; i < 4; i++) {
    cv.line(chart, new cv.Point(Math.round((cw * i) / 4), 0), new cv.Point(Math.round((cw * i) / 4), ch), new cv.Scalar(40, 40, 40, 255), 1)
    cv.line(chart, new cv.Point(0, Math.round((ch * i) / 4)), new cv.Point(cw, Math.round((ch * i) / 4)), new cv.Scalar(40, 40, 40, 255), 1)
  }

  let pMin = Infinity
  let pMax = -Infinity
  for (let i = 0; i < profile.length; i++) {
    if (profile[i] < pMin) pMin = profile[i]
    if (profile[i] > pMax) pMax = profile[i]
  }
  if (pMax <= pMin) pMax = pMin + 1
  const pad = 12
  const pts: { x: number; y: number }[] = []
  for (let i = 0; i < profile.length; i++) {
    const normV = Math.round(((profile[i] - pMin) / (pMax - pMin)) * (ch - 2 * pad))
    pts.push({ x: Math.round((i / profile.length) * cw), y: ch - pad - normV })
  }
  drawPolyline(cv, chart, pts, false, new cv.Scalar(100, 255, 0, 255), 1)

  putLines(cv, chart, [pMax.toFixed(1)], 4, 14, 0, 0.38, [160, 160, 160])
  putLines(cv, chart, [pMin.toFixed(1)], 4, ch - 4, 0, 0.38, [160, 160, 160])

  return { main: out, chart, profile: Array.from(profile) }
}

// ---------------------------------------------------------------------------
// Scale Bar
// ---------------------------------------------------------------------------
export const sciScaleBar: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const out = ctx.track(toBgr(cv, src))
  const w = out.cols
  const h = out.rows

  const ppu = Math.max(Number(params.pixels_per_unit) || 100, 0.001)
  const barUnits = Number(params.bar_length) || 1.0
  let barPx = Math.round(ppu * barUnits)
  const unit = String(params.unit_name ?? 'mm')
  const posIdx = Number(params.position) || 0
  const thickness = Math.round(Number(params.thickness) || 3)
  const margin = Math.round(Number(params.margin) || 20)
  const [b, g, r] = hexToBgr(String(params.color ?? '#FFFFFF'))
  const color = new cv.Scalar(b, g, r, 255)

  barPx = Math.max(10, Math.min(barPx, w - 2 * margin))
  const label = `${barUnits} ${unit}`
  // cv.getTextSize isn't in this OpenCV.js build; HERSHEY_SIMPLEX at scale 1.0
  // averages ~17px per char wide and ~22px tall, scaled linearly with fontScale.
  const fontScale = 0.5
  const th = Math.round(22 * fontScale)
  const tw = Math.round(17 * fontScale * label.length)
  const tickH = thickness * 2

  let x1: number, x2: number, y2: number
  if (posIdx === 0) {
    x2 = w - margin
    y2 = h - margin
    x1 = x2 - barPx
  } else if (posIdx === 1) {
    x1 = margin
    y2 = h - margin
    x2 = x1 + barPx
  } else if (posIdx === 2) {
    x2 = w - margin
    y2 = margin + th + 10 + thickness
    x1 = x2 - barPx
  } else {
    x1 = margin
    y2 = margin + th + 10 + thickness
    x2 = x1 + barPx
  }

  cv.line(out, new cv.Point(x1, y2), new cv.Point(x2, y2), color, thickness)
  cv.line(out, new cv.Point(x1, y2 - tickH), new cv.Point(x1, y2 + tickH), color, thickness)
  cv.line(out, new cv.Point(x2, y2 - tickH), new cv.Point(x2, y2 + tickH), color, thickness)

  const tx = Math.floor((x1 + x2) / 2 - tw / 2)
  const ty = y2 - tickH - 4
  cv.putText(out, label, new cv.Point(tx, ty), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

  return { main: out }
}

// ---------------------------------------------------------------------------
// Focus Metric
// ---------------------------------------------------------------------------
export const sciFocusMetric: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, score: 0 }
  const cv = ctx.cv
  const bgr = ctx.track(toBgr(cv, src))
  const w = bgr.cols
  const h = bgr.rows

  const margin = (Number(params.roi_margin) || 0) / 100
  const mx = Math.round(w * margin)
  const my = Math.round(h * margin)
  const roi = margin > 0 && my < h / 2 && mx < w / 2 ? ctx.track(bgr.roi(new cv.Rect(mx, my, w - 2 * mx, h - 2 * my))) : bgr
  const gray = ctx.track(toGray(cv, roi))

  const method = Number(params.method) || 0
  let score: number
  if (method === 0) {
    const lap = ctx.track(new cv.Mat())
    cv.Laplacian(gray, lap, cv.CV_64F, 1, 1, 0, cv.BORDER_DEFAULT)
    const data = lap.data64F as Float64Array
    let sum = 0
    for (let i = 0; i < data.length; i++) sum += data[i]
    const mean = sum / data.length
    let sqsum = 0
    for (let i = 0; i < data.length; i++) sqsum += (data[i] - mean) ** 2
    score = sqsum / data.length
  } else if (method === 1) {
    const gx = ctx.track(new cv.Mat())
    const gy = ctx.track(new cv.Mat())
    cv.Sobel(gray, gx, cv.CV_64F, 1, 0, 3)
    cv.Sobel(gray, gy, cv.CV_64F, 0, 1, 3)
    const gxData = gx.data64F as Float64Array
    const gyData = gy.data64F as Float64Array
    let sum = 0
    for (let i = 0; i < gxData.length; i++) sum += gxData[i] ** 2 + gyData[i] ** 2
    score = sum / gxData.length
  } else {
    const data = gray.data as Uint8Array
    let sum = 0
    for (let i = 0; i < data.length; i++) sum += data[i]
    const mean = sum / data.length
    let sqsum = 0
    for (let i = 0; i < data.length; i++) sqsum += (data[i] - mean) ** 2
    score = sqsum / data.length / (mean + 1e-8)
  }

  const out = ctx.track(bgr.clone())
  if (params.show_score !== false) {
    putLines(cv, out, [`Focus: ${score.toFixed(2)}`], 8, 22, 0, 0.6, [160, 255, 0])
  }

  return { main: out, score }
}

// ---------------------------------------------------------------------------
// Noise Estimate
// ---------------------------------------------------------------------------
const IMMERKAER = [1, -2, 1, -2, 4, -2, 1, -2, 1]

export const sciNoiseEstimate: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, sigma: 0, snr_db: 0, gain_a: 0, read_b: 0, data: null }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, src))
  const grayF = ctx.track(new cv.Mat())
  gray.convertTo(grayF, cv.CV_32F)
  const w = grayF.cols
  const h = grayF.rows

  const kernel = cv.matFromArray(3, 3, cv.CV_32F, IMMERKAER)
  const conv = ctx.track(new cv.Mat())
  cv.filter2D(grayF, conv, cv.CV_32F, kernel, new cv.Point(-1, -1), 0, cv.BORDER_REPLICATE)
  kernel.delete()

  const convData = conv.data32F as Float32Array
  let s = 0
  for (let i = 0; i < convData.length; i++) s += Math.abs(convData[i])
  const sigma = h < 3 || w < 3 ? 0 : (s * Math.sqrt(Math.PI / 2)) / (6 * (w - 2) * (h - 2))

  const grayData = grayF.data32F as Float32Array
  let meanSum = 0
  for (let i = 0; i < grayData.length; i++) meanSum += grayData[i]
  const meanSignal = meanSum / grayData.length
  const snrDb = sigma > 1e-6 ? 10 * Math.log10(meanSignal ** 2 / sigma ** 2) : 0

  const window = Math.round(Number(params.window) || 7) | 1
  const excludeSat = params.exclude_sat !== false
  const meanMat = ctx.track(new cv.Mat())
  cv.blur(grayF, meanMat, new cv.Size(window, window))
  const sq = ctx.track(new cv.Mat())
  cv.multiply(grayF, grayF, sq)
  const meanSqMat = ctx.track(new cv.Mat())
  cv.blur(sq, meanSqMat, new cv.Size(window, window))

  const meanData = meanMat.data32F as Float32Array
  const meanSqData = meanSqMat.data32F as Float32Array
  const ms: number[] = []
  const vs: number[] = []
  for (let i = 0; i < meanData.length; i++) {
    const m = meanData[i]
    const v = Math.max(0, meanSqData[i] - m * m)
    if (excludeSat && (m <= 1 || m >= 254)) continue
    ms.push(m)
    vs.push(v)
  }

  let gainA = 0
  let readB = 0
  if (ms.length >= 32) {
    let mMin = Infinity, mMax = -Infinity
    for (const m of ms) { if (m < mMin) mMin = m; if (m > mMax) mMax = m }
    const nBins = 24
    const binSums = new Array(nBins).fill(0).map(() => [] as number[])
    for (let i = 0; i < ms.length; i++) {
      const bin = Math.min(nBins - 1, Math.max(0, Math.floor(((ms[i] - mMin) / (mMax - mMin || 1)) * nBins)))
      binSums[bin].push(vs[i])
    }
    const xs: number[] = []
    const ys: number[] = []
    for (let b = 0; b < nBins; b++) {
      if (binSums[b].length >= 8) {
        const sorted = binSums[b].slice().sort((a, c) => a - c)
        xs.push(binSums[b].reduce((a, c) => a + c, 0) / binSums[b].length)
        ys.push(sorted[Math.floor(sorted.length / 2)])
      }
    }
    if (xs.length >= 2) {
      const n = xs.length
      const sumX = xs.reduce((a, b2) => a + b2, 0)
      const sumY = ys.reduce((a, b2) => a + b2, 0)
      const sumXY = xs.reduce((a, x, i) => a + x * ys[i], 0)
      const sumXX = xs.reduce((a, x) => a + x * x, 0)
      const denom = n * sumXX - sumX * sumX
      if (Math.abs(denom) > 1e-9) {
        gainA = (n * sumXY - sumX * sumY) / denom
        readB = (sumY - gainA * sumX) / n
      }
    } else {
      const sorted = vs.slice().sort((a, b2) => a - b2)
      readB = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0
    }
  }

  const sigmaR = Number(sigma.toFixed(3))
  const snrR = Number(snrDb.toFixed(2))
  const gainR = Number(gainA.toFixed(4))
  const readR = Number(readB.toFixed(3))

  const base = ctx.track(toBgr(cv, src))
  if (params.show_overlay !== false) {
    putLines(cv, base, [`sigma=${sigmaR}`, `SNR=${snrR} dB`, `a=${gainR}  b=${readR}`], 8, 22, 24, 0.55, [0, 255, 0])
  }

  return {
    main: base,
    sigma: sigmaR,
    snr_db: snrR,
    gain_a: gainR,
    read_b: readR,
    data: { sigma: sigmaR, snr_db: snrR, gain_a: gainR, read_b: readR, mean: Number(meanSignal.toFixed(2)) },
  }
}

// ---------------------------------------------------------------------------
// Matrix Distribution
// ---------------------------------------------------------------------------
export const sciMatrixDist: NodeImpl = (inputs, params, ctx) => {
  const raw = inputs.data
  let flat: Float32Array | null = null
  if (raw && typeof raw === 'object' && 'cols' in (raw as any)) {
    const mat = raw as any
    flat = mat.channels() === 1 ? Float32Array.from(mat.data as Uint8Array) : null
  } else if (Array.isArray(raw)) {
    flat = Float32Array.from(raw.flat(Infinity) as number[])
  } else if (raw instanceof Float32Array) {
    flat = raw
  }
  if (!flat) return { main: null, bins: null, counts: null, stats: null }

  const valid = Array.from(flat).filter((v) => Number.isFinite(v))
  if (valid.length === 0) return { main: null, bins: null, counts: null, stats: null }

  const nBins = Math.max(8, Math.round(Number(params.bins) || 64))
  const logScale = !!params.log_scale
  const cumulative = !!params.cumulative

  let mMin = Infinity, mMax = -Infinity, sum = 0
  for (const v of valid) {
    if (v < mMin) mMin = v
    if (v > mMax) mMax = v
    sum += v
  }
  const mean = sum / valid.length
  let sqsum = 0
  for (const v of valid) sqsum += (v - mean) ** 2
  const std = Math.sqrt(sqsum / valid.length)
  const mMaxAdj = mMax <= mMin ? mMin + 1 : mMax

  const hist = new Array(nBins).fill(0)
  const binCenters: number[] = []
  const binWidth = (mMaxAdj - mMin) / nBins
  for (let i = 0; i < nBins; i++) binCenters.push(mMin + binWidth * (i + 0.5))
  for (const v of valid) {
    const bin = Math.min(nBins - 1, Math.max(0, Math.floor((v - mMin) / binWidth)))
    hist[bin]++
  }
  let histOut = hist
  if (cumulative) {
    let acc = 0
    histOut = hist.map((v) => (acc += v))
  }
  if (logScale) histOut = histOut.map((v) => Math.log10(v + 1))

  const cv = ctx.cv
  const cw = 400
  const ch = 200
  const chart = ctx.track(new cv.Mat(ch, cw, cv.CV_8UC3, new cv.Scalar(18, 18, 18)))
  for (let i = 1; i < 4; i++) {
    cv.line(chart, new cv.Point(Math.round((cw * i) / 4), 0), new cv.Point(Math.round((cw * i) / 4), ch), new cv.Scalar(40, 40, 40, 255), 1)
    cv.line(chart, new cv.Point(0, Math.round((ch * i) / 4)), new cv.Point(cw, Math.round((ch * i) / 4)), new cv.Scalar(40, 40, 40, 255), 1)
  }
  const hMax = Math.max(...histOut, 1e-9)
  const pad = 8
  const areaW = cw - 2 * pad
  const areaH = ch - 2 * pad
  const barW = Math.max(1, Math.floor(areaW / nBins))
  const [b, g, r] = hexToBgr(String(params.bar_color ?? '#00d4aa'), [170, 212, 0])
  const barColor = new cv.Scalar(b, g, r, 255)
  for (let i = 0; i < nBins; i++) {
    const bh = Math.round((histOut[i] / hMax) * areaH)
    const x0 = pad + Math.round((i / nBins) * areaW)
    const y0 = ch - pad - bh
    cv.rectangle(chart, new cv.Point(x0, y0), new cv.Point(x0 + barW, ch - pad), barColor, -1)
  }
  if (params.show_stats !== false) {
    putLines(cv, chart, [`mean=${mean.toFixed(4)}  std=${std.toFixed(4)}`, `min=${mMin.toFixed(4)}  max=${mMaxAdj.toFixed(4)}  n=${valid.length}`], 8, 14, 14, 0.38, [180, 180, 180])
  }

  return {
    main: chart,
    bins: binCenters,
    counts: histOut,
    hist_0: histOut,
    stats: { mean, std, min: mMin, max: mMaxAdj, count: valid.length },
    hist_min: mMin,
    hist_max: mMaxAdj,
  }
}

// ---------------------------------------------------------------------------
// Colormap / LUT
// ---------------------------------------------------------------------------
const SCI_COLORMAP_NAMES = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Turbo', 'Jet', 'Hot', 'Cool', 'Parula', 'Cividis', 'Rainbow', 'Ocean']

export const sciColormap: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const name = SCI_COLORMAP_NAMES[Number(params.colormap) || 0] ?? 'Viridis'
  const fn = COLORMAPS[name] ?? COLORMAPS.Viridis
  const auto = params.auto_range !== false
  const invert = !!params.invert

  const gray = ctx.track(toGray(cv, src))
  const data = gray.data as Uint8Array

  let lo: number, hi: number
  if (auto) {
    lo = Infinity
    hi = -Infinity
    for (let i = 0; i < data.length; i++) {
      if (data[i] < lo) lo = data[i]
      if (data[i] > hi) hi = data[i]
    }
  } else {
    lo = Number(params.in_min) || 0
    hi = params.in_max === undefined ? 255 : Number(params.in_max)
  }
  if (hi <= lo) hi = lo + 1

  const norm = ctx.track(new cv.Mat(gray.rows, gray.cols, cv.CV_8U))
  const normData = norm.data as Uint8Array
  for (let i = 0; i < data.length; i++) {
    let v = Math.max(0, Math.min(255, Math.round(((data[i] - lo) / (hi - lo)) * 255)))
    if (invert) v = 255 - v
    normData[i] = v
  }

  return { main: ctx.track(applyColormap(cv, norm, fn)) }
}

// ---------------------------------------------------------------------------
// Value Gate (Range Checker)
// ---------------------------------------------------------------------------
const RANGE_COLORS = {
  ok: [80, 200, 100] as [number, number, number],
  high: [60, 80, 220] as [number, number, number],
  low: [200, 160, 50] as [number, number, number],
  crit: [50, 50, 230] as [number, number, number],
  dim: [130, 140, 155] as [number, number, number],
  white: [225, 228, 232] as [number, number, number],
}

function rangeStatus(val: number, lo: number, hi: number, warnLo: number, warnHi: number): { label: string; color: [number, number, number] } {
  if (val < lo || val > hi) return { label: 'CRITICAL', color: RANGE_COLORS.crit }
  if (val < warnLo || val > warnHi) return val < warnLo ? { label: 'LOW', color: RANGE_COLORS.low } : { label: 'HIGH', color: RANGE_COLORS.high }
  return { label: 'OK', color: RANGE_COLORS.ok }
}

export const sciRangeChecker: NodeImpl = (inputs, params, ctx) => {
  const values = (inputs.values as Record<string, unknown>) ?? {}
  const cv = ctx.cv
  const accent = hexToBgr(String(params.accent_color ?? '#4ade80'), [128, 222, 74]).reverse() as [number, number, number]
  const title = String(params.title ?? 'Reference Range Check')

  let ranges: Record<string, [number, number, number, number]> = {}
  try {
    ranges = JSON.parse(String(params.ranges ?? '{}'))
  } catch {
    ranges = {}
  }

  const statusOut: Record<string, string> = {}
  const flags: Record<string, unknown>[] = []
  const keys = Object.keys(values).filter((k) => k !== 'total')
  for (const key of keys) {
    const fval = Number(values[key])
    if (!Number.isFinite(fval)) continue
    const rng = ranges[key]
    let label = '—'
    if (rng && rng.length === 4) {
      label = rangeStatus(fval, rng[0], rng[3], rng[1], rng[2]).label
    }
    statusOut[key] = label
    if (label !== 'OK' && label !== '—') flags.push({ key, value: fval, status: label })
  }

  const IW = 700
  const IH = 500
  const img = ctx.track(new cv.Mat(IH, IW, cv.CV_8UC3, new cv.Scalar(38, 28, 22)))
  const accentColor = new cv.Scalar(accent[0], accent[1], accent[2], 255)

  cv.rectangle(img, new cv.Point(0, 0), new cv.Point(IW, 48), new cv.Scalar(26, 18, 14, 255), -1)
  cv.line(img, new cv.Point(0, 47), new cv.Point(IW, 47), accentColor, 2)
  putLines(cv, img, [title], 10, 30, 0, 0.55, accent)
  putLines(cv, img, [new Date().toISOString().slice(0, 10)], IW - 110, 30, 0, 0.38, RANGE_COLORS.dim)

  const rowH = 44
  let ry = 56
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i]
    const fval = Number(values[key])
    if (!Number.isFinite(fval)) continue
    const bg = i % 2 === 0 ? new cv.Scalar(58, 44, 34, 255) : new cv.Scalar(68, 52, 40, 255)
    cv.rectangle(img, new cv.Point(4, ry), new cv.Point(IW - 4, ry + rowH), bg, -1)
    cv.line(img, new cv.Point(4, ry + rowH), new cv.Point(IW - 4, ry + rowH), new cv.Scalar(88, 66, 52, 255), 1)

    const rng = ranges[key]
    let label: string
    let color: [number, number, number]
    let rangeStr: string
    if (rng && rng.length === 4) {
      const st = rangeStatus(fval, rng[0], rng[3], rng[1], rng[2])
      label = st.label
      color = st.color
      rangeStr = `[${rng[1]} - ${rng[2]}]`
    } else {
      label = '—'
      color = RANGE_COLORS.dim
      rangeStr = 'no range defined'
    }
    const valStr = Number.isInteger(fval) ? String(fval) : fval.toFixed(2)
    putLines(cv, img, [key], 12, ry + 26, 0, 0.5, accent)
    putLines(cv, img, [valStr], 200, ry + 26, 0, 0.52, RANGE_COLORS.white)
    putLines(cv, img, [rangeStr], 320, ry + 26, 0, 0.38, RANGE_COLORS.dim)
    putLines(cv, img, [label], IW - 110, ry + 26, 0, 0.48, color)

    ry += rowH
    if (ry + rowH > IH - 10) break
  }

  const nFlags = flags.length
  const badgeColor = nFlags === 0 ? RANGE_COLORS.ok : nFlags > 2 ? RANGE_COLORS.high : RANGE_COLORS.low
  const badgeTxt = nFlags === 0 ? 'ALL OK' : `${nFlags} FLAG${nFlags > 1 ? 'S' : ''}`
  cv.rectangle(img, new cv.Point(IW - 145, IH - 38), new cv.Point(IW - 5, IH - 5), new cv.Scalar(badgeColor[0], badgeColor[1], badgeColor[2], 255), -1)
  putLines(cv, img, [badgeTxt], IW - 135, IH - 14, 0, 0.5, [10, 20, 10])
  cv.rectangle(img, new cv.Point(0, 0), new cv.Point(IW - 1, IH - 1), accentColor, 2)

  return { status: statusOut, flags, main: img }
}

// ---------------------------------------------------------------------------
// Region Color Stats
// ---------------------------------------------------------------------------
export const sciRegionColorStats: NodeImpl = (inputs, params, ctx) => {
  const img = inputs.image as any
  const labels = inputs.labels_map as any
  if (!img || !labels) return { regions: inputs.regions_in ?? [], count: 0, main: img ?? null }
  const cv = ctx.cv

  const src = ctx.track(toBgr(cv, img))
  let labelData = labels.data32S as Int32Array
  let lw = labels.cols
  let lh = labels.rows
  if (lw !== src.cols || lh !== src.rows) {
    const resized = ctx.track(new cv.Mat())
    cv.resize(labels, resized, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_NEAREST)
    labelData = resized.data32S as Int32Array
    lw = resized.cols
    lh = resized.rows
  }

  const colorspace = Number(params.colorspace) ?? 1
  const doBgr = colorspace === 0 || colorspace === 1
  const doHsv = colorspace === 1 || colorspace === 2
  const showIds = !!params.show_ids

  let hsvData: Uint8Array | null = null
  if (doHsv) {
    const hsv = ctx.track(new cv.Mat())
    cv.cvtColor(src, hsv, cv.COLOR_BGR2HSV)
    hsvData = hsv.data as Uint8Array
  }

  const srcData = src.data as Uint8Array
  const existing = new Map<number, Record<string, unknown>>()
  for (const r of (inputs.regions_in as Record<string, unknown>[]) ?? []) {
    if (r && typeof r.label_id === 'number') existing.set(r.label_id, r)
  }

  const acc = new Map<number, { sb: number; sg: number; sr: number; sb2: number; sg2: number; sr2: number; sh: number[]; ss: number; sv: number; n: number; sx: number; sy: number }>()
  for (let y = 0; y < lh; y++) {
    for (let x = 0; x < lw; x++) {
      const idx = y * lw + x
      const lid = labelData[idx]
      if (lid <= 0) continue
      let a = acc.get(lid)
      if (!a) {
        a = { sb: 0, sg: 0, sr: 0, sb2: 0, sg2: 0, sr2: 0, sh: [], ss: 0, sv: 0, n: 0, sx: 0, sy: 0 }
        acc.set(lid, a)
      }
      const p = idx * 3
      a.sb += srcData[p]
      a.sg += srcData[p + 1]
      a.sr += srcData[p + 2]
      a.sb2 += srcData[p] * srcData[p]
      a.sg2 += srcData[p + 1] * srcData[p + 1]
      a.sr2 += srcData[p + 2] * srcData[p + 2]
      if (hsvData) {
        a.sh.push(hsvData[p])
        a.ss += hsvData[p + 1]
        a.sv += hsvData[p + 2]
      }
      a.n++
      a.sx += x
      a.sy += y
    }
  }

  const regions: Record<string, unknown>[] = []
  for (const [lid, a] of acc) {
    const r: Record<string, unknown> = { ...(existing.get(lid) ?? {}), label_id: lid }
    if (doBgr) {
      const meanB = a.sb / a.n, meanG = a.sg / a.n, meanR = a.sr / a.n
      r.mean_b = round2(meanB)
      r.mean_g = round2(meanG)
      r.mean_r = round2(meanR)
      r.std_b = round2(Math.sqrt(Math.max(0, a.sb2 / a.n - meanB * meanB)))
      r.std_g = round2(Math.sqrt(Math.max(0, a.sg2 / a.n - meanG * meanG)))
      r.std_r = round2(Math.sqrt(Math.max(0, a.sr2 / a.n - meanR * meanR)))
    }
    if (doHsv) {
      r.mean_h = round2(a.sh.reduce((s, v) => s + v, 0) / a.n)
      r.mean_s = round2(a.ss / a.n)
      r.mean_v = round2(a.sv / a.n)
      const sorted = a.sh.slice().sort((x, y) => x - y)
      r.dominant_hue = round2(sorted[Math.floor(sorted.length / 2)] ?? 0)
    }
    regions.push(r)
  }

  const preview = ctx.track(src.clone())
  for (const [lid, a] of acc) {
    const r = regions.find((rr) => rr.label_id === lid)!
    const cx = Math.round(a.sx / a.n)
    const cy = Math.round(a.sy / a.n)
    const color = new cv.Scalar(Number(r.mean_b ?? 128), Number(r.mean_g ?? 128), Number(r.mean_r ?? 128), 255)
    cv.circle(preview, new cv.Point(cx, cy), 4, color, -1)
    cv.circle(preview, new cv.Point(cx, cy), 5, new cv.Scalar(255, 255, 255, 255), 1)
    if (showIds) cv.putText(preview, String(lid), new cv.Point(cx + 6, cy + 4), cv.FONT_HERSHEY_SIMPLEX, 0.3, new cv.Scalar(255, 255, 255, 255), 1, cv.LINE_AA)
  }
  putLines(cv, preview, [`n=${regions.length}`], 6, 18, 0, 0.55, [255, 255, 255])

  return { regions, count: regions.length, main: preview }
}

function round2(v: number): number {
  return Number(v.toFixed(2))
}

// ---------------------------------------------------------------------------
// Frame Accumulator
// ---------------------------------------------------------------------------
interface AccumState {
  count: number
  mean: Float32Array | null
  m2: Float32Array | null
  max: Float32Array | null
  min: Float32Array | null
  prev: Float32Array | null
  diff: Float32Array | null
  buffer: Float32Array[]
  lastReset: number
  lastTick: number | null
  shape: [number, number, number] | null
}

function resetAccum(s: AccumState): void {
  s.count = 0
  s.mean = s.m2 = s.max = s.min = s.prev = s.diff = null
  s.buffer = []
}

export const sciFrameAccumulator: NodeImpl = (inputs, params, ctx) => {
  let state: AccumState = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { count: 0, mean: null, m2: null, max: null, min: null, prev: null, diff: null, buffer: [], lastReset: 0, lastTick: null, shape: null }
    ctx.state.set(ctx.nodeId, state)
  }

  const doReset = Number(params.reset) || 0
  if (doReset > 0.5 && state.lastReset <= 0.5) resetAccum(state)
  state.lastReset = doReset

  const tick = inputs.tick
  if (typeof tick === 'number') {
    if (state.lastTick !== null && tick < state.lastTick) resetAccum(state)
    state.lastTick = tick
  }

  const img = inputs.image as any
  if (!img) return { main: null, frame_count: state.count || state.buffer.length, done: 0 }
  const cv = ctx.cv

  const mode = Number(params.mode) || 0
  const cumulative = params.cumulative !== false

  const bgr = toBgr(cv, img)
  const f = Float32Array.from(bgr.data as Uint8Array)
  const shape: [number, number, number] = [bgr.rows, bgr.cols, bgr.channels()]
  bgr.delete()

  if (cumulative) {
    const targetN = Math.round(Number(params.target_n) || 0)
    let reached = targetN > 0 && state.count >= targetN
    if (!reached) {
      state.count++
      if (!state.mean) {
        state.mean = f.slice()
        state.m2 = new Float32Array(f.length)
        state.max = f.slice()
        state.min = f.slice()
        state.shape = shape
      } else {
        const mean = state.mean!
        const m2 = state.m2!
        const max = state.max!
        const min = state.min!
        for (let i = 0; i < f.length; i++) {
          const delta = f[i] - mean[i]
          mean[i] += delta / state.count
          m2[i] += delta * (f[i] - mean[i])
          if (f[i] > max[i]) max[i] = f[i]
          if (f[i] < min[i]) min[i] = f[i]
        }
      }
      state.diff = state.prev ? f.map((v, i) => Math.abs(v - state.prev![i]) * 4) : f.slice()
      state.prev = f
      reached = targetN > 0 && state.count >= targetN
    }

    if (!state.mean || !state.shape) return { main: null, frame_count: 0, done: 0 }
    let result: Float32Array
    if (mode === 0) result = state.mean
    else if (mode === 1) result = state.max!
    else if (mode === 2) result = state.min!
    else if (mode === 3) {
      const std = state.m2!.map((v) => Math.sqrt(v / Math.max(state.count, 1)))
      const m = Math.max(...std)
      result = m > 0 ? std.map((v) => (v / (m + 1e-8)) * 255) : std
    } else result = state.diff!

    const [h, w, c] = state.shape
    const out = ctx.track(new cv.Mat(h, w, c === 1 ? cv.CV_8UC1 : cv.CV_8UC3))
    const outData = out.data as Uint8Array
    for (let i = 0; i < result.length; i++) outData[i] = Math.max(0, Math.min(255, Math.round(result[i])))
    return { main: out, frame_count: state.count, done: reached ? 1 : 0 }
  }

  const window = Math.round(Number(params.window) || 16)
  state.buffer.push(f)
  state.shape = shape
  if (state.buffer.length > window) state.buffer.shift()

  const n = state.buffer.length
  const len = f.length
  let result: Float32Array
  if (mode === 0) {
    result = new Float32Array(len)
    for (const fr of state.buffer) for (let i = 0; i < len; i++) result[i] += fr[i] / n
  } else if (mode === 1) {
    result = state.buffer[0].slice()
    for (const fr of state.buffer) for (let i = 0; i < len; i++) if (fr[i] > result[i]) result[i] = fr[i]
  } else if (mode === 2) {
    result = state.buffer[0].slice()
    for (const fr of state.buffer) for (let i = 0; i < len; i++) if (fr[i] < result[i]) result[i] = fr[i]
  } else if (mode === 3) {
    const mean = new Float32Array(len)
    for (const fr of state.buffer) for (let i = 0; i < len; i++) mean[i] += fr[i] / n
    const std = new Float32Array(len)
    for (const fr of state.buffer) for (let i = 0; i < len; i++) std[i] += (fr[i] - mean[i]) ** 2 / n
    for (let i = 0; i < len; i++) std[i] = Math.sqrt(std[i])
    const m = Math.max(...std)
    result = m > 0 ? std.map((v) => (v / (m + 1e-8)) * 255) : std
  } else {
    result = n >= 2 ? state.buffer[n - 1].map((v, i) => Math.abs(v - state.buffer[n - 2][i]) * 4) : state.buffer[0].slice()
  }

  const [h, w, c] = state.shape
  const out = ctx.track(new cv.Mat(h, w, c === 1 ? cv.CV_8UC1 : cv.CV_8UC3))
  const outData = out.data as Uint8Array
  for (let i = 0; i < result.length; i++) outData[i] = Math.max(0, Math.min(255, Math.round(result[i])))
  return { main: out, frame_count: n, done: 0 }
}
