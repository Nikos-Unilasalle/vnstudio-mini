/**
 * Image-level analysis nodes ported from the desktop plugins
 * filter_local_stats.py, filter_dog.py, filter_auto_polarity.py and
 * sci_scale_bar_detect.py. Each mirrors its Python counterpart step for step.
 *
 * Typed-array views of a Mat (`mat.data`, `data32F`…) point into the WASM heap and
 * become stale when the heap grows, which any new Mat can trigger. Pixels that are
 * read after another allocation are therefore copied (`.slice()`), and write views
 * are only taken once every output Mat exists.
 */
import type { NodeImpl } from '../types'
import { toGray } from '../cvUtils'

function grayFloat(cv: any, ctx: any, src: any): any {
  const gray = ctx.track(toGray(cv, src))
  const f = ctx.track(new cv.Mat())
  gray.convertTo(f, cv.CV_32F)
  return f
}

/**
 * Min-max stretch of a float Mat to 0–255, truncating like the desktop's
 * `cv2.normalize(...).astype(np.uint8)` (convertTo would round instead, and half a
 * grey level is enough to move a downstream Otsu threshold).
 */
function normalizeU8(cv: any, ctx: any, src: any): any {
  const out = ctx.track(new cv.Mat(src.rows, src.cols, cv.CV_8U))
  const v = src.data32F as Float32Array
  let lo = Infinity, hi = -Infinity
  for (let i = 0; i < v.length; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i] }
  const scale = hi > lo ? 255 / (hi - lo) : 0
  const o = out.data as Uint8Array
  for (let i = 0; i < v.length; i++) o[i] = Math.min(255, Math.floor(Math.fround((v[i] - lo) * scale)))
  return out
}

// ── Local Statistics ─────────────────────────────────────────────────────────

export const filterLocalStats: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, raw: null }
  const cv = ctx.cv
  const g = grayFloat(cv, ctx, src)
  const win = Math.max(3, Math.round(Number(params.window) || 15))
  const stat = Number(params.statistic ?? 1)
  const size = new cv.Size(win, win)

  const mean = ctx.track(new cv.Mat())
  cv.blur(g, mean, size)
  if (stat === 0) return { main: normalizeU8(cv, ctx, mean), raw: mean }

  const sq = ctx.track(new cv.Mat())
  cv.multiply(g, g, sq)
  const meanSq = ctx.track(new cv.Mat())
  cv.blur(sq, meanSq, size)
  // Views into the WASM heap go stale when it grows: allocate first, then take them.
  const raw = ctx.track(new cv.Mat(g.rows, g.cols, cv.CV_32F))
  const m = mean.data32F as Float32Array
  const ms = meanSq.data32F as Float32Array
  const r = raw.data32F as Float32Array
  for (let i = 0; i < r.length; i++) {
    const v = Math.max(ms[i] - m[i] * m[i], 0)
    r[i] = stat === 2 ? v : Math.sqrt(v)
  }
  return { main: normalizeU8(cv, ctx, raw), raw }
}

// ── Difference of Gaussians ──────────────────────────────────────────────────

const MIN_SIGMA = 0.3
const MIN_RATIO = 1.05

export const filterDog: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, raw: null }
  const cv = ctx.cv
  const g = grayFloat(cv, ctx, src)
  const sigma = Math.max(MIN_SIGMA, Number(params.sigma) || 3)
  const ratio = Math.max(MIN_RATIO, Number(params.ratio) || 2)
  const signed = Number(params.output) === 1

  const a = ctx.track(new cv.Mat())
  const b = ctx.track(new cv.Mat())
  cv.GaussianBlur(g, a, new cv.Size(0, 0), sigma, sigma, cv.BORDER_DEFAULT)
  cv.GaussianBlur(g, b, new cv.Size(0, 0), sigma * ratio, sigma * ratio, cv.BORDER_DEFAULT)
  const dog = ctx.track(new cv.Mat())
  cv.subtract(a, b, dog)

  if (signed) {
    const main = ctx.track(new cv.Mat(dog.rows, dog.cols, cv.CV_8U))
    const d = dog.data32F as Float32Array
    let peak = 0
    for (let i = 0; i < d.length; i++) peak = Math.max(peak, Math.abs(d[i]))
    const scale = peak > 0 ? 127 / peak : 0
    const o = main.data as Uint8Array
    for (let i = 0; i < d.length; i++) o[i] = Math.max(0, Math.min(255, Math.round(d[i] * scale + 128)))
    return { main, raw: dog }
  }
  const energy = ctx.track(new cv.Mat(dog.rows, dog.cols, cv.CV_32F))
  const d = dog.data32F as Float32Array
  const e = energy.data32F as Float32Array
  for (let i = 0; i < d.length; i++) e[i] = Math.abs(d[i])
  return { main: normalizeU8(cv, ctx, energy), raw: energy }
}

// ── Auto Polarity ────────────────────────────────────────────────────────────

const PEAK_SMOOTH_SIGMA = 4

function backgroundIsBrightByPeak(cv: any, ctx: any, gray: any): boolean {
  const smooth = ctx.track(new cv.Mat())
  cv.GaussianBlur(gray, smooth, new cv.Size(0, 0), PEAK_SMOOTH_SIGMA, PEAK_SMOOTH_SIGMA, cv.BORDER_DEFAULT)
  const s = smooth.data as Uint8Array
  const hist = new Float64Array(256)
  let sum = 0
  for (let i = 0; i < s.length; i++) {
    hist[s[i]]++
    sum += s[i]
  }
  let peak = 0
  for (let v = 1; v < 256; v++) if (hist[v] > hist[peak]) peak = v
  return peak >= sum / s.length
}

export const filterAutoPolarity: NodeImpl = (inputs, params, ctx) => {
  const img = inputs.image as any
  if (!img) return { main: null, inverted: false }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, img))
  const g = (gray.data as Uint8Array).slice()   // copy: later allocations can move the heap
  const mask = inputs.mask as any
  const h = gray.rows
  const w = gray.cols

  let bgIsBright: boolean
  if (!mask && Number(params.method) === 1) {
    bgIsBright = backgroundIsBrightByPeak(cv, ctx, gray)
  } else {
    let bgSum = 0, bgN = 0, fgSum = 0, fgN = 0
    if (mask) {
      let m = ctx.track(toGray(cv, mask))
      if (m.rows !== h || m.cols !== w) {
        const r = ctx.track(new cv.Mat())
        cv.resize(m, r, new cv.Size(w, h), 0, 0, cv.INTER_NEAREST)
        m = r
      }
      const md = (m.data as Uint8Array).slice()
      for (let i = 0; i < g.length; i++) {
        if (md[i] > 0) { fgSum += g[i]; fgN++ } else { bgSum += g[i]; bgN++ }
      }
    } else {
      const b = Math.max(1, Math.round((Math.min(h, w) * (Number(params.border_pct) || 5)) / 100))
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const v = g[y * w + x]
          if (y < b || y >= h - b || x < b || x >= w - b) { bgSum += v; bgN++ } else { fgSum += v; fgN++ }
        }
      }
    }
    if (bgN === 0 || fgN === 0) return { main: img, inverted: false }
    bgIsBright = bgSum / bgN >= fgSum / fgN
  }

  const wantBright = Number(params.target ?? 0) === 0
  const invert = bgIsBright !== wantBright
  if (!invert) return { main: img, inverted: false }
  const out = ctx.track(new cv.Mat())
  cv.bitwise_not(img, out)
  return { main: out, inverted: true }
}

// ── Scale Bar Detect ─────────────────────────────────────────────────────────

const UNIT_TO_MM = [1e-3, 1.0, 1e-6, 10.0]
const UNIT_NAMES = ['um', 'mm', 'nm', 'cm']
const MIN_LEN_FRAC = 0.04
const MAX_THICK_FRAC = 0.015
const MIN_FILL = 0.8
const MIN_CONTRAST = 25
const MIN_ASPECT = 12
const MIN_LEN_PX = 30
const MAX_BAR_STD = 30
const MIN_MANUAL_PX_PER_MM = 1.5

interface Bar { x: number; y: number; w: number; h: number }

/** Pixels brighter than their vertical neighbour (d rows away) on at least one side. */
function edgeContrast(cv: any, ctx: any, g: Uint8Array, h: number, w: number, d: number): any {
  const out = ctx.track(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0)))
  const o = out.data as Uint8Array
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      const above = y >= d ? g[i] - g[i - d * w] : 0
      const below = y + d < h ? g[i] - g[i + d * w] : 0
      if (Math.max(above, below) >= MIN_CONTRAST) o[i] = 1
    }
  }
  return out
}

function centreLineStd(g: Uint8Array, w: number, bar: Bar): number {
  const y = bar.y + Math.floor(bar.h / 2)
  let s = 0, s2 = 0
  for (let x = bar.x; x < bar.x + bar.w; x++) {
    const v = g[y * w + x]
    s += v
    s2 += v * v
  }
  const n = bar.w
  return Math.sqrt(Math.max(s2 / n - (s / n) ** 2, 0))
}

function bestBar(cv: any, ctx: any, g: Uint8Array, h: number, w: number, minLen: number, maxThick: number): Bar | null {
  const shift = Math.max(2, Math.floor(maxThick / 3))
  const contrast = edgeContrast(cv, ctx, g, h, w, shift)
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(minLen, 1))
  const runs = ctx.track(new cv.Mat())
  cv.morphologyEx(contrast, runs, cv.MORPH_OPEN, kernel)
  kernel.delete()
  const labels = ctx.track(new cv.Mat())
  const stats = ctx.track(new cv.Mat())
  const centroids = ctx.track(new cv.Mat())
  const n = cv.connectedComponentsWithStats(runs, labels, stats, centroids, 8, cv.CV_32S)
  const st = (stats.data32S as Int32Array).slice()
  let best: Bar | null = null
  for (let i = 1; i < n; i++) {
    const bar = { x: st[i * 5], y: st[i * 5 + 1], w: st[i * 5 + 2], h: st[i * 5 + 3] }
    const area = st[i * 5 + 4]
    if (bar.w < minLen || bar.h > maxThick || bar.w < MIN_ASPECT * Math.max(bar.h, 1)) continue
    if (area < MIN_FILL * bar.w * bar.h) continue
    if (centreLineStd(g, w, bar) > MAX_BAR_STD) continue
    if (!best || bar.w > best.w) best = bar
  }
  return best
}

function manualCalibration(value: unknown): number | null {
  const v = Number(value)
  return Number.isFinite(v) && v >= MIN_MANUAL_PX_PER_MM ? v : null
}

export const sciScaleBarDetect: NodeImpl = (inputs, params, ctx) => {
  const img = inputs.image as any
  const empty = { px_per_mm: null, px_per_um: null, length_px: null, bar_mask: null, main: img ?? null }
  const manual = manualCalibration(inputs.manual_px_per_mm)
  if (Number(params.source) === 1 && manual !== null) {
    return { ...empty, px_per_mm: manual, px_per_um: manual / 1000 }
  }
  if (!img) return empty
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, img))
  const h = gray.rows
  const w = gray.cols
  const minLen = Math.max(MIN_LEN_PX, Math.floor(w * MIN_LEN_FRAC))
  const maxThick = Math.max(3, Math.floor(h * MAX_THICK_FRAC))
  const g = (gray.data as Uint8Array).slice()   // copy: later allocations can move the heap
  const inverted = new Uint8Array(g.length)
  for (let i = 0; i < g.length; i++) inverted[i] = 255 - g[i]

  // Bright bars first (most common); dark bars only if no bright bar is found.
  const bar = bestBar(cv, ctx, g, h, w, minLen, maxThick) ?? bestBar(cv, ctx, inverted, h, w, minLen, maxThick)
  const barLen = Number(params.bar_length) || 0
  const unit = Number(params.unit) || 0
  if (!bar || barLen <= 0) {
    const fallback = Number(params.fallback_px_per_mm) || 0
    return fallback > 0 ? { ...empty, px_per_mm: fallback, px_per_um: fallback / 1000 } : empty
  }

  const pxPerMm = bar.w / (barLen * UNIT_TO_MM[unit % UNIT_TO_MM.length])
  const barMask = ctx.track(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0)))
  cv.rectangle(barMask, new cv.Point(bar.x, bar.y), new cv.Point(bar.x + bar.w - 1, bar.y + bar.h - 1), new cv.Scalar(255), -1)
  const vis = ctx.track(new cv.Mat())
  cv.cvtColor(gray, vis, cv.COLOR_GRAY2BGR)
  cv.rectangle(vis, new cv.Point(bar.x, bar.y - 4), new cv.Point(bar.x + bar.w, bar.y + bar.h + 4), new cv.Scalar(0, 0, 255), 2)
  cv.putText(vis, `${bar.w} px = ${barLen} ${UNIT_NAMES[unit % UNIT_NAMES.length]}`,
    new cv.Point(Math.max(0, bar.x), Math.max(15, bar.y - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, new cv.Scalar(0, 0, 255), 2)
  return { px_per_mm: pxPerMm, px_per_um: pxPerMm / 1000, length_px: bar.w, bar_mask: barMask, main: vis }
}
