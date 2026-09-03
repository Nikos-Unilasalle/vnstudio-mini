import type { NodeImpl, RunContext } from '../types'

/**
 * Signal filtering nodes — ported from the desktop app's signal_filters.py.
 *
 * Two modes, one set of nodes:
 *
 * - `value` input (scalar) — streams frame by frame. The filter only knows the
 *   past, so its output lags the signal (half the window for kernel filters,
 *   more for recursive ones).
 * - `list` input — filters an already-recorded series, CENTERED: every point is
 *   estimated from neighbours on both sides, so there's no lag. Recursive
 *   filters (no centered form) run forward then backward — the two lags cancel.
 *
 * When `list` is wired it takes priority and its output carries the smoothed
 * series.
 */

function toScalar(v: unknown, fallback = 0): number {
  if (v === null || v === undefined) return fallback
  if (typeof v === 'number') return Number.isFinite(v) ? v : fallback
  if (Array.isArray(v)) return v.length > 0 ? toScalar(v[0], fallback) : fallback
  if (typeof v === 'object') {
    const dict = v as Record<string, unknown>
    for (const key of ['value', 'scalar', 'result', 'filtered', 'raw']) {
      if (key in dict) return toScalar(dict[key], fallback)
    }
    return fallback
  }
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}

function toSerie(v: unknown): number[] | null {
  if (v === null || v === undefined) return null
  if (!Array.isArray(v)) return null
  const arr = v.map((x) => toScalar(x, NaN))
  return arr.length >= 2 ? arr : null
}

/** Extends the series by reflection so a centered kernel doesn't dip at the edges. */
function bordered(x: number[], half: number): number[] {
  if (half <= 0) return x
  const left = x.slice(1, half + 1).reverse()
  const right = x.slice(Math.max(0, x.length - half - 1), x.length - 1).reverse()
  return [...left, ...x, ...right]
}

function convolveCentered(x: number[], kernel: number[]): number[] {
  const half = Math.floor(kernel.length / 2)
  const padded = bordered(x, half)
  const out: number[] = []
  for (let i = 0; i < x.length; i++) {
    let sum = 0
    for (let k = 0; k < kernel.length; k++) sum += padded[i + k] * kernel[k]
    out.push(sum)
  }
  return out
}

/** Makes a recursive (one-sided) filter symmetric: forward pass, then backward. */
function forwardBackward(x: number[], pass: (s: number[]) => number[]): number[] {
  return pass(pass(x).slice().reverse()).slice().reverse()
}

interface FilterState {
  buf: number[]
  [key: string]: unknown
}

function getState(ctx: RunContext): FilterState {
  let state = ctx.state.get(ctx.nodeId) as FilterState | undefined
  if (!state) {
    state = { buf: [] }
    ctx.state.set(ctx.nodeId, state)
  }
  return state
}

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length
}

/** Wires a { flux, lisser } pair into the value/list dual-mode contract. */
function filterNode(
  flux: (v: number, params: Record<string, any>, state: FilterState) => { filtered: number; raw: number; [k: string]: number },
  lisser: (x: number[], params: Record<string, any>) => number[]
): NodeImpl {
  return (inputs, params, ctx) => {
    const serie = toSerie(inputs.list)
    if (serie === null) {
      const state = getState(ctx)
      const v = toScalar(inputs.value)
      return flux(v, params, state)
    }
    const smoothed = lisser(serie, params)
    return {
      list: smoothed,
      filtered: smoothed[smoothed.length - 1],
      raw: serie[serie.length - 1],
    }
  }
}

// 1. Moving Average
export const pluginFilterMa: NodeImpl = filterNode(
  (v, params, state) => {
    const w = Math.max(2, Math.round(Number(params.window) || 15))
    state.buf.push(v)
    if (state.buf.length > w) state.buf = state.buf.slice(-w)
    return { filtered: mean(state.buf), raw: v }
  },
  (x, params) => {
    const w = Math.max(2, Math.round(Number(params.window) || 15))
    return convolveCentered(x, new Array(w).fill(1 / w))
  }
)

// 2. Exponential Moving Average
export const pluginFilterEma: NodeImpl = filterNode(
  (v, params, state) => {
    const a = (Number(params.alpha) || 20) / 100
    const prev = state.state === undefined ? v : (state.state as number)
    const next = a * v + (1 - a) * prev
    state.state = next
    return { filtered: next, raw: v }
  },
  (x, params) => {
    const a = (Number(params.alpha) || 20) / 100
    const pass = (s: number[]) => {
      const out: number[] = []
      let etat = s[0]
      for (const v of s) {
        etat = a * v + (1 - a) * etat
        out.push(etat)
      }
      return out
    }
    return forwardBackward(x, pass)
  }
)

// 3. Kalman Filter (1D constant-velocity model)
export const pluginFilterKalman: NodeImpl = filterNode(
  (v, params, state) => {
    const Q = (Number(params.q) || 1) / 1000
    const R = (Number(params.r) || 100) / 100
    if (state.x === undefined) state.x = v
    const P = state.P === undefined ? 1 : (state.P as number)
    const x = state.x as number
    const P_pred = P + Q
    const K = P_pred / (P_pred + R)
    const nextX = x + K * (v - x)
    state.x = nextX
    state.P = (1 - K) * P_pred
    return { filtered: nextX, raw: v }
  },
  (x, params) => {
    const Q = (Number(params.q) || 1) / 1000
    const R = (Number(params.r) || 100) / 100
    const pass = (s: number[]) => {
      const out: number[] = []
      let etat = s[0]
      let P = 1
      for (const z of s) {
        const P_pred = P + Q
        const K = P_pred / (P_pred + R)
        etat = etat + K * (z - etat)
        P = (1 - K) * P_pred
        out.push(etat)
      }
      return out
    }
    return forwardBackward(x, pass)
  }
)

// 4. Median Filter
export const pluginFilterMedian: NodeImpl = filterNode(
  (v, params, state) => {
    let w = Math.max(3, Math.round(Number(params.window) || 11))
    if (w % 2 === 0) w += 1
    state.buf.push(v)
    if (state.buf.length > w) state.buf = state.buf.slice(-w)
    return { filtered: median(state.buf), raw: v }
  },
  (x, params) => {
    let w = Math.max(3, Math.round(Number(params.window) || 11))
    if (w % 2 === 0) w += 1
    const half = Math.floor(w / 2)
    const padded = bordered(x, half)
    return x.map((_, i) => median(padded.slice(i, i + w)))
  }
)

// 5. Savitzky-Golay Smoothing (least-squares polynomial kernel via normal equations)
function sgCoeffs(window: number, poly: number): number[] {
  const half = Math.floor(window / 2)
  const xs: number[] = []
  for (let i = -half; i <= half; i++) xs.push(i)
  // A[i][j] = xs[i]^j — Vandermonde, increasing powers
  const A: number[][] = xs.map((xi) => {
    const row: number[] = []
    let p = 1
    for (let j = 0; j <= poly; j++) {
      row.push(p)
      p *= xi
    }
    return row
  })
  // Solve (A^T A) c = A^T e0 for the row that estimates the value at the center (x=0).
  const ncols = poly + 1
  const AtA: number[][] = Array.from({ length: ncols }, () => new Array(ncols).fill(0))
  for (let i = 0; i < ncols; i++) {
    for (let j = 0; j < ncols; j++) {
      let s = 0
      for (let k = 0; k < window; k++) s += A[k][i] * A[k][j]
      AtA[i][j] = s
    }
  }
  // AtA^{-1} row 0, then coeffs = (AtA^{-1} A^T)[0] = sum_j inv[0][j] * A[k][j] for each k
  const inv = invertMatrix(AtA)
  if (!inv) return new Array(window).fill(1 / window)
  const coeffs: number[] = []
  for (let k = 0; k < window; k++) {
    let s = 0
    for (let j = 0; j < ncols; j++) s += inv[0][j] * A[k][j]
    coeffs.push(s)
  }
  return coeffs
}

function invertMatrix(m: number[][]): number[][] | null {
  const n = m.length
  const aug = m.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))])
  for (let col = 0; col < n; col++) {
    let pivot = col
    for (let r = col + 1; r < n; r++) if (Math.abs(aug[r][col]) > Math.abs(aug[pivot][col])) pivot = r
    if (Math.abs(aug[pivot][col]) < 1e-12) return null
    ;[aug[col], aug[pivot]] = [aug[pivot], aug[col]]
    const pv = aug[col][col]
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pv
    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const f = aug[r][col]
      for (let j = 0; j < 2 * n; j++) aug[r][j] -= f * aug[col][j]
    }
  }
  return aug.map((row) => row.slice(n))
}

export const pluginFilterSavgol: NodeImpl = (inputs, params, ctx) => {
  const state = getState(ctx)
  const serie = toSerie(inputs.list)
  let w = Math.round(Number(params.window) || 11)
  if (w % 2 === 0) w += 1
  w = Math.max(5, w)
  const p = Math.min(Math.round(Number(params.polyorder) ?? 2), w - 2)

  if (serie !== null) {
    const coeffs = sgCoeffs(w, p).slice().reverse()
    const smoothed = convolveCentered(serie, coeffs)
    return { list: smoothed, filtered: smoothed[smoothed.length - 1], raw: serie[serie.length - 1] }
  }

  const v = toScalar(inputs.value)
  const sig = `${w}:${p}`
  if (state.sig !== sig) {
    state.coeffs = sgCoeffs(w, p)
    state.sig = sig
  }
  state.buf.push(v)
  if (state.buf.length > w) state.buf = state.buf.slice(-w)
  if (state.buf.length < w) return { filtered: v, raw: v }
  const coeffs = state.coeffs as number[]
  const filtered = coeffs.reduce((s, c, i) => s + c * state.buf[i], 0)
  return { filtered, raw: v }
}

// 6. Low-pass IIR Filter (1st-order RC)
export const pluginFilterLowpass: NodeImpl = filterNode(
  (v, params, state) => {
    const cutHz = (Number(params.cutoff) || 1000) / 1000
    const fps = Math.max(1, Number(params.fps) || 30)
    const r = 1 - Math.exp((-2 * Math.PI * cutHz) / fps)
    const prev = state.state === undefined ? v : (state.state as number)
    const next = (1 - r) * prev + r * v
    state.state = next
    return { filtered: next, raw: v }
  },
  (x, params) => {
    const cutHz = (Number(params.cutoff) || 1000) / 1000
    const fps = Math.max(1, Number(params.fps) || 30)
    const r = 1 - Math.exp((-2 * Math.PI * cutHz) / fps)
    const pass = (s: number[]) => {
      const out: number[] = []
      let etat = s[0]
      for (const v of s) {
        etat = (1 - r) * etat + r * v
        out.push(etat)
      }
      return out
    }
    return forwardBackward(x, pass)
  }
)

// 7. Holt-Winters (double exponential smoothing — level + trend)
export const pluginFilterHolt: NodeImpl = (inputs, params, ctx) => {
  const state = getState(ctx)
  const serie = toSerie(inputs.list)
  const al = (Number(params.alpha) || 20) / 100
  const be = (Number(params.beta) || 10) / 100

  if (serie !== null) {
    const pass = (s: number[]) => {
      const out: number[] = []
      let L = s[0]
      let T = 0
      for (const v of s) {
        const Lprev = L
        L = al * v + (1 - al) * (L + T)
        T = be * (L - Lprev) + (1 - be) * T
        out.push(L + T)
      }
      return out
    }
    const smoothed = forwardBackward(serie, pass)
    return { list: smoothed, filtered: smoothed[smoothed.length - 1], raw: serie[serie.length - 1] }
  }

  const v = toScalar(inputs.value)
  if (state.L === undefined) {
    state.L = v
    state.T = 0
  } else {
    const Lprev = state.L as number
    state.L = al * v + (1 - al) * ((state.L as number) + (state.T as number))
    state.T = be * ((state.L as number) - Lprev) + (1 - be) * (state.T as number)
  }
  return { filtered: (state.L as number) + (state.T as number), trend: state.T, raw: v }
}

// 8. Gaussian Smoothing (buffer convolution)
function gaussKernel(w: number, sigma: number): number[] {
  const half = Math.floor(w / 2)
  const k: number[] = []
  for (let i = -half; i <= half; i++) k.push(Math.exp(-0.5 * (i / sigma) ** 2))
  const sum = k.reduce((a, b) => a + b, 0)
  return k.map((v) => v / sum)
}

export const pluginFilterGaussian: NodeImpl = (inputs, params, ctx) => {
  const state = getState(ctx)
  const serie = toSerie(inputs.list)
  let w = Math.round(Number(params.window) || 15)
  if (w % 2 === 0) w += 1
  const sigma = Number(params.sigma) || 5

  if (serie !== null) {
    const smoothed = convolveCentered(serie, gaussKernel(w, sigma))
    return { list: smoothed, filtered: smoothed[smoothed.length - 1], raw: serie[serie.length - 1] }
  }

  const v = toScalar(inputs.value)
  const sig = `${w}:${sigma}`
  if (state.sig !== sig) {
    state.kernel = gaussKernel(w, sigma)
    state.sig = sig
  }
  state.buf.push(v)
  if (state.buf.length > w) state.buf = state.buf.slice(-w)
  if (state.buf.length < w) return { filtered: v, raw: v }
  const kernel = state.kernel as number[]
  const filtered = kernel.reduce((s, c, i) => s + c * state.buf[i], 0)
  return { filtered, raw: v }
}

// 9. LOESS / LOWESS (local weighted linear regression, degree 1)
function tricubic(u: number): number {
  const c = Math.min(1, Math.abs(u))
  return (1 - c ** 3) ** 3
}

/** Weighted linear regression of y on x with weights w — returns [intercept, slope]. */
function weightedLinReg(x: number[], y: number[], w: number[]): [number, number] {
  let sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0
  for (let i = 0; i < x.length; i++) {
    sw += w[i]
    swx += w[i] * x[i]
    swy += w[i] * y[i]
    swxx += w[i] * x[i] * x[i]
    swxy += w[i] * x[i] * y[i]
  }
  const det = sw * swxx - swx * swx
  if (Math.abs(det) < 1e-12) return [swy / (sw || 1), 0]
  const intercept = (swxx * swy - swx * swxy) / det
  const slope = (sw * swxy - swx * swy) / det
  return [intercept, slope]
}

export const pluginFilterLoess: NodeImpl = (inputs, params, ctx) => {
  const state = getState(ctx)
  const serie = toSerie(inputs.list)
  const span = Math.max(5, Math.round(Number(params.span) || 30))

  if (serie !== null) {
    const n = serie.length
    const k = Math.max(3, Math.min(span, n))
    const t = serie.map((_, i) => i)
    const out: number[] = []
    for (let i = 0; i < n; i++) {
      const dep = Math.max(0, i - Math.floor(k / 2))
      const fin = Math.min(n, i + Math.floor(k / 2) + 1)
      const xi = t.slice(dep, fin)
      const yi = serie.slice(dep, fin)
      const d = xi.map((xv) => Math.abs(xv - t[i]))
      const maxD = Math.max(...d) + 1e-10
      const w = d.map((dv) => tricubic(dv / maxD))
      const [intercept, slope] = weightedLinReg(xi, yi, w)
      out.push(intercept + slope * t[i])
    }
    return { list: out, filtered: out[out.length - 1], raw: serie[serie.length - 1] }
  }

  const v = toScalar(inputs.value)
  state.buf.push(v)
  if (state.buf.length > span * 2) state.buf = state.buf.slice(-span * 2)
  const n = state.buf.length
  if (n < 3) return { filtered: v, raw: v }
  const x = state.buf.map((_, i) => i)
  const y = state.buf
  const x0 = x[n - 1]
  const k = Math.max(3, Math.min(span, n))
  const dists = x.map((xv) => Math.abs(xv - x0))
  const maxD = dists.slice().sort((a, b) => a - b)[k - 1] + 1e-10
  const w = dists.map((dv) => tricubic(dv / maxD))
  const [intercept, slope] = weightedLinReg(x, y, w)
  return { filtered: intercept + slope * x0, raw: v }
}

// 10. Particle Filter (1D, random-walk state model)
function gaussianRandom(): number {
  let u = 0, v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function systematicResample(weights: number[]): number[] {
  const n = weights.length
  const cumsum: number[] = []
  let acc = 0
  for (const w of weights) {
    acc += w
    cumsum.push(acc)
  }
  const u0 = Math.random()
  const indices: number[] = []
  for (let i = 0; i < n; i++) {
    const target = (i + u0) / n
    let idx = cumsum.findIndex((c) => c >= target)
    if (idx === -1) idx = n - 1
    indices.push(idx)
  }
  return indices
}

export const pluginFilterParticle: NodeImpl = (inputs, params, ctx) => {
  const state = getState(ctx)
  const serie = toSerie(inputs.list)
  const N = Math.max(10, Math.round(Number(params.particles) || 100))
  const pStd = (Number(params.process_std) || 10) / 100
  const mStd = (Number(params.meas_std) || 50) / 100

  const step = (particles: number[], z: number): [number[], number] => {
    const predicted = particles.map((p) => p + gaussianRandom() * pStd)
    const logW = predicted.map((p) => -0.5 * ((p - z) / (mStd + 1e-10)) ** 2)
    const maxLogW = Math.max(...logW)
    const w = logW.map((lw) => Math.exp(lw - maxLogW))
    const sumW = w.reduce((a, b) => a + b, 0)
    const weights = w.map((wv) => wv / sumW)
    const estimate = predicted.reduce((s, p, i) => s + p * weights[i], 0)
    const indices = systematicResample(weights)
    return [indices.map((i) => predicted[i]), estimate]
  }

  if (serie !== null) {
    const pass = (s: number[]) => {
      let particles = new Array(N).fill(s[0])
      const out: number[] = []
      for (const z of s) {
        const [next, estimate] = step(particles, z)
        particles = next
        out.push(estimate)
      }
      return out
    }
    const smoothed = forwardBackward(serie, pass)
    return { list: smoothed, filtered: smoothed[smoothed.length - 1], raw: serie[serie.length - 1] }
  }

  const z = toScalar(inputs.value)
  if (!Array.isArray(state.particles) || (state.particles as number[]).length !== N) {
    state.particles = new Array(N).fill(z)
  }
  const [next, estimate] = step(state.particles as number[], z)
  state.particles = next
  return { filtered: estimate, raw: z }
}
