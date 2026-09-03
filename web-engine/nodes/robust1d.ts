import type { NodeImpl } from '../types'
import { drawPolyline } from '../cvUtils'

/** Robust 1-D statistics nodes (ch16 pedagogy) — pure JS math + a small canvas plot. */

function asArray(vals: unknown): number[] {
  if (vals === null || vals === undefined) return []
  if (typeof vals === 'number') return [vals]
  if (!Array.isArray(vals)) return []
  const out: number[] = []
  for (const v of vals) {
    const n = Number(v)
    if (Number.isFinite(n)) out.push(n)
  }
  return out
}

function median(x: number[]): number {
  const sorted = x.slice().sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function mad(x: number[], med: number): number {
  return median(x.map((v) => Math.abs(v - med)))
}

function mean(x: number[]): number {
  return x.reduce((a, b) => a + b, 0) / x.length
}

function canvas(cv: any, w = 520, h = 200): any {
  return new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(255, 255, 255))
}

const PALETTE: [number, number, number][] = [
  [220, 120, 0], [0, 90, 200], [0, 160, 0], [160, 0, 160],
]

function numberline(cv: any, img: any, x: number[], markers: [string, number][]): void {
  const h = img.rows
  const w = img.cols
  if (x.length === 0) return
  const lo = Math.min(...x)
  const hi = Math.max(...x)
  const span = hi - lo || 1
  const pad = 50
  const y = h - 60
  const gx = (v: number) => Math.round(pad + ((v - lo) / span) * (w - 2 * pad))

  cv.line(img, new cv.Point(pad, y), new cv.Point(w - pad, y), new cv.Scalar(180, 180, 180, 255), 1)
  for (const v of x) cv.circle(img, new cv.Point(gx(v), y), 4, new cv.Scalar(150, 150, 150, 255), -1, cv.LINE_AA)
  markers.forEach(([name, val], i) => {
    const [b, g, r] = PALETTE[i % PALETTE.length]
    const col = new cv.Scalar(b, g, r, 255)
    const px = gx(val)
    cv.line(img, new cv.Point(px, y - 40), new cv.Point(px, y + 12), col, 2, cv.LINE_AA)
    cv.putText(img, `${name}=${Number(val.toFixed(3))}`, new cv.Point(10, 24 + i * 22), cv.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv.LINE_AA)
  })
}

// ---------------------------------------------------------------------------
export const sciScalarList: NodeImpl = (_inputs, params) => {
  const raw = String(params.values ?? '')
  const out: number[] = []
  for (const tok of raw.replace(/;/g, ',').replace(/\n/g, ',').replace(/ /g, ',').split(',')) {
    const t = tok.trim()
    if (!t) continue
    const n = Number(t)
    if (Number.isFinite(n)) out.push(n)
  }
  return { list: out, count: out.length }
}

// ---------------------------------------------------------------------------
export const sciRobustLocation: NodeImpl = (inputs, _params, ctx) => {
  const cv = ctx.cv
  const x = asArray(inputs.values)
  if (x.length === 0) return { main: ctx.track(canvas(cv)), mean: 0, median: 0, mean_influence: 0, data: null }

  const meanV = mean(x)
  const medianV = median(x)
  let meanWo = meanV, medianWo = medianV
  if (x.length > 1) {
    const woLast = x.slice(0, -1)
    meanWo = mean(woLast)
    medianWo = median(woLast)
  }

  const img = ctx.track(canvas(cv))
  numberline(cv, img, x, [['mean', meanV], ['median', medianV]])

  return {
    main: img,
    mean: Number(meanV.toFixed(4)),
    median: Number(medianV.toFixed(4)),
    mean_influence: Number((meanV - meanWo).toFixed(4)),
    data: {
      mean: Number(meanV.toFixed(4)),
      median: Number(medianV.toFixed(4)),
      mean_shift_last: Number((meanV - meanWo).toFixed(4)),
      median_shift_last: Number((medianV - medianWo).toFixed(4)),
      n: x.length,
    },
  }
}

// ---------------------------------------------------------------------------
export const sciMadScale: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const x = asArray(inputs.values)
  if (x.length === 0) return { main: ctx.track(canvas(cv)), median: 0, mad: 0, sigma_hat: 0, outliers: 0, zscores: [], data: null }

  const med = median(x)
  const madV = mad(x, med)
  const sigmaHat = 1.4826 * madV
  const zt = Number(params.z_thresh) || 3.5
  const z = madV > 1e-9 ? x.map((v) => (0.6745 * (v - med)) / madV) : x.map(() => 0)
  const flags = z.map((v) => Math.abs(v) > zt)
  const stdV = Math.sqrt(x.reduce((a, v) => a + (v - mean(x)) ** 2, 0) / x.length)

  const img = ctx.track(canvas(cv))
  numberline(cv, img, x, [['median', med], ['sigma_hat', med + sigmaHat]])
  cv.putText(img, `classic std=${stdV.toFixed(2)}  (MAD sigma=${sigmaHat.toFixed(2)})`, new cv.Point(10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, new cv.Scalar(90, 90, 90, 255), 1, cv.LINE_AA)

  const outlierCount = flags.filter(Boolean).length
  return {
    main: img,
    median: Number(med.toFixed(4)),
    mad: Number(madV.toFixed(4)),
    sigma_hat: Number(sigmaHat.toFixed(4)),
    outliers: outlierCount,
    zscores: z.map((v) => Number(v.toFixed(3))),
    data: {
      median: Number(med.toFixed(4)),
      mad: Number(madV.toFixed(4)),
      sigma_hat: Number(sigmaHat.toFixed(4)),
      classic_std: Number(stdV.toFixed(4)),
      outlier_count: outlierCount,
      outlier_values: x.filter((_, i) => flags[i]).map((v) => Number(v.toFixed(3))),
    },
  }
}

// ---------------------------------------------------------------------------
const M_KERNELS = ['Huber', 'Tukey']

function mWeights(e: number[], k: number, kernel: string): number[] {
  return e.map((v) => {
    const ae = Math.abs(v)
    if (kernel === 'Tukey') {
      if (ae > k) return 0
      return (1 - (v / k) ** 2) ** 2
    }
    return ae > k ? k / Math.max(ae, 1e-9) : 1
  })
}

export const sciMEstimator: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const x = asArray(inputs.values)
  const kernelParam = params.kernel
  const kernel = typeof kernelParam === 'number' ? (M_KERNELS[kernelParam] ?? 'Huber') : M_KERNELS.includes(String(kernelParam)) ? String(kernelParam) : 'Huber'
  const kSigma = Number(params.k_sigma) || 1.345
  const iters = Math.round(Number(params.iterations) || 10)

  if (x.length === 0) return { main: ctx.track(canvas(cv, 520, 260)), estimate: 0, ls_estimate: 0, weights: [], data: null }

  const ls = mean(x)
  let theta = median(x)
  let w = x.map(() => 1)
  let sigma = 0
  for (let iter = 0; iter < iters; iter++) {
    const e = x.map((v) => v - theta)
    const med = median(x)
    sigma = 1.4826 * mad(x, med)
    if (sigma < 1e-9) break
    const k = kSigma * sigma
    void k
    w = mWeights(
      e.map((v) => v / sigma),
      kSigma,
      kernel
    )
    const sw = w.reduce((a, b) => a + b, 0)
    if (sw < 1e-9) break
    theta = w.reduce((a, wv, i) => a + wv * x[i], 0) / sw
  }

  const med = median(x)
  sigma = 1.4826 * mad(x, med)
  const kPlot = Math.max(kSigma, 0.5)

  const img = ctx.track(canvas(cv, 520, 260))
  const h = img.rows
  const wImg = img.cols
  const pad = 50
  const xs: number[] = []
  const n = 300
  for (let i = 0; i < n; i++) xs.push(-3 * kPlot + (i / (n - 1)) * 6 * kPlot)
  const psi = xs.map((v) => (kernel === 'Tukey' ? (Math.abs(v) <= kPlot ? v * (1 - (v / kPlot) ** 2) ** 2 : 0) : Math.max(-kPlot, Math.min(kPlot, v))))
  const pmax = Math.max(...psi.map(Math.abs), 1e-6)
  const gx = (v: number) => Math.round(pad + ((v + 3 * kPlot) / (6 * kPlot)) * (wImg - 2 * pad))
  const gy = (v: number) => Math.round(h - 40 - ((v + pmax) / (2 * pmax)) * (h - 70))
  cv.line(img, new cv.Point(pad, gy(0)), new cv.Point(wImg - pad, gy(0)), new cv.Scalar(210, 210, 210, 255), 1)
  const pts = xs.map((xv, i) => ({ x: gx(xv), y: gy(psi[i]) }))
  drawPolyline(cv, img, pts, false, new cv.Scalar(200, 90, 0, 255), 2)
  cv.putText(img, `psi (${kernel}, k=${kPlot.toFixed(3)})`, new cv.Point(pad, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, new cv.Scalar(200, 90, 0, 255), 1, cv.LINE_AA)
  cv.putText(img, `robust=${theta.toFixed(3)}   LS=${ls.toFixed(3)}`, new cv.Point(50, 250), cv.FONT_HERSHEY_SIMPLEX, 0.55, new cv.Scalar(0, 0, 0, 255), 1, cv.LINE_AA)

  return {
    main: img,
    estimate: Number(theta.toFixed(4)),
    ls_estimate: Number(ls.toFixed(4)),
    weights: w.map((v) => Number(v.toFixed(3))),
    data: {
      kernel,
      robust_estimate: Number(theta.toFixed(4)),
      ls_estimate: Number(ls.toFixed(4)),
      sigma_hat: Number(sigma.toFixed(4)),
      weights: w.map((v) => Number(v.toFixed(3))),
    },
  }
}
