import type { NodeImpl } from '../types'
import { toBgr, toGray } from '../cvUtils'
import { applyColormap, magmaColor, plasmaColor, viridisColor, infernoColor } from '../colormaps'
import { gaborKernel } from './imageFx'

/** Min/max of a Float32Array, the input every normalisation here needs. */
function extent(data: Float32Array | Float64Array): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < data.length; i++) {
    if (data[i] < lo) lo = data[i]
    if (data[i] > hi) hi = data[i]
  }
  return [lo, hi]
}

/** CV_32F → CV_8U scaled to the data's own range, i.e. cv2.normalize NORM_MINMAX. */
function normalize8(cv: any, src: any): any {
  const data = src.data32F
  const [lo, hi] = extent(data)
  const out = new cv.Mat(src.rows, src.cols, cv.CV_8U)
  const dst = out.data
  const span = hi - lo
  if (span < 1e-8) {
    dst.fill(0)
    return out
  }
  for (let i = 0; i < data.length; i++) dst[i] = Math.round(((data[i] - lo) / span) * 255)
  return out
}

/** Odd kernel size covering ±3σ, matching the desktop's `int(6σ+1) | 1`. */
function gaussianKernelSize(sigma: number): number {
  return Math.max(3, (Math.trunc(6 * sigma + 1) | 1))
}

/* ------------------------------------------------------------ mask statistics */

export const featMaskStats: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const maskIn = inputs.mask as any
  const empty = { stats: {}, area_px: 0, area_pct: 0, centroid_x: 0, centroid_y: 0 }
  if (!maskIn) return empty

  const gray = toGray(cv, maskIn)
  const binary = new cv.Mat()
  cv.threshold(gray, binary, 127, 255, cv.THRESH_BINARY)
  gray.delete()

  const w = binary.cols
  const h = binary.rows
  const bits = binary.data

  let area = 0
  let sumX = 0
  let sumY = 0
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (let i = 0; i < bits.length; i++) {
    if (!bits[i]) continue
    const x = i % w
    const y = (i / w) | 0
    area++
    sumX += x
    sumY += y
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }

  let refArea = Math.max(1, area)
  const refIn = inputs.ref_mask as any
  if (refIn) {
    const refGray = toGray(cv, refIn)
    const refBin = new cv.Mat()
    cv.threshold(refGray, refBin, 127, 255, cv.THRESH_BINARY)
    refArea = Math.max(1, cv.countNonZero(refBin))
    refGray.delete()
    refBin.delete()
  }

  let cx = 0
  let cy = 0
  if (area > 0) {
    const image = inputs.image as any
    const weighted = params.weighted_centroid === true || String(params.weighted_centroid) === 'true'
    let done = false
    if (weighted && image) {
      // Intensity-weighted centroid — the "centre of pressure" of the region.
      const intensity = toGray(cv, image)
      if (intensity.cols === w && intensity.rows === h) {
        const pixels = intensity.data
        let total = 0
        let wx = 0
        let wy = 0
        for (let i = 0; i < bits.length; i++) {
          if (!bits[i]) continue
          const value = pixels[i]
          total += value
          wx += (i % w) * value
          wy += ((i / w) | 0) * value
        }
        if (total > 0) {
          cx = wx / total
          cy = wy / total
          done = true
        }
      }
      intensity.delete()
    }
    if (!done) {
      cx = sumX / area
      cy = sumY / area
    }
  }

  binary.delete()

  const areaPct = Math.round((100 * area) / refArea * 100) / 100
  const bboxW = area > 0 ? maxX - minX : 0
  const bboxH = area > 0 ? maxY - minY : 0

  return {
    stats: {
      area_px: area,
      area_pct: areaPct,
      centroid_x: Math.round(cx * 10) / 10,
      centroid_y: Math.round(cy * 10) / 10,
      bbox_w: bboxW,
      bbox_h: bboxH,
    },
    area_px: area,
    area_pct: areaPct,
    centroid_x: cx,
    centroid_y: cy,
  }
}

/* ----------------------------------------------------------------- skeleton */

export const featSkeleton: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const maskIn = (inputs.mask ?? inputs.main) as any
  if (!maskIn) return { main: null, branch_count: 0, max_radius: 0 }

  const gray = toGray(cv, maskIn)
  const binary = new cv.Mat()
  cv.threshold(gray, binary, 127, 1, cv.THRESH_BINARY)

  // The distance transform's ridge is the medial axis: every ridge pixel is the
  // centre of a maximal inscribed disc, so DT and skeleton encode one geometry.
  const dt = new cv.Mat()
  cv.distanceTransform(binary, dt, cv.DIST_L2, cv.DIST_MASK_PRECISE)
  const dtData = dt.data32F
  const maxRadius = extent(dtData)[1]

  const minRadius = Math.max(1, Math.round(Number(params.min_radius) || 2))
  const win = 2 * minRadius + 1
  const localMax = new cv.Mat()
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(win, win))
  cv.dilate(dt, localMax, kernel)
  kernel.delete()

  const peaks = localMax.data32F
  const skeleton = new cv.Mat(dt.rows, dt.cols, cv.CV_8U)
  const skelData = skeleton.data
  for (let i = 0; i < dtData.length; i++) {
    skelData[i] = dtData[i] > 0 && Math.abs(dtData[i] - peaks[i]) < 0.5 ? 255 : 0
  }
  localMax.delete()

  const labels = new cv.Mat()
  const branchCount = Math.max(0, cv.connectedComponents(skeleton, labels, 8) - 1)
  labels.delete()

  const w = skeleton.cols
  const h = skeleton.rows
  const vis = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))
  const visData = vis.data
  const binData = binary.data
  const overlay = params.overlay !== false
  for (let i = 0; i < skelData.length; i++) {
    if (skelData[i]) {
      visData[i * 3] = 100
      visData[i * 3 + 1] = 255
      visData[i * 3 + 2] = 0
    } else if (overlay && binData[i]) {
      visData[i * 3] = 255
      visData[i * 3 + 1] = 255
      visData[i * 3 + 2] = 255
    }
  }

  cv.putText(
    vis,
    `branches=${branchCount}  max_r=${maxRadius.toFixed(1)}px`,
    new cv.Point(8, 20),
    cv.FONT_HERSHEY_SIMPLEX,
    0.42,
    new cv.Scalar(255, 255, 255, 255),
    1,
    cv.LINE_AA
  )

  skeleton.delete()
  binary.delete()
  dt.delete()
  gray.delete()

  return { main: vis, branch_count: branchCount, max_radius: Math.round(maxRadius * 100) / 100 }
}

/* --------------------------------------------------------- structure tensor */

export const featStructureTensor: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  if (!image) return { main: null, harris: null, lambda1: null, lambda2: null }

  const gray = toGray(cv, image)
  const flt = new cv.Mat()
  gray.convertTo(flt, cv.CV_32F, 1 / 255)
  gray.delete()

  const sigmaD = Number(params.sigma_deriv) || 1.0
  const sigmaI = Number(params.sigma_int) || 3.0
  const k = Number(params.k_harris) ?? 0.04
  const flatThreshold = Number(params.flat_thresh) ?? 0.001
  const edgeRatio = Number(params.edge_ratio) || 4.0

  const kd = gaussianKernelSize(sigmaD)
  const ki = gaussianKernelSize(sigmaI)

  const blurred = new cv.Mat()
  cv.GaussianBlur(flt, blurred, new cv.Size(kd, kd), sigmaD, sigmaD, cv.BORDER_DEFAULT)

  const ix = new cv.Mat()
  const iy = new cv.Mat()
  cv.Sobel(blurred, ix, cv.CV_32F, 1, 0, 3)
  cv.Sobel(blurred, iy, cv.CV_32F, 0, 1, 3)
  blurred.delete()

  const n = ix.rows * ix.cols
  const ixd = ix.data32F
  const iyd = iy.data32F

  const products = (fn: (i: number) => number) => {
    const m = new cv.Mat(ix.rows, ix.cols, cv.CV_32F)
    const d = m.data32F
    for (let i = 0; i < n; i++) d[i] = fn(i)
    const blurredProduct = new cv.Mat()
    cv.GaussianBlur(m, blurredProduct, new cv.Size(ki, ki), sigmaI, sigmaI, cv.BORDER_DEFAULT)
    m.delete()
    return blurredProduct
  }

  // The tensor is the Gaussian-weighted sum of the gradient outer products.
  const ixx = products((i) => ixd[i] * ixd[i])
  const iyy = products((i) => iyd[i] * iyd[i])
  const ixy = products((i) => ixd[i] * iyd[i])
  ix.delete()
  iy.delete()

  const xx = ixx.data32F
  const yy = iyy.data32F
  const xy = ixy.data32F

  const lam1 = new cv.Mat(ixx.rows, ixx.cols, cv.CV_32F)
  const lam2 = new cv.Mat(ixx.rows, ixx.cols, cv.CV_32F)
  const harris = new cv.Mat(ixx.rows, ixx.cols, cv.CV_32F)
  const l1 = lam1.data32F
  const l2 = lam2.data32F
  const hr = harris.data32F

  for (let i = 0; i < n; i++) {
    const trace = xx[i] + yy[i]
    const det = xx[i] * yy[i] - xy[i] * xy[i]
    // Closed form for the eigenvalues of a symmetric 2×2 matrix.
    const half = (xx[i] - yy[i]) / 2
    const disc = Math.sqrt(Math.max(half * half + xy[i] * xy[i], 0))
    l1[i] = trace / 2 + disc
    l2[i] = Math.max(trace / 2 - disc, 0)
    hr[i] = det - k * trace * trace
  }
  ixx.delete()
  iyy.delete()
  ixy.delete()

  const harrisPositive = new cv.Mat(harris.rows, harris.cols, cv.CV_32F)
  const hp = harrisPositive.data32F
  const harrisMax = extent(hr)[1]
  for (let i = 0; i < n; i++) hp[i] = Math.min(Math.max(hr[i], 0), harrisMax + 1e-8)

  const lam1Max = extent(l1)[1] + 1e-8
  const w = lam1.cols
  const h = lam1.rows
  const classify = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))
  const cls = classify.data
  for (let i = 0; i < n; i++) {
    const n1 = l1[i] / lam1Max
    const n2 = l2[i] / lam1Max
    let colour: [number, number, number]
    if (n1 < flatThreshold) colour = [180, 60, 0] // blue = flat
    else if (n2 >= flatThreshold && n1 / (n2 + 1e-8) < edgeRatio) colour = [0, 60, 220] // red = corner
    else colour = [0, 180, 60] // green = edge
    cls[i * 3] = colour[0]
    cls[i * 3 + 1] = colour[1]
    cls[i * 3 + 2] = colour[2]
  }

  const legend = (text: string, x: number, colour: [number, number, number]) =>
    cv.putText(classify, text, new cv.Point(x, 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, new cv.Scalar(colour[0], colour[1], colour[2], 255), 1, cv.LINE_AA)
  legend('flat', 8, [180, 60, 0])
  legend('edge', 50, [0, 180, 60])
  legend('corner', 95, [0, 60, 220])

  const toMap = (src: any, colour: (v: number) => [number, number, number]) => {
    const eight = normalize8(cv, src)
    const mapped = ctx.track(applyColormap(cv, eight, colour))
    eight.delete()
    return mapped
  }

  const outHarris = toMap(harrisPositive, viridisColor)
  const outLam1 = toMap(lam1, plasmaColor)
  const outLam2 = toMap(lam2, plasmaColor)

  harrisPositive.delete()
  harris.delete()
  lam1.delete()
  lam2.delete()
  flt.delete()

  return { main: classify, harris: outHarris, lambda1: outLam1, lambda2: outLam2 }
}

/* ------------------------------------------------------------- RANSAC line */

const MAX_LINE_POINTS = 4000

export const featRansacLine: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const edges = (inputs.image ?? inputs.main) as any
  if (!edges) return { main: null, inliers: 0, angle: 0, n_points: 0 }

  const gray = toGray(cv, edges)
  const w = gray.cols
  const h = gray.rows
  const data = gray.data

  let pts: [number, number][] = []
  for (let i = 0; i < data.length; i++) {
    if (data[i] > 127) pts.push([i % w, (i / w) | 0])
  }
  gray.delete()

  if (pts.length > MAX_LINE_POINTS) {
    // Even subsample rather than a random draw, so the fit is reproducible.
    const stride = Math.ceil(pts.length / MAX_LINE_POINTS)
    pts = pts.filter((_, i) => i % stride === 0)
  }

  const overlay = ctx.track(toBgr(cv, edges))
  if (pts.length < 2) return { main: overlay, inliers: 0, angle: 0, n_points: pts.length }

  const fitL2 = (subset: [number, number][]): [number, number, number, number] => {
    const flat: number[] = []
    for (const p of subset) flat.push(p[0], p[1])
    const mat = cv.matFromArray(subset.length, 1, cv.CV_32FC2, flat)
    const line = new cv.Mat()
    cv.fitLine(mat, line, cv.DIST_L2, 0, 0.01, 0.01)
    const v = line.data32F
    const out: [number, number, number, number] = [v[0], v[1], v[2], v[3]]
    mat.delete()
    line.delete()
    return out
  }

  /** Perpendicular distance from p to the line through (x0,y0) with unit dir (vx,vy). */
  const distance = (p: [number, number], vx: number, vy: number, x0: number, y0: number) =>
    Math.abs((p[0] - x0) * vy - (p[1] - y0) * vx)

  const mode = Math.round(Number(params.mode) || 0)
  const threshold = Number(params.threshold) || 3.0
  const iterations = Math.max(10, Math.round(Number(params.iterations) || 200))

  let vx: number
  let vy: number
  let x0: number
  let y0: number
  let inliers: number

  if (mode === 0) {
    let bestCount = -1
    let bestMask: boolean[] | null = null
    for (let iter = 0; iter < iterations; iter++) {
      const i = Math.floor(Math.random() * pts.length)
      const j = Math.floor(Math.random() * pts.length)
      if (i === j) continue
      const dx = pts[j][0] - pts[i][0]
      const dy = pts[j][1] - pts[i][1]
      const norm = Math.hypot(dx, dy)
      if (norm < 1e-6) continue
      const ux = dx / norm
      const uy = dy / norm
      const mask: boolean[] = new Array(pts.length)
      let count = 0
      for (let p = 0; p < pts.length; p++) {
        const inside = distance(pts[p], ux, uy, pts[i][0], pts[i][1]) < threshold
        mask[p] = inside
        if (inside) count++
      }
      if (count > bestCount) {
        bestCount = count
        bestMask = mask
      }
    }
    if (!bestMask || bestCount < 2) {
      ;[vx, vy, x0, y0] = fitL2(pts)
      inliers = pts.length
    } else {
      // Refit on the consensus set: the two-point hypothesis only selects it.
      ;[vx, vy, x0, y0] = fitL2(pts.filter((_, i) => bestMask![i]))
      inliers = bestCount
    }
  } else {
    ;[vx, vy, x0, y0] = fitL2(pts)
    inliers = pts.reduce((n, p) => n + (distance(p, vx, vy, x0, y0) < threshold ? 1 : 0), 0)
  }

  const t = Math.max(h, w)
  const green = new cv.Scalar(0, 220, 0, 255)
  cv.line(
    overlay,
    new cv.Point(Math.round(x0 - vx * t), Math.round(y0 - vy * t)),
    new cv.Point(Math.round(x0 + vx * t), Math.round(y0 + vy * t)),
    green,
    2,
    cv.LINE_AA
  )

  const angle = (Math.atan2(vy, vx) * 180) / Math.PI
  const label = `${mode === 0 ? 'RANSAC' : 'L2'}  inliers=${inliers}/${pts.length}  angle=${angle.toFixed(1)}`
  const at = new cv.Point(8, 22)
  cv.putText(overlay, label, at, cv.FONT_HERSHEY_SIMPLEX, 0.55, new cv.Scalar(0, 0, 0, 255), 3, cv.LINE_AA)
  cv.putText(overlay, label, at, cv.FONT_HERSHEY_SIMPLEX, 0.55, green, 1, cv.LINE_AA)

  return { main: overlay, inliers, angle: Math.round(angle * 100) / 100, n_points: pts.length }
}

/* -------------------------------------------------------------------- HOG */

export const featHog: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  if (!image) return { main: null, signature: [], n_bins: 0 }

  const gray = toGray(cv, image)
  const flt = new cv.Mat()
  gray.convertTo(flt, cv.CV_32F)

  const nBins = Math.max(4, Math.round(Number(params.orientations) || 9))
  const cellPx = Math.max(4, Math.round(Number(params.cell_px) || 16))

  const gx = new cv.Mat()
  const gy = new cv.Mat()
  cv.Sobel(flt, gx, cv.CV_32F, 1, 0, 3)
  cv.Sobel(flt, gy, cv.CV_32F, 0, 1, 3)
  flt.delete()

  const w = gray.cols
  const h = gray.rows
  const gxd = gx.data32F
  const gyd = gy.data32F

  // Per-pixel magnitude and orientation, folded to [0,180) so that gradients
  // pointing opposite ways land in the same bin — an edge has no direction.
  const magnitude = new Float32Array(w * h)
  const angle = new Float32Array(w * h)
  for (let i = 0; i < magnitude.length; i++) {
    magnitude[i] = Math.hypot(gxd[i], gyd[i])
    let a = (Math.atan2(gyd[i], gxd[i]) * 180) / Math.PI
    a %= 180
    if (a < 0) a += 180
    angle[i] = a
  }
  gx.delete()
  gy.delete()

  // Global orientation histogram — the compact shape signature.
  const global = new Float64Array(nBins)
  for (let i = 0; i < magnitude.length; i++) {
    const bin = Math.min(nBins - 1, Math.floor((angle[i] / 180) * nBins))
    global[bin] += magnitude[i]
  }
  const total = global.reduce((a, b) => a + b, 0) || 1
  const signature = Array.from(global, (v) => Math.round((v / total) * 10000) / 10000)

  // Cell rose plot: in each cell, a spoke per orientation whose length is that
  // bin's share of the cell's gradient energy. This is what skimage draws.
  const cellsX = Math.max(1, Math.floor(w / cellPx))
  const cellsY = Math.max(1, Math.floor(h / cellPx))
  const rose = new cv.Mat(h, w, cv.CV_32F, new cv.Scalar(0))
  const roseData = rose.data32F

  const plot = (x0: number, y0: number, x1: number, y1: number, value: number) => {
    // Short spokes only, so a simple DDA is both adequate and cheap.
    const steps = Math.max(1, Math.round(Math.hypot(x1 - x0, y1 - y0)))
    for (let s = 0; s <= steps; s++) {
      const x = Math.round(x0 + ((x1 - x0) * s) / steps)
      const y = Math.round(y0 + ((y1 - y0) * s) / steps)
      if (x < 0 || y < 0 || x >= w || y >= h) continue
      const at = y * w + x
      if (value > roseData[at]) roseData[at] = value
    }
  }

  const half = cellPx / 2
  const cellHist = new Float64Array(nBins)
  for (let cy = 0; cy < cellsY; cy++) {
    for (let cx = 0; cx < cellsX; cx++) {
      cellHist.fill(0)
      for (let y = cy * cellPx; y < (cy + 1) * cellPx; y++) {
        for (let x = cx * cellPx; x < (cx + 1) * cellPx; x++) {
          const i = y * w + x
          const bin = Math.min(nBins - 1, Math.floor((angle[i] / 180) * nBins))
          cellHist[bin] += magnitude[i]
        }
      }
      let peak = 0
      for (let b = 0; b < nBins; b++) if (cellHist[b] > peak) peak = cellHist[b]
      if (peak <= 0) continue
      const centreX = cx * cellPx + half
      const centreY = cy * cellPx + half
      for (let b = 0; b < nBins; b++) {
        const strength = cellHist[b] / peak
        if (strength < 0.05) continue
        // The spoke is drawn along the edge, i.e. perpendicular to the gradient.
        const theta = ((b + 0.5) / nBins) * Math.PI + Math.PI / 2
        const dx = Math.cos(theta) * half * strength
        const dy = Math.sin(theta) * half * strength
        plot(centreX - dx, centreY - dy, centreX + dx, centreY + dy, strength)
      }
    }
  }

  const roseU8 = normalize8(cv, rose)
  rose.delete()
  const hogVis = applyColormap(cv, roseU8, infernoColor)
  roseU8.delete()

  let out: any
  if (params.overlay !== false) {
    const base = toBgr(cv, image)
    out = ctx.track(new cv.Mat())
    cv.addWeighted(base, 0.4, hogVis, 0.9, 0, out)
    base.delete()
    hogVis.delete()
  } else {
    out = ctx.track(hogVis)
  }

  gray.delete()
  return { main: out, signature, n_bins: nBins }
}

/* -------------------------------------------------------------- Gabor bank */

/** One labelled response panel of the montage. */
function gaborPanel(cv: any, response: any, label: string, panelPx: number, colorize: boolean): any {
  const scale = panelPx / Math.max(response.rows, response.cols)
  const pw = Math.max(1, Math.round(response.cols * scale))
  const ph = Math.max(1, Math.round(response.rows * scale))
  const small = new cv.Mat()
  cv.resize(response, small, new cv.Size(pw, ph), 0, 0, cv.INTER_AREA)

  let panel: any
  if (colorize) {
    panel = applyColormap(cv, small, magmaColor)
  } else {
    panel = new cv.Mat()
    cv.cvtColor(small, panel, cv.COLOR_GRAY2BGR)
  }
  small.delete()

  cv.rectangle(panel, new cv.Point(0, 0), new cv.Point(pw - 1, ph - 1), new cv.Scalar(60, 60, 60, 255), 1)
  const at = new cv.Point(6, 20)
  cv.putText(panel, label, at, cv.FONT_HERSHEY_SIMPLEX, 0.6, new cv.Scalar(0, 0, 0, 255), 3, cv.LINE_AA)
  cv.putText(panel, label, at, cv.FONT_HERSHEY_SIMPLEX, 0.6, new cv.Scalar(255, 255, 255, 255), 1, cv.LINE_AA)
  return panel
}

export const featGaborBank: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  if (!image) return { main: null, responses_grid: null, energy_map: null, signature: [], n_orientations: 0 }

  const gray = toGray(cv, image)
  const flt = new cv.Mat()
  gray.convertTo(flt, cv.CV_32F)
  gray.delete()

  const nTheta = Math.max(2, Math.round(Number(params.n_theta) || 4))
  const wavelength = Number(params.wavelength) || 8.0
  const sigma = Number(params.sigma) || 4.0
  const gamma = Number(params.gamma) || 0.5
  let ksize = Math.max(7, Math.round(Number(params.ksize) || 31))
  if (ksize % 2 === 0) ksize += 1
  const panelPx = Math.max(96, Math.round(Number(params.panel_px) || 256))
  const colorize = params.colorize !== false

  const w = flt.cols
  const h = flt.rows
  const n = w * h
  const responses: Float32Array[] = []

  for (let t = 0; t < nTheta; t++) {
    const theta = (t * Math.PI) / nTheta
    // Real and imaginary (ψ=π/2) parts give the complex response; its magnitude
    // is phase-invariant, so a bar and its edge score the same.
    const kReal = gaborKernel(ksize, sigma, theta, wavelength, gamma, 0)
    const kImag = gaborKernel(ksize, sigma, theta, wavelength, gamma, Math.PI / 2)
    const matReal = cv.matFromArray(ksize, ksize, cv.CV_32F, Array.from(kReal))
    const matImag = cv.matFromArray(ksize, ksize, cv.CV_32F, Array.from(kImag))
    const respReal = new cv.Mat()
    const respImag = new cv.Mat()
    cv.filter2D(flt, respReal, cv.CV_32F, matReal)
    cv.filter2D(flt, respImag, cv.CV_32F, matImag)
    const re = respReal.data32F
    const im = respImag.data32F
    const magnitude = new Float32Array(n)
    for (let i = 0; i < n; i++) magnitude[i] = Math.hypot(re[i], im[i])
    responses.push(magnitude)
    matReal.delete()
    matImag.delete()
    respReal.delete()
    respImag.delete()
  }
  flt.delete()

  const dominant = new Uint8Array(n)
  const energy = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    let best = 0
    let bestValue = responses[0][i]
    for (let t = 1; t < nTheta; t++) {
      if (responses[t][i] > bestValue) {
        bestValue = responses[t][i]
        best = t
      }
    }
    dominant[i] = best
    energy[i] = bestValue
  }

  const [, energyMax] = extent(energy)
  const [, energyMin] = [0, 0]
  const energyLo = extent(energy)[0]
  const energySpan = energyMax - energyLo || 1
  const energyU8 = new cv.Mat(h, w, cv.CV_8U)
  const energyData = energyU8.data
  for (let i = 0; i < n; i++) energyData[i] = Math.round(((energy[i] - energyLo) / energySpan) * 255)
  void energyMin

  // Hue encodes the winning orientation, value the energy at that orientation.
  const hsvArr = new Array<number>(n * 3)
  for (let i = 0, j = 0; i < n; i++, j += 3) {
    hsvArr[j] = Math.round((dominant[i] / nTheta) * 180)
    hsvArr[j + 1] = 220
    hsvArr[j + 2] = energyData[i]
  }
  const hsv = cv.matFromArray(h, w, cv.CV_8UC3, hsvArr)
  const orientMap = ctx.track(new cv.Mat())
  cv.cvtColor(hsv, orientMap, cv.COLOR_HSV2BGR)
  hsv.delete()

  const energyVis = ctx.track(applyColormap(cv, energyU8, magmaColor))

  // Panels share one brightness scale so they can be compared directly.
  let globalMax = 0
  for (const r of responses) {
    const m = extent(r)[1]
    if (m > globalMax) globalMax = m
  }
  globalMax = globalMax || 1

  const panels: any[] = []
  for (let t = 0; t < nTheta; t++) {
    const scaled = new cv.Mat(h, w, cv.CV_8U)
    const sd = scaled.data
    for (let i = 0; i < n; i++) sd[i] = Math.min(255, Math.round((responses[t][i] / globalMax) * 255))
    const degrees = Math.round(((t * Math.PI) / nTheta) * (180 / Math.PI))
    panels.push(gaborPanel(cv, scaled, `${degrees} deg`, panelPx, colorize))
    scaled.delete()
  }
  energyU8.delete()

  const cols = Math.ceil(Math.sqrt(panels.length))
  const rows = Math.ceil(panels.length / cols)
  const cellH = Math.max(...panels.map((p) => p.rows))
  const cellW = Math.max(...panels.map((p) => p.cols))
  const gap = 4
  const grid = ctx.track(
    new cv.Mat(rows * cellH + (rows + 1) * gap, cols * cellW + (cols + 1) * gap, cv.CV_8UC3, new cv.Scalar(18, 18, 18, 255))
  )
  panels.forEach((panel, index) => {
    const rr = Math.floor(index / cols)
    const cc = index % cols
    const y = gap + rr * (cellH + gap)
    const x = gap + cc * (cellW + gap)
    panel.copyTo(grid.roi(new cv.Rect(x, y, panel.cols, panel.rows)))
    panel.delete()
  })

  const means = responses.map((r) => {
    let sum = 0
    for (let i = 0; i < n; i++) sum += r[i]
    return sum / n
  })
  const sigMax = Math.max(...means) || 1
  const signature = means.map((m) => Math.round((m / sigMax) * 10000) / 10000)

  return { main: orientMap, responses_grid: grid, energy_map: energyVis, signature, n_orientations: nTheta }
}

/* -------------------------------------------------------------- shape gate */

export const featShapeGate: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const maskIn = inputs.mask as any
  const image = inputs.image as any
  if (!maskIn) return { mask_kept: null, mask_rej: null, main: image ?? null, count: 0 }

  const gray = toGray(cv, maskIn)
  const binary = new cv.Mat()
  cv.threshold(gray, binary, 0, 1, cv.THRESH_BINARY)
  gray.delete()

  const w = binary.cols
  const h = binary.rows

  const useCirc = params.use_circularity !== false
  const minCirc = Number(params.min_circularity) ?? 0.35
  const useAspect = !!params.use_aspect
  const maxAspect = Number(params.max_aspect) || 3.0
  const useSolidity = !!params.use_solidity
  const minSolidity = Number(params.min_solidity) ?? 0.8
  const useConvexity = !!params.use_convexity
  const minConvexity = Number(params.min_convexity) ?? 0.8
  const useEccentricity = !!params.use_eccentricity
  const maxEccentricity = Number(params.max_eccentricity) ?? 0.9
  const useRoundness = !!params.use_roundness
  const minRoundness = Number(params.min_roundness) ?? 0.6
  const minSize = Math.max(1, Math.round(Number(params.min_size) || 20))

  const labels = new cv.Mat()
  const nLabels = cv.connectedComponents(binary, labels, 8)
  const labelData = labels.data32S

  const maskKept = ctx.track(cv.Mat.zeros(h, w, cv.CV_8U))
  const maskRej = ctx.track(cv.Mat.zeros(h, w, cv.CV_8U))
  const keptData = maskKept.data
  const rejData = maskRej.data

  interface Info {
    bx: number
    by: number
    bw: number
    bh: number
    passes: boolean
    circ: number
    ar: number
    solidity: number
    convexity: number
    eccentricity: number
    roundness: number
  }
  const infos: Info[] = []
  let kept = 0

  const areas = new Int32Array(nLabels)
  for (let i = 0; i < labelData.length; i++) areas[labelData[i]]++

  for (let label = 1; label < nLabels; label++) {
    const area = areas[label]
    if (area < minSize) continue

    const component = new cv.Mat(h, w, cv.CV_8U)
    const comp = component.data
    for (let i = 0; i < labelData.length; i++) comp[i] = labelData[i] === label ? 255 : 0

    const contours = new cv.MatVector()
    const hierarchy = new cv.Mat()
    cv.findContours(component, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    hierarchy.delete()
    if (contours.size() === 0) {
      contours.delete()
      component.delete()
      continue
    }
    const contour = contours.get(0)

    const perimeter = cv.arcLength(contour, true)
    // C = 4πA/P² — 1 for a perfect disc, falling for elongation or a rough edge.
    const circ = perimeter > 0 ? Math.min(1, (4 * Math.PI * area) / (perimeter * perimeter)) : 0

    const rect = cv.minAreaRect(contour)
    const ww = rect.size.width
    const hh = rect.size.height
    const short = Math.min(ww, hh)
    const long = Math.max(ww, hh)
    const ar = short > 0 ? Math.round((long / short) * 1000) / 1000 : 1

    const hull = new cv.Mat()
    cv.convexHull(contour, hull, false, true)
    const hullArea = cv.contourArea(hull)
    const hullPerimeter = cv.arcLength(hull, true)
    hull.delete()

    const solidity = hullArea > 0 ? Math.round((area / hullArea) * 1000) / 1000 : 1
    const convexity = perimeter > 0 ? Math.round((hullPerimeter / perimeter) * 1000) / 1000 : 1

    // e = √(1 − λ₂/λ₁) from the second-order central moments.
    const m = cv.moments(contour)
    let eccentricity = 0
    if (m.m00 > 0) {
      const term = Math.sqrt((m.mu20 - m.mu02) ** 2 + 4 * m.mu11 ** 2)
      const lam1 = (m.mu20 + m.mu02 + term) / 2
      const lam2 = (m.mu20 + m.mu02 - term) / 2
      if (lam1 > 0 && lam2 >= 0) eccentricity = Math.round(Math.sqrt(1 - lam2 / lam1) * 1000) / 1000
    }

    const roundness = long > 0 ? Math.round(((4 * area) / (Math.PI * long * long)) * 1000) / 1000 : 0

    const passes =
      (!useCirc || circ >= minCirc) &&
      (!useAspect || ar <= maxAspect) &&
      (!useSolidity || solidity >= minSolidity) &&
      (!useConvexity || convexity >= minConvexity) &&
      (!useEccentricity || eccentricity <= maxEccentricity) &&
      (!useRoundness || roundness >= minRoundness)

    const target = passes ? keptData : rejData
    for (let i = 0; i < labelData.length; i++) if (labelData[i] === label) target[i] = 255
    if (passes) kept++

    const box = cv.boundingRect(contour)
    infos.push({ bx: box.x, by: box.y, bw: box.width, bh: box.height, passes, circ, ar, solidity, convexity, eccentricity, roundness })

    contours.delete()
    component.delete()
  }

  labels.delete()
  binary.delete()

  const preview = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))
  const view = preview.data
  if (image) {
    const base = toBgr(cv, image)
    const sized = new cv.Mat()
    if (base.cols !== w || base.rows !== h) cv.resize(base, sized, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR)
    else base.copyTo(sized)
    sized.data.forEach((v: number, i: number) => {
      view[i] = v
    })
    base.delete()
    sized.delete()
  } else {
    for (let i = 0; i < keptData.length; i++) {
      if (keptData[i]) {
        view[i * 3] = 60
        view[i * 3 + 1] = 200
        view[i * 3 + 2] = 60
      } else if (rejData[i]) {
        view[i * 3] = 60
        view[i * 3 + 1] = 60
        view[i * 3 + 2] = 200
      }
    }
  }

  // Rejected regions are dimmed rather than hidden, so the gate's effect is legible.
  for (let i = 0; i < rejData.length; i++) {
    if (!rejData[i]) continue
    view[i * 3] = view[i * 3] * 0.25
    view[i * 3 + 1] = view[i * 3 + 1] * 0.25
    view[i * 3 + 2] = view[i * 3 + 2] * 0.25
  }

  for (const info of infos) {
    const colour = info.passes ? new cv.Scalar(0, 220, 80, 255) : new cv.Scalar(60, 60, 255, 255)
    cv.rectangle(preview, new cv.Point(info.bx, info.by), new cv.Point(info.bx + info.bw, info.by + info.bh), colour, 1)
    const parts: string[] = []
    if (useCirc) parts.push(`C=${info.circ.toFixed(2)}`)
    if (useAspect) parts.push(`E=${info.ar.toFixed(1)}`)
    if (useSolidity) parts.push(`S=${info.solidity.toFixed(2)}`)
    if (useConvexity) parts.push(`Cv=${info.convexity.toFixed(2)}`)
    if (useEccentricity) parts.push(`e=${info.eccentricity.toFixed(2)}`)
    if (useRoundness) parts.push(`Rd=${info.roundness.toFixed(2)}`)
    if (parts.length) {
      cv.putText(preview, parts.join(' '), new cv.Point(info.bx, Math.max(info.by - 4, 10)), cv.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1, cv.LINE_AA)
    }
  }

  cv.putText(preview, `kept: ${kept}`, new cv.Point(8, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, new cv.Scalar(255, 255, 255, 255), 1, cv.LINE_AA)

  return { mask_kept: maskKept, mask_rej: maskRej, main: preview, count: kept }
}
