import type { NodeImpl } from '../types'
import { drawArrowedLine, toBgr, toGray } from '../cvUtils'
import { applyColormap, COLORMAPS } from '../colormaps'

function putLines(cv: any, mat: any, lines: string[], x: number, y0: number, dy: number, scale: number, color: [number, number, number]): void {
  const c = new cv.Scalar(color[0], color[1], color[2], 255)
  lines.forEach((line, i) => cv.putText(mat, line, new cv.Point(x, y0 + i * dy), cv.FONT_HERSHEY_SIMPLEX, scale, c, 1, cv.LINE_AA))
}

function toBinary(cv: any, ctx: any, mask: any): any {
  const gray = ctx.track(toGray(cv, mask))
  const bin = ctx.track(new cv.Mat())
  cv.threshold(gray, bin, 127, 255, cv.THRESH_BINARY)
  return bin
}

// ---------------------------------------------------------------------------
// GLCM
// ---------------------------------------------------------------------------
const GLCM_LEVELS = [8, 16, 32, 64]
const GLCM_ANGLE_SETS = [[0], [0, Math.PI / 2], [0, Math.PI / 4, Math.PI / 2, (3 * Math.PI) / 4]]

export const sciGlcm: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, contrast: 0, homogeneity: 0, energy: 0, entropy: 0, correlation: 0 }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, src))

  const levels = GLCM_LEVELS[Number(params.levels) ?? 1] ?? 16
  const d = Math.round(Number(params.distance) || 1)
  const angles = GLCM_ANGLE_SETS[Math.min(2, Number(params.angles) ?? 2)]
  const symmetric = params.symmetric !== false

  // Quantize, then centre-crop to at most 256x256 (128px half-window) for speed.
  const w0 = gray.cols
  const h0 = gray.rows
  const crop = 128
  const cy = Math.floor(h0 / 2)
  const cx = Math.floor(w0 / 2)
  const y0 = Math.max(0, cy - crop)
  const x0 = Math.max(0, cx - crop)
  const y1 = Math.min(h0, cy + crop)
  const x1 = Math.min(w0, cx + crop)
  const roi = ctx.track(gray.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0)))
  const roiData = roi.data as Uint8Array
  const w = roi.cols
  const h = roi.rows
  const quant = new Uint8Array(w * h)
  for (let i = 0; i < quant.length; i++) quant[i] = Math.min(levels - 1, Math.floor((roiData[i] / 255) * (levels - 1)))

  const glcm = new Float64Array(levels * levels)
  for (const theta of angles) {
    const dx = Math.round(d * Math.cos(theta))
    const dy = Math.round(d * Math.sin(theta))
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const ny = y + dy
        const nx = x + dx
        if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue
        const i = quant[y * w + x]
        const j = quant[ny * w + nx]
        glcm[i * levels + j]++
        if (symmetric) glcm[j * levels + i]++
      }
    }
  }
  let total = 0
  for (let i = 0; i < glcm.length; i++) total += glcm[i]
  if (total > 0) for (let i = 0; i < glcm.length; i++) glcm[i] /= total

  let contrast = 0, homogeneity = 0, energy = 0, entropy = 0
  let muI = 0, muJ = 0
  for (let i = 0; i < levels; i++) {
    for (let j = 0; j < levels; j++) {
      const p = glcm[i * levels + j]
      contrast += (i - j) ** 2 * p
      homogeneity += p / (1 + Math.abs(i - j))
      energy += p * p
      if (p > 0) entropy -= p * Math.log2(p)
      muI += i * p
      muJ += j * p
    }
  }
  let sigI = 0, sigJ = 0
  for (let i = 0; i < levels; i++) {
    for (let j = 0; j < levels; j++) {
      const p = glcm[i * levels + j]
      sigI += (i - muI) ** 2 * p
      sigJ += (j - muJ) ** 2 * p
    }
  }
  sigI = Math.sqrt(sigI)
  sigJ = Math.sqrt(sigJ)
  let correlation = 0
  if (sigI > 1e-9 && sigJ > 1e-9) {
    let sum = 0
    for (let i = 0; i < levels; i++) for (let j = 0; j < levels; j++) sum += (i - muI) * (j - muJ) * glcm[i * levels + j]
    correlation = sum / (sigI * sigJ)
  }

  let maxP = 0
  for (let i = 0; i < glcm.length; i++) if (glcm[i] > maxP) maxP = glcm[i]
  const glcmVis = ctx.track(new cv.Mat(levels, levels, cv.CV_8U))
  const glcmVisData = glcmVis.data as Uint8Array
  for (let i = 0; i < glcm.length; i++) glcmVisData[i] = Math.round((glcm[i] / (maxP + 1e-12)) * 255)
  const visSize = 256
  const resized = ctx.track(new cv.Mat())
  cv.resize(glcmVis, resized, new cv.Size(visSize, visSize), 0, 0, cv.INTER_NEAREST)
  const heatmap = ctx.track(applyColormap(cv, resized, COLORMAPS.Inferno))

  putLines(cv, heatmap, [
    `contrast=${contrast.toFixed(3)}`,
    `homog=${homogeneity.toFixed(3)}`,
    `energy=${energy.toFixed(4)}`,
    `entropy=${entropy.toFixed(3)}`,
    `corr=${correlation.toFixed(3)}`,
  ], 4, 14, 14, 0.35, [255, 255, 255])

  return {
    main: heatmap,
    contrast: Number(contrast.toFixed(4)),
    homogeneity: Number(homogeneity.toFixed(4)),
    energy: Number(energy.toFixed(6)),
    entropy: Number(entropy.toFixed(4)),
    correlation: Number(correlation.toFixed(4)),
  }
}

// ---------------------------------------------------------------------------
// Hausdorff Distance
// ---------------------------------------------------------------------------
function directedHausdorff(pts: { x: number; y: number }[], distMap: Float32Array, w: number, percentile: number): { dist: number; worst: { x: number; y: number } | null } {
  if (pts.length === 0) return { dist: 0, worst: null }
  const dists = pts.map((p) => distMap[p.y * w + p.x])
  const sorted = dists.slice().sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.floor((percentile / 100) * sorted.length))
  const pctDist = sorted[idx]
  let worstIdx = 0
  let best = Infinity
  for (let i = 0; i < dists.length; i++) {
    const diff = Math.abs(dists[i] - pctDist)
    if (diff < best) {
      best = diff
      worstIdx = i
    }
  }
  return { dist: pctDist, worst: pts[worstIdx] }
}

export const sciHausdorff: NodeImpl = (inputs, params, ctx) => {
  const maskA = inputs.mask_a as any
  const maskB = inputs.mask_b as any
  if (!maskA || !maskB) return { main: null, h_ab: 0, h_ba: 0, h_max: 0 }
  const cv = ctx.cv

  let binA = toBinary(cv, ctx, maskA)
  let binB = toBinary(cv, ctx, maskB)
  const H = Math.max(binA.rows, binB.rows)
  const W = Math.max(binA.cols, binB.cols)
  if (binA.rows !== H || binA.cols !== W) {
    const r = ctx.track(new cv.Mat())
    cv.resize(binA, r, new cv.Size(W, H), 0, 0, cv.INTER_NEAREST)
    binA = r
  }
  if (binB.rows !== H || binB.cols !== W) {
    const r = ctx.track(new cv.Mat())
    cv.resize(binB, r, new cv.Size(W, H), 0, 0, cv.INTER_NEAREST)
    binB = r
  }

  const percentile = Number(params.percentile) || 100
  const drawArrow = params.draw_arrow !== false

  const notB = ctx.track(new cv.Mat())
  cv.bitwise_not(binB, notB)
  const notA = ctx.track(new cv.Mat())
  cv.bitwise_not(binA, notA)
  const distToB = ctx.track(new cv.Mat())
  cv.distanceTransform(notB, distToB, cv.DIST_L2, 5)
  const distToA = ctx.track(new cv.Mat())
  cv.distanceTransform(notA, distToA, cv.DIST_L2, 5)

  const collectPts = (bin: any) => {
    const data = bin.data as Uint8Array
    const pts: { x: number; y: number }[] = []
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (data[y * W + x] > 0) pts.push({ x, y })
    return pts
  }
  const ptsA = collectPts(binA)
  const ptsB = collectPts(binB)

  const { dist: hAb, worst: worstA } = directedHausdorff(ptsA, distToB.data32F as Float32Array, W, percentile)
  const { dist: hBa, worst: worstB } = directedHausdorff(ptsB, distToA.data32F as Float32Array, W, percentile)
  const hMax = Math.max(hAb, hBa)

  const bg = inputs.image as any
  let overlay: any
  if (bg) {
    overlay = ctx.track(toBgr(cv, bg))
    if (overlay.rows !== H || overlay.cols !== W) {
      const r = ctx.track(new cv.Mat())
      cv.resize(overlay, r, new cv.Size(W, H), 0, 0, cv.INTER_LINEAR)
      overlay = r
    }
    overlay = ctx.track(overlay.clone())
  } else {
    overlay = ctx.track(new cv.Mat(H, W, cv.CV_8UC3, new cv.Scalar(0, 0, 0)))
  }

  const cntA = new cv.MatVector()
  const hierA = ctx.track(new cv.Mat())
  cv.findContours(binA, cntA, hierA, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  cv.drawContours(overlay, cntA, -1, new cv.Scalar(200, 200, 0, 255), 2)
  cntA.delete()
  const cntB = new cv.MatVector()
  const hierB = ctx.track(new cv.Mat())
  cv.findContours(binB, cntB, hierB, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  cv.drawContours(overlay, cntB, -1, new cv.Scalar(0, 140, 255, 255), 2)
  cntB.delete()

  if (drawArrow && worstA && worstB) {
    const distValA = (distToB.data32F as Float32Array)[worstA.y * W + worstA.x]
    const r = Math.round(distValA) + 2
    const y1s = Math.max(0, worstA.y - r)
    const y2s = Math.min(H, worstA.y + r + 1)
    const x1s = Math.max(0, worstA.x - r)
    const x2s = Math.min(W, worstA.x + r + 1)
    const binBData = binB.data as Uint8Array
    let nearest: { x: number; y: number } | null = null
    let bestD = Infinity
    for (let y = y1s; y < y2s; y++) {
      for (let x = x1s; x < x2s; x++) {
        if (binBData[y * W + x] > 0) {
          const dd = Math.hypot(x - worstA.x, y - worstA.y)
          if (dd < bestD) {
            bestD = dd
            nearest = { x, y }
          }
        }
      }
    }
    if (nearest) {
      drawArrowedLine(cv, overlay, new cv.Point(worstA.x, worstA.y), new cv.Point(nearest.x, nearest.y), new cv.Scalar(0, 0, 220, 255), 2, 0.15)
    }
  }

  let label = `H=${hMax.toFixed(1)}px  A->B=${hAb.toFixed(1)}  B->A=${hBa.toFixed(1)}`
  if (percentile < 100) label = `HD${percentile.toFixed(0)}: ${label}`
  putLines(cv, overlay, [label], 8, 22, 0, 0.5, [255, 255, 255])
  putLines(cv, overlay, ['A'], 8, 44, 0, 0.45, [200, 200, 0])
  putLines(cv, overlay, ['B'], 22, 44, 0, 0.45, [0, 140, 255])

  return { main: overlay, h_ab: Number(hAb.toFixed(2)), h_ba: Number(hBa.toFixed(2)), h_max: Number(hMax.toFixed(2)) }
}

// ---------------------------------------------------------------------------
// Boundary F1
// ---------------------------------------------------------------------------
function boundaryPixels(cv: any, ctx: any, binary: any): any {
  const kernel = cv.getStructuringElement(cv.MORPH_CROSS, new cv.Size(3, 3))
  const eroded = ctx.track(new cv.Mat())
  cv.erode(binary, eroded, kernel)
  kernel.delete()
  const boundary = ctx.track(new cv.Mat())
  cv.subtract(binary, eroded, boundary)
  return boundary
}

function matchRatio(from: Uint8Array, toDilated: Uint8Array): number {
  let nFrom = 0
  let matched = 0
  for (let i = 0; i < from.length; i++) {
    if (from[i] > 0) {
      nFrom++
      if (toDilated[i] > 0) matched++
    }
  }
  return nFrom === 0 ? 1 : matched / nFrom
}

export const sciBoundaryF1: NodeImpl = (inputs, params, ctx) => {
  const pred = inputs.pred as any
  const truth = inputs.truth as any
  if (!pred || !truth) return { main: null, boundary_f1: 0, precision: 0, recall: 0, iou: 0 }
  const cv = ctx.cv

  let binPred = toBinary(cv, ctx, pred)
  let binTruth = toBinary(cv, ctx, truth)
  if (binPred.rows !== binTruth.rows || binPred.cols !== binTruth.cols) {
    const H = Math.max(binPred.rows, binTruth.rows)
    const W = Math.max(binPred.cols, binTruth.cols)
    const rp = ctx.track(new cv.Mat())
    cv.resize(binPred, rp, new cv.Size(W, H), 0, 0, cv.INTER_NEAREST)
    binPred = rp
    const rt = ctx.track(new cv.Mat())
    cv.resize(binTruth, rt, new cv.Size(W, H), 0, 0, cv.INTER_NEAREST)
    binTruth = rt
  }
  const H = binPred.rows
  const W = binPred.cols
  const tol = Math.round(Number(params.tolerance) || 2)

  const bndPred = boundaryPixels(cv, ctx, binPred)
  const bndTruth = boundaryPixels(cv, ctx, binTruth)

  const accKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(2 * tol + 1, 2 * tol + 1))
  const dilatedTruth = ctx.track(new cv.Mat())
  cv.dilate(bndTruth, dilatedTruth, accKernel)
  const dilatedPred = ctx.track(new cv.Mat())
  cv.dilate(bndPred, dilatedPred, accKernel)
  accKernel.delete()

  const pC = matchRatio(bndPred.data as Uint8Array, dilatedTruth.data as Uint8Array)
  const rC = matchRatio(bndTruth.data as Uint8Array, dilatedPred.data as Uint8Array)
  const bf = pC + rC > 0 ? (2 * pC * rC) / (pC + rC) : 0

  const predData = binPred.data as Uint8Array
  const truthData = binTruth.data as Uint8Array
  let inter = 0, union = 0
  for (let i = 0; i < predData.length; i++) {
    const p = predData[i] > 0
    const t = truthData[i] > 0
    if (p && t) inter++
    if (p || t) union++
  }
  const iou = union > 0 ? inter / union : 1

  const bgImg = inputs.image as any
  let base: any
  if (bgImg) {
    base = ctx.track(toBgr(cv, bgImg))
    if (base.rows !== H || base.cols !== W) {
      const r = ctx.track(new cv.Mat())
      cv.resize(base, r, new cv.Size(W, H), 0, 0, cv.INTER_LINEAR)
      base = r
    }
  } else {
    base = ctx.track(new cv.Mat(H, W, cv.CV_8UC3, new cv.Scalar(0, 0, 0)))
  }
  const overlay = ctx.track(base.clone())

  const matchedLayer = new Uint8Array(H * W)
  const bndPredData = bndPred.data as Uint8Array
  const dilatedTruthData = dilatedTruth.data as Uint8Array
  for (let i = 0; i < matchedLayer.length; i++) matchedLayer[i] = bndPredData[i] > 0 && dilatedTruthData[i] > 0 ? 255 : 0
  const matchedMat = ctx.track(new cv.Mat(H, W, cv.CV_8U))
  ;(matchedMat.data as Uint8Array).set(matchedLayer)

  const lw = Math.round(Number(params.line_width) || 0)
  const thick = lw > 0 ? lw : Math.max(2, Math.round(Math.max(H, W) / 400))
  const drawKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(2 * thick + 1, 2 * thick + 1))
  const truthDraw = ctx.track(new cv.Mat())
  cv.dilate(bndTruth, truthDraw, drawKernel)
  const predDraw = ctx.track(new cv.Mat())
  cv.dilate(bndPred, predDraw, drawKernel)
  const matchedDraw = ctx.track(new cv.Mat())
  cv.dilate(matchedMat, matchedDraw, drawKernel)
  drawKernel.delete()

  const overlayData = overlay.data as Uint8Array
  const truthDrawData = truthDraw.data as Uint8Array
  const predDrawData = predDraw.data as Uint8Array
  const matchedDrawData = matchedDraw.data as Uint8Array
  for (let i = 0, p = 0; i < truthDrawData.length; i++, p += 3) {
    if (truthDrawData[i] > 0) {
      overlayData[p] = 0; overlayData[p + 1] = 140; overlayData[p + 2] = 255
    }
    if (predDrawData[i] > 0) {
      overlayData[p] = 200; overlayData[p + 1] = 200; overlayData[p + 2] = 0
    }
    if (matchedDrawData[i] > 0) {
      overlayData[p] = 0; overlayData[p + 1] = 200; overlayData[p + 2] = 0
    }
  }

  const label = `BF=${bf.toFixed(3)}  IoU=${iou.toFixed(3)}  P_c=${pC.toFixed(3)}  R_c=${rC.toFixed(3)}  tol=${tol}px`
  putLines(cv, overlay, [label], 8, 20, 0, 0.45, [255, 255, 255])
  putLines(cv, overlay, ['pred'], 8, H - 36, 0, 0.38, [200, 200, 0])
  putLines(cv, overlay, ['truth'], 42, H - 36, 0, 0.38, [0, 140, 255])
  putLines(cv, overlay, ['match'], 82, H - 36, 0, 0.38, [0, 200, 0])

  return { main: overlay, boundary_f1: Number(bf.toFixed(4)), precision: Number(pC.toFixed(4)), recall: Number(rC.toFixed(4)), iou: Number(iou.toFixed(4)) }
}

// ---------------------------------------------------------------------------
// Robust Box Fit
// ---------------------------------------------------------------------------
const MAD_TO_STD = 1.4826

function median(arr: number[]): number {
  const sorted = arr.slice().sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function ordinaryBox(xs: number[], ys: number[]): [number, number, number, number] {
  const xmin = Math.min(...xs), xmax = Math.max(...xs)
  const ymin = Math.min(...ys), ymax = Math.max(...ys)
  return [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]
}

export const sciRobustBbox: NodeImpl = (inputs, params, ctx) => {
  const mask = inputs.mask as any
  const empty = { main: null, box_x: 0, box_y: 0, box_w: 0, box_h: 0, n_rejected: 0 }
  if (!mask) return empty
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, mask))
  const data = gray.data as Uint8Array
  const w = gray.cols
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i < data.length; i++) {
    if (data[i] > 127) {
      xs.push(i % w)
      ys.push(Math.floor(i / w))
    }
  }
  if (xs.length === 0) return empty

  const mode = Number(params.mode) ?? 1
  const tol = Number(params.tolerance) || 3.0

  const ordBox = ordinaryBox(xs, ys)
  const mx = median(xs)
  const my = median(ys)
  const madx = median(xs.map((x) => Math.abs(x - mx))) || 1
  const mady = median(ys.map((y) => Math.abs(y - my))) || 1
  const keep = xs.map((x, i) => Math.abs(x - mx) <= tol * MAD_TO_STD * madx && Math.abs(ys[i] - my) <= tol * MAD_TO_STD * mady)
  const anyKept = keep.some(Boolean)
  const kx = anyKept ? xs.filter((_, i) => keep[i]) : xs
  const ky = anyKept ? ys.filter((_, i) => keep[i]) : ys
  const robBox = ordinaryBox(kx, ky)
  const nRejected = anyKept ? keep.filter((k) => !k).length : 0

  const chosen = mode === 1 ? robBox : ordBox

  const bg = inputs.image as any
  const H = gray.rows
  const W = gray.cols
  let base: any
  if (bg) {
    base = ctx.track(toBgr(cv, bg))
    if (base.rows !== H || base.cols !== W) {
      const r = ctx.track(new cv.Mat())
      cv.resize(base, r, new cv.Size(W, H), 0, 0, cv.INTER_LINEAR)
      base = r
    }
  } else {
    base = ctx.track(new cv.Mat())
    cv.cvtColor(gray, base, cv.COLOR_GRAY2BGR)
  }
  const overlay = ctx.track(base.clone())
  const [ox, oy, ow, oh] = ordBox
  cv.rectangle(overlay, new cv.Point(ox, oy), new cv.Point(ox + ow, oy + oh), new cv.Scalar(0, 0, 255, 255), 1, cv.LINE_AA)
  const [rx, ry, rw, rh] = robBox
  cv.rectangle(overlay, new cv.Point(rx, ry), new cv.Point(rx + rw, ry + rh), new cv.Scalar(0, 220, 0, 255), 2, cv.LINE_AA)

  const label = `${mode === 1 ? 'Robust' : 'Ordinary'}  rejected=${nRejected}/${xs.length}  tol=${tol.toFixed(1)}`
  const black = new cv.Scalar(0, 0, 0, 255)
  const white = new cv.Scalar(255, 255, 255, 255)
  cv.putText(overlay, label, new cv.Point(8, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 3, cv.LINE_AA)
  cv.putText(overlay, label, new cv.Point(8, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv.LINE_AA)
  putLines(cv, overlay, ['red=ordinary'], 8, H - 30, 0, 0.38, [0, 0, 255])
  putLines(cv, overlay, ['green=robust'], 8, H - 14, 0, 0.38, [0, 220, 0])

  const [cx2, cy2, cw2, ch2] = chosen
  return { main: overlay, box_x: cx2, box_y: cy2, box_w: cw2, box_h: ch2, n_rejected: nRejected }
}

// ---------------------------------------------------------------------------
// Region Classifier
// ---------------------------------------------------------------------------
const CLASS_COLORS: [number, number, number][] = [
  [80, 200, 80], [80, 80, 220], [0, 200, 240], [220, 120, 40],
  [180, 80, 220], [80, 220, 200], [220, 160, 60], [200, 80, 140],
]

export const sciRegionClassifier: NodeImpl = (inputs, params, ctx) => {
  const regions = (inputs.regions as Record<string, unknown>[]) ?? []
  const labelsMap = inputs.labels_map as any
  const img = inputs.image as any
  const code = String(params.code ?? "return 'Other'")
  const showLabels = params.show_labels !== false
  const outline = !!params.outline_only

  if (regions.length === 0) return { regions_out: [], counts: {}, overlay: img ?? null }

  const cv = ctx.cv
  let fn: ((r: Record<string, unknown>) => unknown) | null = null
  try {
    fn = new Function('r', code) as (r: Record<string, unknown>) => unknown
  } catch {
    fn = null
  }

  const classColorMap = new Map<string, [number, number, number]>()
  let colorIdx = 0
  const classified: Record<string, unknown>[] = []
  const counts: Record<string, number> = {}

  for (const r of regions) {
    let cls: string
    try {
      cls = fn ? String(fn(r) ?? 'Unknown') : 'Unknown'
    } catch (e) {
      cls = `Error: ${e instanceof Error ? e.message : String(e)}`
    }
    const newR = { ...r, class: cls }
    classified.push(newR)
    counts[cls] = (counts[cls] ?? 0) + 1
    if (!classColorMap.has(cls)) {
      classColorMap.set(cls, CLASS_COLORS[colorIdx % CLASS_COLORS.length])
      colorIdx++
    }
  }

  let overlay: any = img ?? null
  if (img && labelsMap) {
    const src = ctx.track(toBgr(cv, img))
    let labelData = labelsMap.data32S as Int32Array
    let lw = labelsMap.cols
    let lh = labelsMap.rows
    if (lw !== src.cols || lh !== src.rows) {
      const resized = ctx.track(new cv.Mat())
      cv.resize(labelsMap, resized, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_NEAREST)
      labelData = resized.data32S as Int32Array
      lw = resized.cols
      lh = resized.rows
    }
    overlay = ctx.track(src.clone())

    for (const cr of classified) {
      const lid = cr.label_id as number | undefined
      if (lid === undefined) continue
      const cls = String(cr.class ?? 'Unknown')
      const [b, g, r2] = classColorMap.get(cls) ?? [200, 200, 200]
      const color = new cv.Scalar(b, g, r2, 255)

      const regionMask = ctx.track(new cv.Mat(lh, lw, cv.CV_8U, new cv.Scalar(0)))
      const maskData = regionMask.data as Uint8Array
      let sx = 0, sy = 0, n = 0
      for (let i = 0; i < labelData.length; i++) {
        if (labelData[i] === lid) {
          maskData[i] = 255
          sx += i % lw
          sy += Math.floor(i / lw)
          n++
        }
      }
      if (n === 0) continue

      const contours = new cv.MatVector()
      const hierarchy = ctx.track(new cv.Mat())
      cv.findContours(regionMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
      if (contours.size() === 0) {
        contours.delete()
        continue
      }
      if (outline) {
        cv.drawContours(overlay, contours, -1, color, 2)
      } else {
        const fill = ctx.track(new cv.Mat(lh, lw, cv.CV_8UC3, new cv.Scalar(0, 0, 0)))
        cv.drawContours(fill, contours, -1, color, -1)
        const blended = ctx.track(new cv.Mat())
        cv.addWeighted(overlay, 0.7, fill, 0.3, 0, blended)
        blended.copyTo(overlay)
        cv.drawContours(overlay, contours, -1, color, 1)
      }
      contours.delete()

      if (showLabels) {
        const centroid = cr.centroid as [number, number] | undefined
        const cxr = centroid ? Math.round(centroid[0]) : Math.round(sx / n)
        const cyr = centroid ? Math.round(centroid[1]) : Math.round(sy / n)
        cv.putText(overlay, cls, new cv.Point(cxr - 14, cyr + 4), cv.FONT_HERSHEY_SIMPLEX, 0.38, new cv.Scalar(255, 255, 255, 255), 1, cv.LINE_AA)
      }
    }

    let lx = overlay.cols - 140
    let ly = 10
    for (const [cls, color] of classColorMap) {
      cv.rectangle(overlay, new cv.Point(lx, ly), new cv.Point(lx + 14, ly + 14), new cv.Scalar(color[0], color[1], color[2], 255), -1)
      cv.putText(overlay, `${cls} (${counts[cls] ?? 0})`, new cv.Point(lx + 18, ly + 11), cv.FONT_HERSHEY_SIMPLEX, 0.35, new cv.Scalar(220, 220, 220, 255), 1, cv.LINE_AA)
      ly += 18
    }
  }

  return { regions_out: classified, counts, overlay }
}

// ---------------------------------------------------------------------------
// Cluster Heatmap
// ---------------------------------------------------------------------------
const CLUSTER_CMAP_NAMES = ['Viridis', 'Plasma', 'Turbo', 'Jet', 'Hot', 'Cool', 'Inferno', 'Magma']
const ID_KEYS = ['id', 'label', 'cluster_id', 'region_id', 'idx']

export const sciClusterHeatmap: NodeImpl = (inputs, params, ctx) => {
  const labels = inputs.labels_map as any
  const regions = (inputs.regions as Record<string, unknown>[]) ?? []
  const img = inputs.image as any
  if (!labels || regions.length === 0) return { main: img ?? null }
  const cv = ctx.cv

  const featName = String(params.feature ?? 'area').trim() || 'area'
  const cmapName = CLUSTER_CMAP_NAMES[Number(params.colormap) || 0] ?? 'Viridis'
  const cmapFn = COLORMAPS[cmapName] ?? COLORMAPS.Viridis
  const alpha = Number(params.alpha) ?? 0.85
  const bgAlpha = Number(params.bg_alpha) ?? 0.25
  const showValues = !!params.show_values
  const showColorbar = params.colorbar !== false

  const labelData = labels.data32S as Int32Array
  const h = labels.rows
  const w = labels.cols

  const lblToVal = new Map<number, number>()
  for (const r of regions) {
    if (!r || typeof r !== 'object' || !(featName in r)) continue
    let id: number | null = null
    for (const k of ID_KEYS) {
      if (k in r) {
        const v = Number(r[k])
        if (Number.isFinite(v)) { id = v; break }
      }
    }
    if (id === null) continue
    const val = Number(r[featName])
    if (Number.isFinite(val)) lblToVal.set(id, val)
  }
  if (lblToVal.size === 0) return { main: img ?? null }

  const vals = [...lblToVal.values()]
  const vmin = Math.min(...vals)
  const vmax = Math.max(...vals)
  const vrange = vmax > vmin ? vmax - vmin : 1

  const normU8 = ctx.track(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0)))
  const normData = normU8.data as Uint8Array
  const validPix = new Uint8Array(h * w)
  for (let i = 0; i < labelData.length; i++) {
    const val = lblToVal.get(labelData[i])
    if (val !== undefined) {
      normData[i] = Math.max(0, Math.min(255, Math.round(((val - vmin) / vrange) * 255)))
      validPix[i] = 1
    }
  }
  const colored = ctx.track(applyColormap(cv, normU8, cmapFn))
  const coloredData = colored.data as Uint8Array
  for (let i = 0; i < validPix.length; i++) {
    if (!validPix[i]) {
      coloredData[i * 3] = 0
      coloredData[i * 3 + 1] = 0
      coloredData[i * 3 + 2] = 0
    }
  }

  let out: any
  if (img) {
    const base = ctx.track(toBgr(cv, img))
    let baseFit = base
    if (base.rows !== h || base.cols !== w) {
      const r = ctx.track(new cv.Mat())
      cv.resize(base, r, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR)
      baseFit = r
    }
    out = ctx.track(new cv.Mat(h, w, cv.CV_8UC3))
    const baseData = baseFit.data as Uint8Array
    const outData = out.data as Uint8Array
    for (let i = 0; i < outData.length; i++) outData[i] = Math.max(0, Math.min(255, Math.round(baseData[i] * bgAlpha + coloredData[i] * alpha)))
  } else {
    out = ctx.track(colored.clone())
  }

  if (showValues) {
    for (const r of regions) {
      if (!r || typeof r !== 'object' || !(featName in r)) continue
      const cx = Math.round(Number(r.centroid_x) || 0)
      const cy = Math.round(Number(r.centroid_y) || 0)
      const val = r[featName]
      const txt = typeof val === 'number' ? val.toFixed(1) : String(val)
      cv.putText(out, txt, new cv.Point(cx - 12, cy + 4), cv.FONT_HERSHEY_SIMPLEX, 0.32, new cv.Scalar(255, 255, 255, 255), 1, cv.LINE_AA)
    }
  }

  if (showColorbar && h > 40) {
    const barW = 12
    const barH = Math.max(40, h - 30)
    const barX = w - barW - 8
    const barY = 12
    for (let y = 0; y < barH; y++) {
      const v = Math.round((1 - y / barH) * 255)
      const [r, g, b] = cmapFn(v)
      cv.rectangle(out, new cv.Point(barX, barY + y), new cv.Point(barX + barW, barY + y + 1), new cv.Scalar(b, g, r, 255), -1)
    }
    cv.rectangle(out, new cv.Point(barX, barY), new cv.Point(barX + barW, barY + barH), new cv.Scalar(180, 180, 180, 255), 1)
    const white = [240, 240, 240] as [number, number, number]
    putLines(cv, out, [vmax.toPrecision(3)], barX - 1, barY + 9, 0, 0.28, white)
    putLines(cv, out, [vmin.toPrecision(3)], barX - 1, barY + barH + 1, 0, 0.28, white)
    putLines(cv, out, [featName.slice(0, 8)], barX - 2, barY + barH + 12, 0, 0.26, [180, 180, 180])
  }

  return { main: out }
}
