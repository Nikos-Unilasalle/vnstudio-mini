import type { NodeImpl } from '../types'
import { toBgr, toGray } from '../cvUtils'
import { applyColormap, jetColor, viridisColor } from '../colormaps'

// ---------------------------------------------------------------------------
// K-Means Segmentation
// ---------------------------------------------------------------------------
export const cvKmeansSegmentation: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, k_used: 0 }
  const cv = ctx.cv

  const bgr = ctx.track(toBgr(cv, src))
  const k = Math.max(2, Math.round(Number(params.k) || 4))
  const useLab = (Number(params.color_space) ?? 1) === 1
  const attempts = Math.max(1, Math.round(Number(params.attempts) || 3))
  const maxIter = Math.max(1, Math.round(Number(params.max_iter) || 100))

  const work = ctx.track(new cv.Mat())
  if (useLab) cv.cvtColor(bgr, work, cv.COLOR_BGR2Lab)
  else bgr.copyTo(work)

  const n = work.rows * work.cols
  const samples = ctx.track(new cv.Mat(n, 3, cv.CV_32F))
  const srcBytes = work.data as Uint8Array
  const sampleData = samples.data32F as Float32Array
  for (let i = 0; i < n; i++) {
    sampleData[i * 3] = srcBytes[i * 3]
    sampleData[i * 3 + 1] = srcBytes[i * 3 + 1]
    sampleData[i * 3 + 2] = srcBytes[i * 3 + 2]
  }

  const labels = ctx.track(new cv.Mat())
  const centers = ctx.track(new cv.Mat())
  // TermCriteria type flags aren't exposed as named constants in this build; 3 = COUNT|EPS.
  const criteria = new cv.TermCriteria(3, maxIter, 0.2)
  cv.kmeans(samples, k, labels, criteria, attempts, cv.KMEANS_PP_CENTERS, centers)

  const centersData = centers.data32F as Float32Array
  const labelsData = labels.data32S as Int32Array
  const segmented = ctx.track(new cv.Mat(work.rows, work.cols, cv.CV_8UC3))
  const outData = segmented.data as Uint8Array
  for (let i = 0; i < n; i++) {
    const label = labelsData[i]
    outData[i * 3] = Math.round(centersData[label * 3])
    outData[i * 3 + 1] = Math.round(centersData[label * 3 + 1])
    outData[i * 3 + 2] = Math.round(centersData[label * 3 + 2])
  }

  let result = segmented
  if (useLab) {
    const converted = ctx.track(new cv.Mat())
    cv.cvtColor(segmented, converted, cv.COLOR_Lab2BGR)
    result = converted
  }

  return { main: result, k_used: k }
}

// ---------------------------------------------------------------------------
// Local Binary Pattern
// ---------------------------------------------------------------------------
type LbpMethod = 'uniform' | 'default' | 'ror' | 'var'

function bilinearSample(data: Uint8Array, w: number, h: number, x: number, y: number): number {
  if (x < 0 || y < 0 || x > w - 1 || y > h - 1) return 0
  const x0 = Math.floor(x)
  const y0 = Math.floor(y)
  const x1 = Math.min(w - 1, x0 + 1)
  const y1 = Math.min(h - 1, y0 + 1)
  const fx = x - x0
  const fy = y - y0
  const v00 = data[y0 * w + x0]
  const v10 = data[y0 * w + x1]
  const v01 = data[y1 * w + x0]
  const v11 = data[y1 * w + x1]
  return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy
}

export const cvLbp: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, hist_image: null, data: null }
  const cv = ctx.cv

  const gray = ctx.track(toGray(cv, src))
  const w = gray.cols
  const h = gray.rows
  const data = gray.data as Uint8Array

  const P = Math.max(4, Math.round(Number(params.points) || 8))
  const R = Number(params.radius) || 1.0
  const method = (['uniform', 'default', 'ror', 'var'].includes(String(params.method)) ? params.method : 'uniform') as LbpMethod
  const showHistogram = params.show_histogram !== false

  const offsets: [number, number][] = []
  for (let p = 0; p < P; p++) {
    const theta = (2 * Math.PI * p) / P
    offsets.push([R * Math.cos(theta), R * Math.sin(theta)])
  }

  const lbp = new Float64Array(w * h)
  const margin = Math.ceil(R)

  for (let y = margin; y < h - margin; y++) {
    for (let x = margin; x < w - margin; x++) {
      const center = data[y * w + x]
      const samples: number[] = []
      for (const [dx, dy] of offsets) samples.push(bilinearSample(data, w, h, x + dx, y + dy))

      if (method === 'var') {
        const mean = samples.reduce((a, b) => a + b, 0) / P
        const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / P
        lbp[y * w + x] = variance
        continue
      }

      const bits = samples.map((s) => (s >= center ? 1 : 0))

      if (method === 'default') {
        let code = 0
        for (let p = 0; p < P; p++) code += bits[p] << p
        lbp[y * w + x] = code
      } else if (method === 'ror') {
        let min = Infinity
        for (let r = 0; r < P; r++) {
          let code = 0
          for (let p = 0; p < P; p++) code += bits[(p + r) % P] << p
          if (code < min) min = code
        }
        lbp[y * w + x] = min
      } else {
        // uniform: <=2 circular 0/1 transitions -> count of 1 bits, else P+1
        let transitions = 0
        for (let p = 0; p < P; p++) if (bits[p] !== bits[(p + 1) % P]) transitions++
        const ones = bits.reduce((a, b) => a + b, 0)
        lbp[y * w + x] = transitions <= 2 ? ones : P + 1
      }
    }
  }

  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < lbp.length; i++) {
    if (lbp[i] < lo) lo = lbp[i]
    if (lbp[i] > hi) hi = lbp[i]
  }

  const lbpNorm = ctx.track(new cv.Mat(h, w, cv.CV_8U))
  const normData = lbpNorm.data as Uint8Array
  const range = hi > lo ? hi - lo : 1
  for (let i = 0; i < lbp.length; i++) normData[i] = hi > lo ? Math.round(((lbp[i] - lo) / range) * 255) : 0

  const lbpColor = ctx.track(applyColormap(cv, lbpNorm, jetColor))

  // Histogram of LBP codes.
  let nBins: number
  let hist: number[]
  if (method === 'var') {
    nBins = 64
    hist = new Array(nBins).fill(0)
    const binRange = hi > lo ? hi - lo : 1
    for (let i = 0; i < lbp.length; i++) {
      const bin = Math.min(nBins - 1, Math.floor(((lbp[i] - lo) / binRange) * nBins))
      hist[bin]++
    }
  } else {
    nBins = Math.max(1, Math.round(hi) + 1)
    hist = new Array(nBins).fill(0)
    for (let i = 0; i < lbp.length; i++) hist[Math.min(nBins - 1, Math.round(lbp[i]))]++
  }
  const total = hist.reduce((a, b) => a + b, 0)
  const normalizedHist = total > 0 ? hist.map((v) => v / total) : hist

  let histImage: any = null
  if (showHistogram) histImage = ctx.track(renderHistogramBars(cv, normalizedHist))

  return {
    main: lbpColor,
    hist_image: histImage,
    data: { histogram: normalizedHist, n_bins: nBins },
  }
}

function renderHistogramBars(cv: any, hist: number[]): any {
  const W = 512
  const H = 256
  const margin = 20
  const canvas = new cv.Mat(H, W, cv.CV_8UC3, new cv.Scalar(30, 30, 30))
  const n = hist.length
  if (n === 0) return canvas

  const plotW = W - 2 * margin
  const plotH = H - 2 * margin
  const maxVal = Math.max(...hist, 1e-9)
  const barW = Math.max(1, Math.floor(plotW / n))
  const barColor = new cv.Scalar(0, 200, 255, 255)

  for (let i = 0; i < n; i++) {
    const barH = Math.round((hist[i] / maxVal) * plotH)
    const x0 = margin + i * barW
    const x1 = x0 + Math.max(1, barW - 1)
    const y0 = H - margin
    const y1 = y0 - barH
    cv.rectangle(canvas, new cv.Point(x0, y1), new cv.Point(x1, y0), barColor, -1)
  }
  cv.line(canvas, new cv.Point(margin, H - margin), new cv.Point(W - margin, H - margin), new cv.Scalar(200, 200, 200, 255), 1)
  return canvas
}

// ---------------------------------------------------------------------------
// SSIM / PSNR
// ---------------------------------------------------------------------------
function ssimMap(cv: any, ctx: any, a: any, b: any): { map: any; score: number } {
  const C1 = (0.01 * 255) ** 2
  const C2 = (0.03 * 255) ** 2
  const winSize = 7

  const af = ctx.track(new cv.Mat())
  const bf = ctx.track(new cv.Mat())
  a.convertTo(af, cv.CV_32F)
  b.convertTo(bf, cv.CV_32F)

  const box = (m: any) => {
    const out = ctx.track(new cv.Mat())
    cv.boxFilter(m, out, cv.CV_32F, new cv.Size(winSize, winSize), new cv.Point(-1, -1), true, cv.BORDER_REFLECT)
    return out
  }

  const muA = box(af)
  const muB = box(bf)

  const aa = ctx.track(new cv.Mat())
  const bb = ctx.track(new cv.Mat())
  const ab = ctx.track(new cv.Mat())
  cv.multiply(af, af, aa)
  cv.multiply(bf, bf, bb)
  cv.multiply(af, bf, ab)

  const muAA = box(aa)
  const muBB = box(bb)
  const muAB = box(ab)

  const n = muA.rows * muA.cols
  const muAd = muA.data32F as Float32Array
  const muBd = muB.data32F as Float32Array
  const muAAd = muAA.data32F as Float32Array
  const muBBd = muBB.data32F as Float32Array
  const muABd = muAB.data32F as Float32Array

  const map = ctx.track(new cv.Mat(muA.rows, muA.cols, cv.CV_32F))
  const mapData = map.data32F as Float32Array

  let sum = 0
  for (let i = 0; i < n; i++) {
    const muAi = muAd[i]
    const muBi = muBd[i]
    const varA = muAAd[i] - muAi * muAi
    const varB = muBBd[i] - muBi * muBi
    const covAB = muABd[i] - muAi * muBi
    const numerator = (2 * muAi * muBi + C1) * (2 * covAB + C2)
    const denominator = (muAi * muAi + muBi * muBi + C1) * (varA + varB + C2)
    const val = denominator !== 0 ? numerator / denominator : 1
    mapData[i] = val
    sum += val
  }

  return { map, score: sum / n }
}

export const cvSsim: NodeImpl = (inputs, params, ctx) => {
  const img = inputs.image as any
  const ref = inputs.reference as any
  if (!img || !ref) return { main: null, ssim: 0, psnr: 0, data: null }
  const cv = ctx.cv

  const outputMode = String(params.output ?? 'SSIM Map')
  const useGray = params.grayscale !== false

  const imgBgr = ctx.track(toBgr(cv, img))
  let refBgr = ctx.track(toBgr(cv, ref))
  if (refBgr.cols !== imgBgr.cols || refBgr.rows !== imgBgr.rows) {
    const resized = ctx.track(new cv.Mat())
    cv.resize(refBgr, resized, new cv.Size(imgBgr.cols, imgBgr.rows), 0, 0, cv.INTER_LINEAR)
    refBgr = resized
  }

  let score: number
  let map: any
  if (useGray) {
    const a = ctx.track(toGray(cv, imgBgr))
    const b = ctx.track(toGray(cv, refBgr))
    ;({ map, score } = ssimMap(cv, ctx, a, b))
  } else {
    // Average per-channel SSIM maps, matching skimage's channel_axis mean-collapse.
    const channelsA = new cv.MatVector()
    const channelsB = new cv.MatVector()
    cv.split(imgBgr, channelsA)
    cv.split(refBgr, channelsB)
    let acc: Float32Array | null = null
    let dims = { rows: 0, cols: 0 }
    let scoreSum = 0
    for (let c = 0; c < 3; c++) {
      const { map: chMap, score: chScore } = ssimMap(cv, ctx, channelsA.get(c), channelsB.get(c))
      scoreSum += chScore
      const chData = chMap.data32F as Float32Array
      if (!acc) {
        acc = new Float32Array(chData.length)
        dims = { rows: chMap.rows, cols: chMap.cols }
      }
      for (let i = 0; i < chData.length; i++) acc[i] += chData[i] / 3
    }
    channelsA.delete()
    channelsB.delete()
    score = scoreSum / 3
    map = ctx.track(new cv.Mat(dims.rows, dims.cols, cv.CV_32F))
    ;(map.data32F as Float32Array).set(acc!)
  }

  // PSNR / MSE over the full BGR images.
  const dataA = imgBgr.data as Uint8Array
  const dataB = refBgr.data as Uint8Array
  let sqErr = 0
  for (let i = 0; i < dataA.length; i++) {
    const d = dataA[i] - dataB[i]
    sqErr += d * d
  }
  const mse = sqErr / dataA.length
  const psnr = mse > 0 ? 10 * Math.log10((255 * 255) / mse) : 100

  let result: any
  if (outputMode === 'SSIM Map') {
    const normalized = ctx.track(new cv.Mat(map.rows, map.cols, cv.CV_8U))
    const src = map.data32F as Float32Array
    const dst = normalized.data as Uint8Array
    for (let i = 0; i < src.length; i++) dst[i] = Math.max(0, Math.min(255, Math.round(((src[i] + 1) * 0.5) * 255)))
    result = ctx.track(applyColormap(cv, normalized, viridisColor))
  } else if (outputMode === 'Difference') {
    const diff = ctx.track(new cv.Mat())
    cv.absdiff(imgBgr, refBgr, diff)
    result = diff
  } else {
    result = ctx.track(imgBgr.clone())
  }

  const line1 = `SSIM: ${score.toFixed(4)}`
  const line2 = `PSNR: ${psnr.toFixed(2)} dB`
  const outline = new cv.Scalar(0, 0, 0, 255)
  const fill = new cv.Scalar(255, 255, 255, 255)
  cv.putText(result, line1, new cv.Point(10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, outline, 4, cv.LINE_AA)
  cv.putText(result, line1, new cv.Point(10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, fill, 1, cv.LINE_AA)
  cv.putText(result, line2, new cv.Point(10, 58), cv.FONT_HERSHEY_SIMPLEX, 0.7, outline, 4, cv.LINE_AA)
  cv.putText(result, line2, new cv.Point(10, 58), cv.FONT_HERSHEY_SIMPLEX, 0.7, fill, 1, cv.LINE_AA)

  return { main: result, ssim: score, psnr, data: { ssim: score, psnr, mse } }
}

// ---------------------------------------------------------------------------
// Skeletonize (Zhang-Suen thinning)
// ---------------------------------------------------------------------------
function zhangSuenThin(bin: Uint8Array, w: number, h: number): Uint8Array {
  const img = new Uint8Array(bin)
  const at = (x: number, y: number) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : img[y * w + x])

  let changed = true
  while (changed) {
    changed = false
    for (let step = 0; step < 2; step++) {
      const toRemove: number[] = []
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          if (!img[y * w + x]) continue
          const p2 = at(x, y - 1)
          const p3 = at(x + 1, y - 1)
          const p4 = at(x + 1, y)
          const p5 = at(x + 1, y + 1)
          const p6 = at(x, y + 1)
          const p7 = at(x - 1, y + 1)
          const p8 = at(x - 1, y)
          const p9 = at(x - 1, y - 1)
          const neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
          const B = neighbors.reduce((a, b) => a + b, 0)
          if (B < 2 || B > 6) continue

          let A = 0
          for (let i = 0; i < 8; i++) if (neighbors[i] === 0 && neighbors[(i + 1) % 8] === 1) A++
          if (A !== 1) continue

          if (step === 0) {
            if (p2 * p4 * p6 !== 0) continue
            if (p4 * p6 * p8 !== 0) continue
          } else {
            if (p2 * p4 * p8 !== 0) continue
            if (p2 * p6 * p8 !== 0) continue
          }
          toRemove.push(y * w + x)
        }
      }
      if (toRemove.length > 0) {
        changed = true
        for (const idx of toRemove) img[idx] = 0
      }
    }
  }
  return img
}

export const cvSkeletonize: NodeImpl = (inputs, params, ctx) => {
  const mask = inputs.mask as any
  if (!mask) return { main: null, preview: null, length_px: 0 }
  const cv = ctx.cv

  let gray = ctx.track(toGray(cv, mask))
  const closeSize = Math.max(0, Math.round(Number(params.close_holes) ?? 3))
  if (closeSize > 0) {
    const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(closeSize, closeSize))
    const closed = ctx.track(new cv.Mat())
    cv.morphologyEx(gray, closed, cv.MORPH_CLOSE, kernel)
    kernel.delete()
    gray = closed
  }

  const binary = ctx.track(new cv.Mat())
  cv.threshold(gray, binary, 0, 255, cv.THRESH_BINARY)

  const minSize = Math.max(0, Math.round(Number(params.min_size) ?? 32))
  let cleaned = binary
  if (minSize > 0) {
    const labels = ctx.track(new cv.Mat())
    cv.connectedComponents(binary, labels, 8, cv.CV_32S)
    const labelData = labels.data32S as Int32Array
    const areas = new Map<number, number>()
    for (let i = 0; i < labelData.length; i++) {
      const l = labelData[i]
      if (l > 0) areas.set(l, (areas.get(l) ?? 0) + 1)
    }
    const kept = ctx.track(new cv.Mat(binary.rows, binary.cols, cv.CV_8U, new cv.Scalar(0)))
    const keptData = kept.data as Uint8Array
    for (let i = 0; i < labelData.length; i++) {
      const l = labelData[i]
      if (l > 0 && (areas.get(l) ?? 0) >= minSize) keptData[i] = 255
    }
    cleaned = kept
  }

  const cleanedBits = new Uint8Array(cleaned.rows * cleaned.cols)
  const cleanedData = cleaned.data as Uint8Array
  for (let i = 0; i < cleanedData.length; i++) cleanedBits[i] = cleanedData[i] > 0 ? 1 : 0

  const thinned = zhangSuenThin(cleanedBits, cleaned.cols, cleaned.rows)

  const skeleton = ctx.track(new cv.Mat(cleaned.rows, cleaned.cols, cv.CV_8U))
  const skelData = skeleton.data as Uint8Array
  let length = 0
  for (let i = 0; i < thinned.length; i++) {
    skelData[i] = thinned[i] ? 255 : 0
    if (thinned[i]) length++
  }

  const preview = ctx.track(new cv.Mat())
  cv.cvtColor(cleaned, preview, cv.COLOR_GRAY2BGR)
  const previewData = preview.data as Uint8Array
  for (let i = 0; i < thinned.length; i++) {
    if (!thinned[i]) continue
    previewData[i * 3] = 0
    previewData[i * 3 + 1] = 0
    previewData[i * 3 + 2] = 255
  }

  return { main: skeleton, preview, length_px: length }
}

// ---------------------------------------------------------------------------
// Directional Dilate
// ---------------------------------------------------------------------------
function lineKernel(cv: any, length: number, angleDeg: number): any {
  const size = Math.max(1, length | 1)
  const kern = new Uint8Array(size * size)
  const c = Math.floor(size / 2)
  const a = (angleDeg * Math.PI) / 180
  const dx = Math.cos(a)
  const dy = -Math.sin(a)
  const steps = size * 2
  for (let s = 0; s < steps; s++) {
    const t = -c + (2 * c * s) / (steps - 1)
    const x = Math.round(c + t * dx)
    const y = Math.round(c + t * dy)
    if (x >= 0 && x < size && y >= 0 && y < size) kern[y * size + x] = 1
  }
  return cv.matFromArray(size, size, cv.CV_8U, kern)
}

export const cvDirectionalDilate: NodeImpl = (inputs, params, ctx) => {
  const mask = inputs.mask as any
  if (!mask) return { main: null, preview: null }
  const cv = ctx.cv

  const gray = ctx.track(toGray(cv, mask))
  const binary = ctx.track(new cv.Mat())
  cv.threshold(gray, binary, 0, 255, cv.THRESH_BINARY)

  const length = Math.round(Number(params.length) || 11)
  const iterations = Math.max(1, Math.round(Number(params.iterations) || 1))
  const fixedMode = (Number(params.mode) || 0) === 1

  let out = ctx.track(binary.clone())

  if (fixedMode) {
    const kernel = lineKernel(cv, length, Number(params.angle) || 0)
    const dilated = ctx.track(new cv.Mat())
    cv.dilate(out, dilated, kernel, new cv.Point(-1, -1), iterations)
    kernel.delete()
    out = dilated
  } else {
    const sigma = Number(params.tensor_sigma) || 2.0
    const nBins = Math.max(2, Math.round(Number(params.bins) || 8))

    const f = ctx.track(new cv.Mat())
    binary.convertTo(f, cv.CV_32F, 1 / 255, 0)

    const gx = ctx.track(new cv.Mat())
    const gy = ctx.track(new cv.Mat())
    cv.Sobel(f, gx, cv.CV_32F, 1, 0, 3)
    cv.Sobel(f, gy, cv.CV_32F, 0, 1, 3)

    const gxData = gx.data32F as Float32Array
    const gyData = gy.data32F as Float32Array
    const n = gxData.length
    const gxx = ctx.track(new cv.Mat(gx.rows, gx.cols, cv.CV_32F))
    const gyy = ctx.track(new cv.Mat(gx.rows, gx.cols, cv.CV_32F))
    const gxy = ctx.track(new cv.Mat(gx.rows, gx.cols, cv.CV_32F))
    const gxxData = gxx.data32F as Float32Array
    const gyyData = gyy.data32F as Float32Array
    const gxyData = gxy.data32F as Float32Array
    for (let i = 0; i < n; i++) {
      gxxData[i] = gxData[i] * gxData[i]
      gyyData[i] = gyData[i] * gyData[i]
      gxyData[i] = gxData[i] * gyData[i]
    }

    const blurSize = Math.max(3, Math.round(sigma * 4) | 1)
    const jxx = ctx.track(new cv.Mat())
    const jyy = ctx.track(new cv.Mat())
    const jxy = ctx.track(new cv.Mat())
    cv.GaussianBlur(gxx, jxx, new cv.Size(blurSize, blurSize), sigma)
    cv.GaussianBlur(gyy, jyy, new cv.Size(blurSize, blurSize), sigma)
    cv.GaussianBlur(gxy, jxy, new cv.Size(blurSize, blurSize), sigma)

    const jxxData = jxx.data32F as Float32Array
    const jyyData = jyy.data32F as Float32Array
    const jxyData = jxy.data32F as Float32Array
    const binaryData = binary.data as Uint8Array

    const edges = Array.from({ length: nBins + 1 }, (_, i) => (180 * i) / nBins)
    const kernels = edges.slice(0, nBins).map((lo, i) => lineKernel(cv, length, (lo + edges[i + 1]) / 2))

    const bitmaps: Uint8Array[] = Array.from({ length: nBins }, () => new Uint8Array(n))
    for (let i = 0; i < n; i++) {
      if (binaryData[i] === 0) continue
      const gradAngle = 0.5 * Math.atan2(2 * jxyData[i], jxxData[i] - jyyData[i])
      let tangentDeg = ((gradAngle * 180) / Math.PI + 90) % 180
      if (tangentDeg < 0) tangentDeg += 180
      let bin = Math.floor((tangentDeg / 180) * nBins)
      if (bin >= nBins) bin = nBins - 1
      bitmaps[bin][i] = 255
    }

    const result = ctx.track(binary.clone())
    for (let b = 0; b < nBins; b++) {
      let any = false
      for (let i = 0; i < n; i++) if (bitmaps[b][i]) { any = true; break }
      if (!any) continue
      const layer = ctx.track(new cv.Mat(binary.rows, binary.cols, cv.CV_8U))
      ;(layer.data as Uint8Array).set(bitmaps[b])
      const dilatedLayer = ctx.track(new cv.Mat())
      cv.dilate(layer, dilatedLayer, kernels[b], new cv.Point(-1, -1), iterations)
      cv.bitwise_or(result, dilatedLayer, result)
    }
    kernels.forEach((k) => k.delete())
    out = result
  }

  if (params.then_close !== false) {
    const closeKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3))
    const closed = ctx.track(new cv.Mat())
    cv.morphologyEx(out, closed, cv.MORPH_CLOSE, closeKernel)
    closeKernel.delete()
    out = closed
  }

  const preview = ctx.track(new cv.Mat())
  cv.cvtColor(binary, preview, cv.COLOR_GRAY2BGR)
  const previewData = preview.data as Uint8Array
  const outData = out.data as Uint8Array
  const binData = binary.data as Uint8Array
  for (let i = 0; i < outData.length; i++) {
    if (outData[i] > 0 && binData[i] === 0) {
      previewData[i * 3] = 60
      previewData[i * 3 + 1] = 220
      previewData[i * 3 + 2] = 60
    }
  }

  return { main: out, preview }
}
