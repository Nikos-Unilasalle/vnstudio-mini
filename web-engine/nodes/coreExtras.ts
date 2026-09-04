import type { NodeImpl } from '../types'
import { isMat, toBgr, toGray } from '../cvUtils'

/* ------------------------------------------------------------------ canvas */

/** Pure workspace annotation — no data flows through it. */
export const note: NodeImpl = () => ({})

/** Link organiser: whatever comes in goes straight back out. */
export const reroute: NodeImpl = (inputs) => ({ out: inputs.in ?? null })

/* ---------------------------------------------------------------- geometry */

export const geomFlip: NodeImpl = (inputs, params, ctx) => {
  // Canvas component names the handle `main`, the schema names it `image`.
  const src = (inputs.image ?? inputs.main) as any
  if (!src) return { main: null }
  const dst = ctx.track(new ctx.cv.Mat())
  ctx.cv.flip(src, dst, Math.round(Number(params.flip_mode) ?? 1))
  return { main: dst }
}

export const geomResize: NodeImpl = (inputs, params, ctx) => {
  const src = (inputs.image ?? inputs.main) as any
  if (!src) return { main: null, width: 0, height: 0 }
  const cv = ctx.cv
  const iw = src.cols
  const ih = src.rows

  const mode = Math.round(Number(params.mode) || 0)
  let ow: number
  let oh: number
  if (mode === 1) {
    ow = Math.max(1, Math.trunc(Number(params.width) || 640))
    oh = Math.max(1, Math.trunc((ih * ow) / iw))
  } else if (mode === 2) {
    oh = Math.max(1, Math.trunc(Number(params.height) || 480))
    ow = Math.max(1, Math.trunc((iw * oh) / ih))
  } else if (mode === 3) {
    ow = Math.max(1, Math.trunc(Number(params.width) || 640))
    oh = Math.max(1, Math.trunc(Number(params.height) || 480))
  } else {
    const scale = Number(params.scale) || 0.5
    ow = Math.max(1, Math.trunc(iw * scale))
    oh = Math.max(1, Math.trunc(ih * scale))
  }

  const interpMap = [null, cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_LANCZOS4, cv.INTER_AREA]
  const idx = Math.round(Number(params.interpolation) || 0)
  let interp = idx >= 0 && idx < interpMap.length ? interpMap[idx] : null
  // "Auto" picks the filter that suits the direction: INTER_AREA is the only
  // one that averages when shrinking, INTER_LINEAR is cheap when growing.
  if (interp === null) interp = ow * oh < iw * ih ? cv.INTER_AREA : cv.INTER_LINEAR

  const dst = ctx.track(new cv.Mat())
  cv.resize(src, dst, new cv.Size(ow, oh), 0, 0, interp)
  return { main: dst, width: ow, height: oh }
}

/* -------------------------------------------------------------- color mask */

function hexRgb(raw: unknown): [number, number, number] {
  let hex = String(raw ?? '#FF0000').trim().replace(/^#/, '')
  if (hex.length === 3) hex = hex.split('').map((c) => c + c).join('')
  if (hex.length !== 6 || !/^[0-9a-fA-F]{6}$/.test(hex)) hex = 'FF0000'
  return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)]
}

/** Runs a single BGR triple through OpenCV's own BGR→HSV so the hue matches cv.inRange. */
function bgrToHsvPixel(cv: any, b: number, g: number, r: number): [number, number, number] {
  const one = cv.matFromArray(1, 1, cv.CV_8UC3, [b, g, r])
  const hsv = new cv.Mat()
  cv.cvtColor(one, hsv, cv.COLOR_BGR2HSV)
  const out: [number, number, number] = [hsv.data[0], hsv.data[1], hsv.data[2]]
  one.delete()
  hsv.delete()
  return out
}

export const filterColorMask: NodeImpl = (inputs, params, ctx) => {
  const input = inputs.image as any
  if (!input) return { mask: null, masked: null }
  const cv = ctx.cv
  const image = ctx.track(toBgr(cv, input))
  const [r, g, b] = hexRgb(params.color)

  const mask = ctx.track(new cv.Mat())

  if (Math.round(Number(params.mode) || 0) === 1) {
    // RGB distance: straight Euclidean distance in BGR space, per pixel.
    const thresh = Math.round(Number(params.threshold) ?? 30)
    const src = image.data
    const out = new Uint8Array(image.rows * image.cols)
    for (let i = 0, p = 0; p < out.length; i += 3, p++) {
      const db = src[i] - b
      const dg = src[i + 1] - g
      const dr = src[i + 2] - r
      out[p] = db * db + dg * dg + dr * dr <= thresh * thresh ? 255 : 0
    }
    const built = cv.matFromArray(image.rows, image.cols, cv.CV_8U, Array.from(out))
    built.copyTo(mask)
    built.delete()
  } else {
    const hTol = Math.round(Number(params.h_tol) ?? 10)
    const sTol = Math.round(Number(params.s_tol) ?? 40)
    const vTol = Math.round(Number(params.v_tol) ?? 40)
    const [th, ts, tv] = bgrToHsvPixel(cv, b, g, r)
    const hMin = ((th - hTol) % 180 + 180) % 180
    const hMax = ((th + hTol) % 180 + 180) % 180
    const sMin = Math.max(0, ts - sTol)
    const sMax = Math.min(255, ts + sTol)
    const vMin = Math.max(0, tv - vTol)
    const vMax = Math.min(255, tv + vTol)

    const hsv = new cv.Mat()
    cv.cvtColor(image, hsv, cv.COLOR_BGR2HSV)
    const range = (lo: number[], hi: number[], dst: any) => {
      const low = new cv.Mat(hsv.rows, hsv.cols, cv.CV_8UC3, new cv.Scalar(lo[0], lo[1], lo[2]))
      const high = new cv.Mat(hsv.rows, hsv.cols, cv.CV_8UC3, new cv.Scalar(hi[0], hi[1], hi[2]))
      cv.inRange(hsv, low, high, dst)
      low.delete()
      high.delete()
    }
    if (hMin < hMax) {
      range([hMin, sMin, vMin], [hMax, sMax, vMax], mask)
    } else {
      // Hue wrapped past 179, so the band is two pieces that have to be OR'd.
      const lower = new cv.Mat()
      const upper = new cv.Mat()
      range([hMin, sMin, vMin], [179, sMax, vMax], lower)
      range([0, sMin, vMin], [hMax, sMax, vMax], upper)
      cv.bitwise_or(lower, upper, mask)
      lower.delete()
      upper.delete()
    }
    hsv.delete()
  }

  const masked = ctx.track(new cv.Mat())
  cv.bitwise_and(image, image, masked, mask)
  return { mask, masked }
}

export const filterMorphology: NodeImpl = (inputs, params, ctx) => {
  // The canvas component exposes an image port alongside the mask one.
  const mask = (inputs.mask ?? inputs.image ?? inputs.main) as any
  if (!mask) return { mask: null }
  const cv = ctx.cv
  const size = Math.max(1, Math.round(Number(params.size) || 3))
  const kernel = cv.Mat.ones(size, size, cv.CV_8U)
  const dst = ctx.track(new cv.Mat())
  const anchor = new cv.Point(-1, -1)
  if (Math.round(Number(params.operation) || 0) === 0) cv.dilate(mask, dst, kernel, anchor, 1)
  else cv.erode(mask, dst, kernel, anchor, 1)
  kernel.delete()
  return { mask: dst }
}

/* ------------------------------------------------------------- mask utils */

export const utilCoordToMask: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const ref = inputs.image as any
  const data = inputs.data
  const w = ref ? ref.cols : Math.max(1, Math.round(Number(params.width) || 640))
  const h = ref ? ref.rows : Math.max(1, Math.round(Number(params.height) || 480))
  const mask = ctx.track(cv.Mat.zeros(h, w, cv.CV_8U))

  const items: any[] = Array.isArray(data) ? data : data ? [data] : []
  const white = new cv.Scalar(255)

  const fill = (pts: number[][], hull: boolean) => {
    if (pts.length <= 2) return
    const flat = pts.flatMap((p) => [Math.trunc(p[0]), Math.trunc(p[1])])
    let poly = cv.matFromArray(pts.length, 1, cv.CV_32SC2, flat)
    if (hull) {
      const h2 = new cv.Mat()
      cv.convexHull(poly, h2, false, true)
      poly.delete()
      poly = h2
    }
    const vec = new cv.MatVector()
    vec.push_back(poly)
    cv.fillPoly(mask, vec, white)
    vec.delete()
    poly.delete()
  }

  for (const item of items) {
    if (!item || typeof item !== 'object') continue
    if (Array.isArray(item.landmarks)) {
      fill(item.landmarks.map((lm: any) => [lm.x * w, lm.y * h]), true)
    } else if (Array.isArray(item.pts)) {
      fill(item.pts.map((p: any) => [p[0] * w, p[1] * h]), false)
    } else if (typeof item.xmin === 'number') {
      cv.rectangle(
        mask,
        new cv.Point(Math.trunc(item.xmin * w), Math.trunc(item.ymin * h)),
        new cv.Point(Math.trunc((item.xmin + item.width) * w), Math.trunc((item.ymin + item.height) * h)),
        white,
        -1
      )
    }
  }
  return { mask }
}

export const utilMaskBlend: NodeImpl = (inputs, _params, ctx) => {
  const rawA = (inputs.image_a ?? inputs.image) as any
  if (!rawA) return { main: null }
  const rawB = inputs.image_b as any
  const rawMask = inputs.mask as any
  if (!rawB || !rawMask) return { main: rawA }

  const cv = ctx.cv
  const a = ctx.track(toBgr(cv, rawA))
  const size = new cv.Size(a.cols, a.rows)

  const grayMask = toGray(cv, rawMask)
  const m = new cv.Mat()
  cv.resize(grayMask, m, size, 0, 0, cv.INTER_LINEAR)
  grayMask.delete()

  const bBgr = toBgr(cv, rawB)
  const b = new cv.Mat()
  cv.resize(bBgr, b, size, 0, 0, cv.INTER_LINEAR)
  bBgr.delete()

  const out = ctx.track(new cv.Mat(a.rows, a.cols, cv.CV_8UC3))
  const src = a.data
  const dstB = b.data
  const mm = m.data
  const dst = out.data
  for (let p = 0, i = 0; p < mm.length; p++, i += 3) {
    const alpha = mm[p] / 255
    const inv = 1 - alpha
    dst[i] = src[i] * inv + dstB[i] * alpha
    dst[i + 1] = src[i + 1] * inv + dstB[i + 1] * alpha
    dst[i + 2] = src[i + 2] * inv + dstB[i + 2] * alpha
  }
  m.delete()
  b.delete()
  return { main: out }
}

/* ---------------------------------------------------------- optical flow */

const FLOW_PRESETS = [
  { pyrScale: 0.5, levels: 3, winsize: 15, iterations: 3, polyN: 5, polySigma: 1.2 },
  { pyrScale: 0.5, levels: 5, winsize: 31, iterations: 7, polyN: 7, polySigma: 1.5 },
  { pyrScale: 0.5, levels: 2, winsize: 7, iterations: 3, polyN: 5, polySigma: 1.1 },
  { pyrScale: 0.5, levels: 5, winsize: 25, iterations: 5, polyN: 7, polySigma: 1.5 },
  { pyrScale: 0.5, levels: 2, winsize: 10, iterations: 2, polyN: 5, polySigma: 1.1 },
]

interface FlowState {
  prev: any | null
}

/**
 * Grayscale, forced to 8-bit. Farneback needs CV_8U: handed a float Mat — which
 * is what Canvas, Gray-Scott and the other float32 generators emit — it does not
 * raise, it silently returns an all-zero field that reads as "nothing moved".
 */
function toGray8(cv: any, src: any): any {
  const gray = toGray(cv, src)
  if (gray.depth() === cv.CV_8U) return gray
  const eight = new cv.Mat()
  // Float images in this engine are normalised 0–1.
  gray.convertTo(eight, cv.CV_8U, 255)
  gray.delete()
  return eight
}

export const analysisFlow: NodeImpl = (inputs, params, ctx) => {
  // The canvas component wires this node's input handle as `main` even though the
  // schema names the port `image`, so both spellings have to be accepted.
  const img = (inputs.image ?? inputs.main) as any
  if (!img) return { main: null, data: null }
  const cv = ctx.cv

  let state = ctx.state.get(ctx.nodeId) as FlowState | undefined
  if (!state) {
    state = { prev: null }
    ctx.state.set(ctx.nodeId, state)
  }

  const gray = toGray8(cv, img)
  const presetIdx = Math.round(Number(params.preset) || 0)
  const p =
    presetIdx < FLOW_PRESETS.length
      ? FLOW_PRESETS[presetIdx]
      : {
          pyrScale: Number(params.pyr_scale) || 0.5,
          levels: Math.round(Number(params.levels) || 3),
          winsize: Math.round(Number(params.winsize) || 15),
          iterations: Math.round(Number(params.iterations) || 3),
          polyN: Math.round(Number(params.poly_n) || 5),
          polySigma: Number(params.poly_sigma) || 1.2,
        }

  let flow: any = null
  if (state.prev && state.prev.rows === gray.rows && state.prev.cols === gray.cols) {
    flow = ctx.track(new cv.Mat())
    cv.calcOpticalFlowFarneback(state.prev, gray, flow, p.pyrScale, p.levels, p.winsize, p.iterations, p.polyN, p.polySigma, 0)
  }

  // The previous frame has to outlive the run, so it is owned here rather than tracked.
  if (state.prev) state.prev.delete()
  state.prev = gray

  return { main: img, data: flow }
}

/** True for the CV_32FC2 field `analysis_flow` hands downstream. */
function isFlowMat(v: unknown): boolean {
  return isMat(v) && (v as any).channels() === 2
}

export const analysisFlowViz: NodeImpl = (inputs, _params, ctx) => {
  const flow = inputs.data as any
  if (!isFlowMat(flow)) return { main: null }
  const cv = ctx.cv

  const planes = new cv.MatVector()
  cv.split(flow, planes)
  const fx = planes.get(0)
  const fy = planes.get(1)
  const mag = new cv.Mat()
  const ang = new cv.Mat()
  cv.cartToPolar(fx, fy, mag, ang)

  const magData = mag.data32F
  const angData = ang.data32F
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < magData.length; i++) {
    if (magData[i] < lo) lo = magData[i]
    if (magData[i] > hi) hi = magData[i]
  }
  const span = hi - lo || 1

  const hsvArr = new Array<number>(magData.length * 3)
  for (let i = 0, j = 0; i < magData.length; i++, j += 3) {
    hsvArr[j] = Math.round(((angData[i] * 180) / Math.PI) / 2)
    hsvArr[j + 1] = 255
    hsvArr[j + 2] = Math.round(((magData[i] - lo) / span) * 255)
  }

  const hsv = cv.matFromArray(flow.rows, flow.cols, cv.CV_8UC3, hsvArr)
  const out = ctx.track(new cv.Mat())
  cv.cvtColor(hsv, out, cv.COLOR_HSV2BGR)

  hsv.delete()
  mag.delete()
  ang.delete()
  fx.delete()
  fy.delete()
  planes.delete()
  return { main: out }
}

/* ------------------------------------------------------------- monitor */

function meanOfChannel(cv: any, mat: any, channel: number): number {
  const mean = cv.mean(mat)
  return mean[channel] ?? 0
}

export const analysisMonitor: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const data = inputs.data
  const img = (inputs.image ?? inputs.main) as any
  const mask = inputs.mask as any

  let mode = Math.round(Number(params.mode) || 0)
  const scale = Number(params.scale) ?? 1
  const offset = Number(params.offset) || 0
  const precision = Math.max(0, Math.min(5, Math.round(Number(params.precision) ?? 3)))

  // "Auto" resolves against whatever is actually plugged in.
  if (mode === 0) {
    if (data != null) {
      if (isFlowMat(data)) mode = 1
      else if (isMat(data)) mode = 3
      else if (Array.isArray(data)) mode = 7
      else if (typeof data === 'number') mode = 8
      else mode = 3
    } else if (mask) mode = 2
    else if (img) mode = 3
    else mode = 3
  }

  let val = 0
  let unit = ''

  if (mode === 1 && isFlowMat(data)) {
    const flow = data as any
    const planes = new cv.MatVector()
    cv.split(flow, planes)
    const fx = planes.get(0)
    const fy = planes.get(1)
    const mag = new cv.Mat()
    const ang = new cv.Mat()
    cv.cartToPolar(fx, fy, mag, ang)

    let maskRes: any = null
    if (mask) {
      const g = toGray(cv, mask)
      maskRes = new cv.Mat()
      cv.resize(g, maskRes, new cv.Size(flow.cols, flow.rows), 0, 0, cv.INTER_NEAREST)
      g.delete()
    }
    const magData = mag.data32F
    let sum = 0
    let count = 0
    for (let i = 0; i < magData.length; i++) {
      if (maskRes && maskRes.data[i] === 0) continue
      sum += magData[i]
      count++
    }
    val = count > 0 ? sum / count : 0
    unit = 'flux'
    maskRes?.delete()
    mag.delete()
    ang.delete()
    fx.delete()
    fy.delete()
    planes.delete()
  } else if (mode === 2) {
    const m = mask ?? (isMat(data) ? data : null)
    if (m) {
      const g = toGray(cv, m)
      val = cv.countNonZero(g)
      g.delete()
      unit = 'px'
    }
  } else if (mode >= 3 && mode <= 6) {
    const im = img ?? (isMat(data) ? data : null)
    if (im) {
      if (im.channels() === 1) val = meanOfChannel(cv, im, 0)
      else if (mode === 3) {
        const m = cv.mean(im)
        val = (m[0] + m[1] + m[2]) / 3
      } else if (mode === 4) val = meanOfChannel(cv, im, 2)
      else if (mode === 5) val = meanOfChannel(cv, im, 1)
      else val = meanOfChannel(cv, im, 0)
      unit = 'lvl'
    }
  } else if (mode === 7) {
    if (Array.isArray(data)) val = data.length
    else if (data && typeof data === 'object') val = 1
    else if (typeof data === 'number') val = data
    unit = 'items'
  } else if (mode === 8) {
    if (typeof data === 'number') val = data
    else if (typeof data === 'string') val = Number.parseFloat(data) || 0
  }

  const finalVal = val * scale + offset
  const txt = `${finalVal.toFixed(precision)} ${unit}`.trim()

  const source = img ?? (mask ?? null)
  let out: any = null
  if (source) {
    out = ctx.track(toBgr(cv, source))
    cv.putText(out, txt, new cv.Point(20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, new cv.Scalar(0, 255, 0, 255), 2)
  }

  ctx.emit('value', finalVal)
  return { main: out, scalar: finalVal, data_out: finalVal, display_text: txt }
}
