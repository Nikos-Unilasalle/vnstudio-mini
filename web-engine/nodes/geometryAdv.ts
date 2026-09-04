import type { NodeImpl } from '../types'
import { toBgr } from '../cvUtils'

/* -------------------------------------------------------------- primitives */

type Point = [number, number]

/** Pulls an (x, y) out of any of the point shapes that flow through the graph. */
function pointOf(value: unknown): Point | null {
  if (Array.isArray(value) && value.length >= 2) return [Number(value[0]), Number(value[1])]
  if (!value || typeof value !== 'object') return null
  const p = value as Record<string, any>
  if (typeof p.x === 'number' && typeof p.y === 'number') return [p.x, p.y]
  if (Array.isArray(p.pts) && p.pts.length) {
    const first = p.pts[0]
    if (Array.isArray(first) && first.length >= 2) return [Number(first[0]), Number(first[1])]
  }
  if (p.center && typeof p.center === 'object') return [Number(p.center.x) || 0, Number(p.center.y) || 0]
  return null
}

function contourPoints(value: unknown): Point[] {
  const dict = value as { pts?: unknown[] } | null | undefined
  if (!dict || !Array.isArray(dict.pts)) return []
  return dict.pts.map(pointOf).filter((p): p is Point => p !== null)
}

/** Points → a CV_32FC2 Mat, the shape every contour routine here expects. */
function toContourMat(cv: any, points: Point[]): any {
  return cv.matFromArray(points.length, 1, cv.CV_32FC2, points.flat())
}

/**
 * The four corners of a rotated rectangle.
 *
 * `cv.boxPoints` is not in this OpenCV build, so this is the same computation
 * OpenCV performs in `RotatedRect::points`, corner order included — downstream
 * drawing depends on that order.
 */
export function boxPoints(rect: { center: { x: number; y: number }; size: { width: number; height: number }; angle: number }): Point[] {
  const radians = (rect.angle * Math.PI) / 180
  const b = Math.cos(radians) * 0.5
  const a = Math.sin(radians) * 0.5
  const { x: cx, y: cy } = rect.center
  const { width, height } = rect.size

  const p0: Point = [cx - a * height - b * width, cy + b * height - a * width]
  const p1: Point = [cx + a * height - b * width, cy - b * height - a * width]
  return [p0, p1, [2 * cx - p0[0], 2 * cy - p0[1]], [2 * cx - p1[0], 2 * cy - p1[1]]]
}

/* ------------------------------------------------------------ contour shape */

export const geomApproxPoly: NodeImpl = (inputs, params, ctx) => {
  const source = inputs.contour
  const points = contourPoints(source)
  if (points.length < 2) return { approx_contour: null }

  const cv = ctx.cv
  const mat = toContourMat(cv, points)
  const closed = params.closed !== false
  // Epsilon is a fraction of the perimeter, so the parameter means the same
  // thing whatever the contour's size.
  const epsilon = ((Number(params.epsilon_pct) ?? 2) / 100) * cv.arcLength(mat, true)

  const approx = new cv.Mat()
  cv.approxPolyDP(mat, approx, epsilon, closed)
  const data = approx.data32F
  const out: Point[] = []
  for (let i = 0; i < approx.rows; i++) out.push([data[i * 2], data[i * 2 + 1]])
  mat.delete()
  approx.delete()

  return { approx_contour: { ...(source as object), pts: out } }
}

export const geomFitShape: NodeImpl = (inputs, params, ctx) => {
  const points = contourPoints(inputs.contour)
  if (points.length < 2) return { bbox: null, min_rect: null }
  const cv = ctx.cv

  const pad = Number(params.padding) || 0
  const xs = points.map((p) => p[0])
  const ys = points.map((p) => p[1])
  const x = Math.max(0, Math.min(...xs) - pad)
  const y = Math.max(0, Math.min(...ys) - pad)
  const width = Math.min(1 - x, Math.max(...xs) - Math.min(...xs) + 2 * pad)
  const height = Math.min(1 - y, Math.max(...ys) - Math.min(...ys) + 2 * pad)

  const bbox = {
    xmin: x,
    ymin: y,
    width,
    height,
    label: 'bbox',
    _type: 'graphics',
    shape: 'rect',
    pts: [
      [x, y],
      [x + width, y + height],
    ],
    color: '#ffffff',
    relative: true,
  }

  // minAreaRect works in integers, so normalised coordinates are scaled up into
  // a large pixel space first and the corners scaled back afterwards.
  const SCALE = 10000
  const scaled = toContourMat(cv, points.map(([px, py]) => [px * SCALE, py * SCALE] as Point))
  const rect = cv.minAreaRect(scaled)
  scaled.delete()

  const minRect = {
    label: 'min_rect',
    _type: 'graphics',
    shape: 'polygon',
    pts: boxPoints(rect).map(([px, py]) => [px / SCALE, py / SCALE]),
    color: '#ff00ff',
    relative: true,
  }

  return { bbox, min_rect: minRect }
}

/* ---------------------------------------------------------------- warping */

/** Accepts a 2×3 matrix as nested rows or as six flat numbers. */
function affineValues(value: unknown): number[] | null {
  if (Array.isArray(value) && value.length === 6 && value.every((v) => typeof v === 'number')) return value as number[]
  if (Array.isArray(value) && value.length >= 2 && Array.isArray(value[0])) {
    const flat = (value as number[][]).slice(0, 2).flat()
    return flat.length === 6 ? flat : null
  }
  return null
}

export const geomWarpAffine: NodeImpl = (inputs, _params, ctx) => {
  const img = (inputs.image ?? inputs.main) as any
  if (!img) return { main: null }
  const values = affineValues(inputs.matrix)
  if (!values) return { main: img }

  const cv = ctx.cv
  const matrix = cv.matFromArray(2, 3, cv.CV_64F, values)
  const out = ctx.track(new cv.Mat())
  cv.warpAffine(img, out, matrix, new cv.Size(img.cols, img.rows))
  matrix.delete()
  return { main: out }
}

export const geomRotateNoCrop: NodeImpl = (inputs, params, ctx) => {
  const img = (inputs.image ?? inputs.main) as any
  if (!img) return { main: null }
  const angle = Number(params.angle) || 0
  if (angle === 0) return { main: img }

  const cv = ctx.cv
  const w = img.cols
  const h = img.rows
  const cx = Math.floor(w / 2)
  const cy = Math.floor(h / 2)

  const matrix = cv.getRotationMatrix2D(new cv.Point(cx, cy), angle, 1)
  const m = matrix.data64F
  const cos = Math.abs(m[0])
  const sin = Math.abs(m[1])

  // The rotated image's bounding box, so no corner is cropped away.
  const newW = Math.trunc(h * sin + w * cos)
  const newH = Math.trunc(h * cos + w * sin)
  m[2] += newW / 2 - cx
  m[5] += newH / 2 - cy

  const out = ctx.track(new cv.Mat())
  cv.warpAffine(img, out, matrix, new cv.Size(newW, newH))
  matrix.delete()
  return { main: out }
}

export const geomPerspective: NodeImpl = (inputs, params, ctx) => {
  const img = (inputs.image ?? inputs.main) as any
  const src = inputs.src_pts as unknown[] | undefined
  if (!img) return { main: null }
  if (!Array.isArray(src) || src.length < 4) return { main: img }

  const cv = ctx.cv
  const w = img.cols
  const h = img.rows
  const outW = Math.max(1, Math.round(Number(params.width) || 800))
  const outH = Math.max(1, Math.round(Number(params.height) || 600))

  const corners: number[] = []
  for (const entry of src.slice(0, 4)) {
    const p = pointOf(entry) ?? [0, 0]
    corners.push(p[0] * w, p[1] * h)
  }

  // Source corners are taken in the order given — top-left, top-right,
  // bottom-right, bottom-left — and mapped onto the output rectangle.
  const srcMat = cv.matFromArray(4, 1, cv.CV_32FC2, corners)
  const dstMat = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, outW, 0, outW, outH, 0, outH])
  const matrix = cv.getPerspectiveTransform(srcMat, dstMat)

  const out = ctx.track(new cv.Mat())
  cv.warpPerspective(img, out, matrix, new cv.Size(outW, outH))
  srcMat.delete()
  dstMat.delete()
  matrix.delete()
  return { main: out }
}

export const utilManualPoints: NodeImpl = (inputs, params) => {
  const points: { x: number; y: number }[] = []
  for (let i = 1; i <= 4; i++) {
    // A wired point wins over the typed parameter.
    const wired = pointOf(inputs[`p${i}`])
    const x = wired ? wired[0] : Number(params[`x${i}`] ?? 0.1)
    const y = wired ? wired[1] : Number(params[`y${i}`] ?? 0.1)
    points.push({ x, y })
  }
  return { points }
}

/* ------------------------------------------------------------------ boxes */

interface BoxDict {
  xmin?: number
  ymin?: number
  width?: number
  height?: number
  score?: number
  pts?: number[][]
  id?: number
  [key: string]: unknown
}

function boxArea(box: BoxDict): number {
  return (Number(box.width) || 0) * (Number(box.height) || 0)
}

export const utilCoordCenter: NodeImpl = (inputs) => {
  const d = inputs.data as BoxDict | null | undefined
  if (!d || typeof d !== 'object' || typeof d.xmin !== 'number') {
    return { cx: 0, cy: 0, dict: { xmin: 0, ymin: 0, width: 0, height: 0 } }
  }
  const cx = d.xmin + (Number(d.width) || 0) / 2
  const cy = (Number(d.ymin) || 0) + (Number(d.height) || 0) / 2
  // The centre travels on as a zero-size box, so it plugs into anything that
  // consumes a box.
  return { cx, cy, dict: { xmin: cx, ymin: cy, width: 0, height: 0 } }
}

export const boxSelection: NodeImpl = (inputs, params, ctx) => {
  const raw = inputs.boxes_list
  const boxes: BoxDict[] = Array.isArray(raw) ? raw.filter((b): b is BoxDict => !!b && typeof b === 'object') : []
  const image = inputs.image as any
  if (!boxes.length) return { box: null, main: image ?? null, index: -1 }

  const mode = Math.round(Number(params.mode) || 0)
  let selected = 0
  if (mode === 1) boxes.forEach((b, i) => { if (boxArea(b) > boxArea(boxes[selected])) selected = i })
  else if (mode === 2) boxes.forEach((b, i) => { if (boxArea(b) < boxArea(boxes[selected])) selected = i })
  else if (mode === 3) boxes.forEach((b, i) => { if ((Number(b.score) || 0) > (Number(boxes[selected].score) || 0)) selected = i })
  else selected = Math.max(0, Math.min(Math.round(Number(params.index) || 0), boxes.length - 1))

  const box = boxes[selected]
  let overlay: any = null
  if (image) {
    const cv = ctx.cv
    overlay = ctx.track(toBgr(cv, image))
    const w = overlay.cols
    const h = overlay.rows
    const x = Number(box.xmin) || 0
    const y = Number(box.ymin) || 0
    cv.rectangle(
      overlay,
      new cv.Point(Math.trunc(x * w), Math.trunc(y * h)),
      new cv.Point(Math.trunc((x + (Number(box.width) || 0)) * w), Math.trunc((y + (Number(box.height) || 0)) * h)),
      new cv.Scalar(0, 255, 255, 255),
      Math.max(1, Math.round(Number(params.thickness) || 3))
    )
  }

  return { box, main: overlay, index: selected }
}

export const bboxTransform: NodeImpl = (inputs, params, ctx) => {
  const collected: BoxDict[] = []
  if (Array.isArray(inputs.boxes_list)) {
    collected.push(...(inputs.boxes_list as unknown[]).filter((b): b is BoxDict => !!b && typeof b === 'object'))
  }
  if (inputs.box && typeof inputs.box === 'object') collected.push(inputs.box as BoxDict)

  const image = inputs.image as any
  const w = image ? image.cols : 0
  const h = image ? image.rows : 0

  const resizeMode = Math.round(Number(params.resize_mode) || 0)
  const amount = Number(params.resize_amount) || 0
  const doClamp = params.clamp !== false
  const filterMode = Math.round(Number(params.filter_mode) || 0)
  const minSize = Number(params.min_size) || 0
  const maxSize = Number(params.max_size) || 0

  const out: BoxDict[] = []
  for (const original of collected) {
    let box: BoxDict = { ...original }

    if (amount !== 0) {
      const x = Number(box.xmin) || 0
      const y = Number(box.ymin) || 0
      const bw = Number(box.width) || 0
      const bh = Number(box.height) || 0
      const cx = x + bw / 2
      const cy = y + bh / 2
      let nw: number
      let nh: number
      if (resizeMode === 0) {
        // Percent of the box's own size, so every box grows proportionally.
        const factor = Math.max(0, 1 + amount / 100)
        nw = bw * factor
        nh = bh * factor
      } else {
        // A fixed pixel margin per side, which needs the image dimensions.
        const dx = w ? amount / w : 0
        const dy = h ? amount / h : 0
        nw = Math.max(0, bw + 2 * dx)
        nh = Math.max(0, bh + 2 * dy)
      }
      box = { ...box, xmin: cx - nw / 2, ymin: cy - nh / 2, width: nw, height: nh }
    }

    if (doClamp) {
      const x = Math.max(0, Number(box.xmin) || 0)
      const y = Math.max(0, Number(box.ymin) || 0)
      const x2 = Math.min(1, (Number(box.xmin) || 0) + (Number(box.width) || 0))
      const y2 = Math.min(1, (Number(box.ymin) || 0) + (Number(box.height) || 0))
      box = { ...box, xmin: x, ymin: y, width: Math.max(0, x2 - x), height: Math.max(0, y2 - y) }
    }

    // Keep the drawable corners consistent with the numbers above.
    if (Array.isArray(box.pts)) {
      const x = Number(box.xmin) || 0
      const y = Number(box.ymin) || 0
      box = { ...box, pts: [[x, y], [x + (Number(box.width) || 0), y + (Number(box.height) || 0)]] }
    }

    const bw = Number(box.width) || 0
    const bh = Number(box.height) || 0
    let metrics: number[] | null = null
    if (filterMode === 1) metrics = [bw * bh * 100]
    else if (filterMode === 2) metrics = w && h ? [bw * w * bh * h] : null
    else if (filterMode === 3) metrics = w && h ? [bw * w, bh * h] : null
    if (metrics && metrics.some((m) => m < minSize || (maxSize > 0 && m > maxSize))) continue

    out.push(box)
  }

  // Re-index so anything downstream keying on id stays contiguous.
  out.forEach((box, i) => {
    box.id = i
  })

  let overlay: any = null
  if (image) {
    const cv = ctx.cv
    overlay = ctx.track(toBgr(cv, image))
    if (params.draw !== false) {
      out.forEach((box, i) => {
        const x = Number(box.xmin) || 0
        const y = Number(box.ymin) || 0
        const colour = new cv.Scalar(((i * 67 + 40) % 200) + 55, ((i * 137 + 80) % 200) + 55, ((i * 197 + 120) % 200) + 55, 255)
        cv.rectangle(
          overlay,
          new cv.Point(Math.trunc(x * w), Math.trunc(y * h)),
          new cv.Point(Math.trunc((x + (Number(box.width) || 0)) * w), Math.trunc((y + (Number(box.height) || 0)) * h)),
          colour,
          2
        )
      })
    }
  }

  return { boxes_list: out, main: overlay, count: out.length }
}
