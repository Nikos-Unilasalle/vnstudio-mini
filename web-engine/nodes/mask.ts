import type { NodeImpl } from '../types'
import { toGray } from '../cvUtils'

export const maskCircle: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const img = inputs.image as any

  const w = img ? img.cols : Math.max(1, Math.round(Number(params.img_w) || 512))
  const h = img ? img.rows : Math.max(1, Math.round(Number(params.img_h) || 512))

  const cx = ((Number(params.center_x) ?? 50) / 100) * w
  const cy = ((Number(params.center_y) ?? 50) / 100) * h
  const ref = Math.min(w, h)
  const rx = Math.max(1, ((Number(params.radius_x) ?? 45) / 100) * ref)
  const ry = Math.max(1, ((Number(params.radius_y) ?? 45) / 100) * ref)
  const feather = Math.max(0, Math.round(Number(params.feather) || 0))
  const invert = !!params.invert

  let mask = ctx.track(cv.Mat.zeros(h, w, cv.CV_8U))
  cv.ellipse(mask, new cv.Point(Math.round(cx), Math.round(cy)), new cv.Size(Math.round(rx), Math.round(ry)), 0, 0, 360, new cv.Scalar(255), -1)

  if (feather > 0) {
    const dist = ctx.track(new cv.Mat())
    cv.distanceTransform(mask, dist, cv.DIST_L2, 3)
    const scaled = ctx.track(new cv.Mat())
    dist.convertTo(scaled, cv.CV_8U, 255.0 / feather, 0)
    mask = scaled
  }

  if (invert) {
    const inverted = ctx.track(new cv.Mat())
    cv.bitwise_not(mask, inverted)
    mask = inverted
  }

  const base = img ? ctx.track(toBgrLocal(cv, img)) : ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(30, 30, 30)))
  const masked = ctx.track(new cv.Mat())
  base.copyTo(masked, mask)

  return { mask, masked }
}

function toBgrLocal(cv: any, src: any): any {
  const out = new cv.Mat()
  if (src.channels() === 1) cv.cvtColor(src, out, cv.COLOR_GRAY2BGR)
  else if (src.channels() === 4) cv.cvtColor(src, out, cv.COLOR_BGRA2BGR)
  else src.copyTo(out)
  return out
}

export const maskPointQuery: NodeImpl = (inputs) => {
  const mask = inputs.mask as any
  const x = Number(inputs.x) || 0
  const y = Number(inputs.y) || 0
  if (!mask) return { inside: false }

  const w = mask.cols
  const h = mask.rows
  let ix: number, iy: number
  if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
    ix = Math.round(x * (w - 1))
    iy = Math.round(y * (h - 1))
  } else {
    ix = Math.round(x)
    iy = Math.round(y)
  }
  if (ix < 0 || ix >= w || iy < 0 || iy >= h) return { inside: false }

  const channels = mask.channels()
  const value = mask.ucharPtr(iy, ix)[0]
  void channels
  return { inside: value !== 0 }
}

export const filterFloatThreshold: NodeImpl = (inputs, params, ctx) => {
  const raw = inputs.raw as any
  if (!raw) return { mask: null, count: 0 }
  const cv = ctx.cv

  let data: Float32Array
  let width: number
  let height: number
  if (raw && typeof raw === 'object' && 'bands' in raw) {
    const band = (raw as any).bands[0]
    data = band.data ?? band
    width = band.width
    height = band.height
  } else if (raw && typeof raw === 'object' && typeof raw.cols === 'number') {
    // A live cv.Mat (e.g. a single-channel float band).
    width = raw.cols
    height = raw.rows
    data = raw.data32F as Float32Array
  } else {
    return { mask: null, count: 0 }
  }
  if (!data || !width || !height) return { mask: null, count: 0 }

  const low = Number(params.low) ?? -1.0
  const high = Number(params.high) ?? 0.0
  const invert = !!params.invert

  const mask = ctx.track(new cv.Mat(height, width, cv.CV_8U))
  const maskData = mask.data as Uint8Array
  let count = 0
  for (let i = 0; i < maskData.length; i++) {
    const inRange = data[i] >= low && data[i] <= high
    const value = invert ? !inRange : inRange
    maskData[i] = value ? 255 : 0
    if (value) count++
  }

  return { mask, count }
}

export const maskToImage: NodeImpl = (inputs, _params, ctx) => {
  const mask = inputs.mask as any
  if (!mask) return { main: null }
  const cv = ctx.cv
  if (mask.channels() === 1) {
    return { main: ctx.track(toBgrLocal(cv, mask)) }
  }
  return { main: mask }
}

export const fillHoles: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.mask as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const binary = ctx.track(toGray(cv, src))
  const method = Number(params.method) || 0

  if (method === 0) {
    const contours = new cv.MatVector()
    const hierarchy = ctx.track(new cv.Mat())
    cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    const filled = ctx.track(cv.Mat.zeros(binary.rows, binary.cols, cv.CV_8U))
    cv.drawContours(filled, contours, -1, new cv.Scalar(255), -1)
    contours.delete()
    return { main: filled }
  }

  // Flooding the border finds everything reachable from outside; the inverse is the holes.
  const padded = ctx.track(new cv.Mat())
  cv.copyMakeBorder(binary, padded, 1, 1, 1, 1, cv.BORDER_CONSTANT, new cv.Scalar(0))
  const flooded = ctx.track(padded.clone())
  const ffMask = ctx.track(cv.Mat.zeros(padded.rows + 2, padded.cols + 2, cv.CV_8U))
  cv.floodFill(flooded, ffMask, new cv.Point(0, 0), new cv.Scalar(255))

  const inverted = ctx.track(new cv.Mat())
  cv.bitwise_not(flooded, inverted)
  const holes = ctx.track(inverted.roi(new cv.Rect(1, 1, binary.cols, binary.rows)).clone())

  if (method === 1) {
    const filled = ctx.track(new cv.Mat())
    cv.bitwise_or(binary, holes, filled)
    return { main: filled }
  }

  const maxHole = Number(params.max_hole_px) || 500
  const holeContours = new cv.MatVector()
  const holeHierarchy = ctx.track(new cv.Mat())
  cv.findContours(holes, holeContours, holeHierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  const smallHoles = ctx.track(cv.Mat.zeros(binary.rows, binary.cols, cv.CV_8U))
  for (let i = 0; i < holeContours.size(); i++) {
    const contour = holeContours.get(i)
    if (cv.contourArea(contour) > maxHole) continue
    const single = new cv.MatVector()
    single.push_back(contour)
    cv.drawContours(smallHoles, single, -1, new cv.Scalar(255), -1)
    single.delete()
  }
  holeContours.delete()

  const filled = ctx.track(new cv.Mat())
  cv.bitwise_or(binary, smallHoles, filled)
  return { main: filled }
}

export const maskOperations: NodeImpl = (inputs, params, ctx) => {
  const a = inputs.mask_a as any
  const b = inputs.mask_b as any
  if (!a && !b) return { mask: null }
  const cv = ctx.cv
  const reference = a ?? b

  const first = a ? ctx.track(toGray(cv, a)) : ctx.track(cv.Mat.zeros(reference.rows, reference.cols, cv.CV_8U))
  let second = b ? ctx.track(toGray(cv, b)) : ctx.track(cv.Mat.zeros(reference.rows, reference.cols, cv.CV_8U))

  if (second.rows !== first.rows || second.cols !== first.cols) {
    const resized = ctx.track(new cv.Mat())
    cv.resize(second, resized, new cv.Size(first.cols, first.rows), 0, 0, cv.INTER_NEAREST)
    second = resized
  }

  const result = ctx.track(new cv.Mat())
  const operation = Number(params.operation) || 0
  if (operation === 0) {
    cv.bitwise_or(first, second, result)
  } else if (operation === 1) {
    const negated = ctx.track(new cv.Mat())
    cv.bitwise_not(second, negated)
    cv.bitwise_and(first, negated, result)
  } else {
    cv.bitwise_and(first, second, result)
  }
  return { mask: result }
}
