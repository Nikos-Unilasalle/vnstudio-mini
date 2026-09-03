import type { NodeImpl } from '../types'
import { drawPolyline, parseColor, toBgr, toGray } from '../cvUtils'

interface NormalisedPoint {
  x: number
  y: number
}

function parsePoints(raw: unknown): NormalisedPoint[] {
  try {
    const parsed = JSON.parse(String(raw ?? '[]'))
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export const utilRoiPolygon: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  const maskIn = inputs.mask_in as any
  if (!src) return { main: null, mask: null, masked: null, masked_inv: null, pts: [] }
  const cv = ctx.cv
  const w = src.cols
  const h = src.rows

  const points = parsePoints(params.points)
  // No polygon drawn yet means "let everything through", matching the desktop node.
  const mask = ctx.track(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(points.length === 0 ? 255 : 0)))

  if (points.length >= 2) {
    const pixels = points.map((p) => ({ x: p.x * w, y: p.y * h }))
    if (points.length >= 3 && params.filled !== false) {
      const flat = pixels.flatMap((p) => [Math.round(p.x), Math.round(p.y)])
      const pointsMat = cv.matFromArray(points.length, 1, cv.CV_32SC2, flat)
      const vector = new cv.MatVector()
      vector.push_back(pointsMat)
      cv.fillPoly(mask, vector, new cv.Scalar(255))
      pointsMat.delete()
      vector.delete()
    } else {
      drawPolyline(cv, mask, pixels, points.length >= 3, new cv.Scalar(255), Math.max(1, Number(params.thickness) || 2))
    }
  }

  if (maskIn) {
    const incoming = ctx.track(toGray(cv, maskIn))
    cv.bitwise_and(mask, incoming, mask)
  }

  const masked = ctx.track(new cv.Mat())
  src.copyTo(masked, mask)

  const inverse = ctx.track(new cv.Mat())
  cv.bitwise_not(mask, inverse)
  const maskedInverse = ctx.track(new cv.Mat())
  src.copyTo(maskedInverse, inverse)

  const overlay = ctx.track(toBgr(cv, src))
  if (points.length >= 2) {
    const pixels = points.map((p) => ({ x: p.x * w, y: p.y * h }))
    drawPolyline(cv, overlay, pixels, true, new cv.Scalar(0, 255, 136, 255), 2)
  }

  return { main: overlay, mask, masked, masked_inv: maskedInverse, pts: points }
}

/** cv.RotatedRect has no points() helper in OpenCV.js, so derive the corners here. */
function rotatedRectPoints(rect: {
  center: { x: number; y: number }
  size: { width: number; height: number }
  angle: number
}): NormalisedPoint[] {
  const { center, size, angle } = rect
  const radians = (angle * Math.PI) / 180
  const b = Math.cos(radians) * 0.5
  const a = Math.sin(radians) * 0.5
  const p0 = { x: center.x - a * size.height - b * size.width, y: center.y + b * size.height - a * size.width }
  const p1 = { x: center.x + a * size.height - b * size.width, y: center.y - b * size.height - a * size.width }
  return [p0, p1, { x: 2 * center.x - p0.x, y: 2 * center.y - p0.y }, { x: 2 * center.x - p1.x, y: 2 * center.y - p1.y }]
}

function warpCrop(
  cv: any,
  image: any,
  cx: number,
  cy: number,
  angle: number,
  width: number,
  height: number,
  pad: number,
  interpolation: number
): any {
  const padded = new cv.Mat()
  cv.copyMakeBorder(image, padded, pad, pad, pad, pad, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0))
  const rotation = cv.getRotationMatrix2D(new cv.Point(cx, cy), angle, 1)
  const warped = new cv.Mat()
  cv.warpAffine(padded, warped, rotation, new cv.Size(padded.cols, padded.rows), interpolation, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0))
  padded.delete()
  rotation.delete()

  const x0 = Math.max(0, Math.round(cx - width / 2))
  const y0 = Math.max(0, Math.round(cy - height / 2))
  const x1 = Math.min(warped.cols, Math.round(cx + width / 2))
  const y1 = Math.min(warped.rows, Math.round(cy + height / 2))
  if (x1 <= x0 || y1 <= y0) return warped

  const cropped = warped.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0)).clone()
  warped.delete()
  return cropped
}

export const pluginRotate: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const angle = Number(params.angle) || 0
  const scale = Number(params.scale) || 1.0
  const w = src.cols
  const h = src.rows
  const center = new cv.Point(Math.floor(w / 2), Math.floor(h / 2))
  const matrix = cv.getRotationMatrix2D(center, angle, scale)
  const out = ctx.track(new cv.Mat())
  cv.warpAffine(src, out, matrix, new cv.Size(w, h))
  matrix.delete()
  return { main: out }
}

export const pluginOffset: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const tx = Number(params.x_offset) || 0
  const ty = Number(params.y_offset) || 0
  const matrix = cv.matFromArray(2, 3, cv.CV_32F, [1, 0, tx, 0, 1, ty])
  const out = ctx.track(new cv.Mat())
  cv.warpAffine(src, out, matrix, new cv.Size(src.cols, src.rows))
  matrix.delete()
  return { main: out }
}

function parseRect(raw: unknown): { x: number; y: number; w: number; h: number } {
  try {
    const parsed = JSON.parse(String(raw ?? '{}'))
    return {
      x: Number(parsed.x) || 0.1,
      y: Number(parsed.y) || 0.1,
      w: Number(parsed.w) || 0.8,
      h: Number(parsed.h) || 0.8,
    }
  } catch {
    return { x: 0.1, y: 0.1, w: 0.8, h: 0.8 }
  }
}

export const geomCropRect: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, mask: null, width: 0, height: 0, box: null }
  const cv = ctx.cv
  const w = src.cols
  const h = src.rows
  const rect = parseRect(params.rect)

  const x1 = Math.max(0, Math.round(rect.x * w))
  const y1 = Math.max(0, Math.round(rect.y * h))
  const x2 = Math.min(w, Math.round((rect.x + rect.w) * w))
  const y2 = Math.min(h, Math.round((rect.y + rect.h) * h))

  const mask = ctx.track(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0)))
  if (x2 <= x1 || y2 <= y1) {
    mask.setTo(new cv.Scalar(255))
    return { main: src, mask, width: w, height: h, box: null }
  }

  const roi = mask.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1))
  roi.setTo(new cv.Scalar(255))
  roi.delete()

  const cropped = ctx.track(src.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1)).clone())

  return {
    main: cropped,
    mask,
    width: cropped.cols,
    height: cropped.rows,
    box: { xmin: rect.x, ymin: rect.y, width: rect.w, height: rect.h },
  }
}

export const geomCropper: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  const rawData = inputs.data
  if (!src || !rawData) return { main: src ?? null }
  const cv = ctx.cv

  const data = (Array.isArray(rawData) ? rawData[0] : rawData) as Record<string, number> | undefined
  if (!data || typeof data.xmin !== 'number') return { main: src }

  const w = src.cols
  const h = src.rows
  const pad = (Number(params.padding) || 10) / 100

  const x1 = Math.max(0, Math.round((data.xmin - pad / 2) * w))
  const y1 = Math.max(0, Math.round((data.ymin - pad / 2) * h))
  const x2 = Math.min(w, Math.round((data.xmin + (data.width ?? 0) + pad / 2) * w))
  const y2 = Math.min(h, Math.round((data.ymin + (data.height ?? 0) + pad / 2) * h))

  if (x2 > x1 && y2 > y1) {
    return { main: ctx.track(src.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1)).clone()) }
  }
  return { main: src }
}

function fitFovCircle(cv: any, ctx: any, gray: any, darkThresh: number, margin: number): { cx: number; cy: number; r: number } {
  const bright = ctx.track(new cv.Mat())
  cv.threshold(gray, bright, darkThresh, 255, cv.THRESH_BINARY)
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(15, 15))
  const closed = ctx.track(new cv.Mat())
  cv.morphologyEx(bright, closed, cv.MORPH_CLOSE, kernel)
  kernel.delete()

  const contours = new cv.MatVector()
  const hierarchy = ctx.track(new cv.Mat())
  cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  const w = gray.cols
  const h = gray.rows
  if (contours.size() === 0) {
    contours.delete()
    return { cx: Math.floor(w / 2), cy: Math.floor(h / 2), r: Math.floor(Math.min(w, h) / 2) - margin }
  }

  let largest = contours.get(0)
  let largestArea = cv.contourArea(largest)
  for (let i = 1; i < contours.size(); i++) {
    const c = contours.get(i)
    const area = cv.contourArea(c)
    if (area > largestArea) {
      largestArea = area
      largest = c
    }
  }

  const circle = cv.minEnclosingCircle(largest)
  contours.delete()
  return { cx: Math.round(circle.center.x), cy: Math.round(circle.center.y), r: Math.max(10, Math.round(circle.radius) - margin) }
}

function hexToBgr(hex: string, fallback: [number, number, number] = [255, 255, 255]): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex ?? ''))
  if (!m) return fallback
  const int = parseInt(m[1], 16)
  const r = (int >> 16) & 255
  const g = (int >> 8) & 255
  const b = int & 255
  return [b, g, r]
}

export const removeVignette: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, mask: null, cropped: null }
  const cv = ctx.cv

  const bgr = ctx.track(toBgr(cv, src))
  const w = bgr.cols
  const h = bgr.rows
  const gray = ctx.track(toGray(cv, bgr))

  const mode = Number(params.mode) || 0
  const fillMode = Number(params.fill) || 0
  const feather = Math.round(Number(params.feather_px) || 0)
  const darkThresh = Math.round(Number(params.dark_thresh) || 30)
  const margin = Math.round(Number(params.margin_px) || 8)

  let cx: number, cy: number, r: number
  if (mode === 1) {
    cx = Math.round(((Number(params.center_x) || 50) / 100) * w)
    cy = Math.round(((Number(params.center_y) || 50) / 100) * h)
    r = Math.max(1, Math.round(((Number(params.radius_pct) || 46) / 100) * Math.min(w, h)))
  } else {
    ;({ cx, cy, r } = fitFovCircle(cv, ctx, gray, darkThresh, margin))
  }

  let fillLayer: any
  if (fillMode === 1) {
    const k = Math.max(51, Math.floor(Math.min(w, h) / 20) | 1)
    fillLayer = ctx.track(new cv.Mat())
    cv.GaussianBlur(bgr, fillLayer, new cv.Size(k, k), 0)
  } else if (fillMode === 2) {
    fillLayer = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0)))
  } else {
    const [b, g, rr] = hexToBgr(String(params.fill_color ?? '#ffffff'))
    fillLayer = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(b, g, rr)))
  }

  const result = ctx.track(bgr.clone())
  const fovMask = ctx.track(new cv.Mat(h, w, cv.CV_8U))
  const resultData = result.data as Uint8Array
  const bgrData = bgr.data as Uint8Array
  const fillData = fillLayer.data as Uint8Array
  const maskData = fovMask.data as Uint8Array

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x
      const dist = Math.hypot(x - cx, y - cy)
      const alpha = feather > 0 ? Math.max(0, Math.min(1, (r - dist) / feather)) : dist <= r ? 1 : 0
      maskData[idx] = Math.round(alpha * 255)
      const p = idx * 3
      resultData[p] = Math.round(bgrData[p] * alpha + fillData[p] * (1 - alpha))
      resultData[p + 1] = Math.round(bgrData[p + 1] * alpha + fillData[p + 1] * (1 - alpha))
      resultData[p + 2] = Math.round(bgrData[p + 2] * alpha + fillData[p + 2] * (1 - alpha))
    }
  }

  const x1 = Math.max(0, cx - r)
  const y1 = Math.max(0, cy - r)
  const x2 = Math.min(w, cx + r)
  const y2 = Math.min(h, cy + r)
  const cropped = x2 > x1 && y2 > y1 ? ctx.track(result.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1)).clone()) : result

  return { main: result, mask: fovMask, cropped }
}

export const cvUndistort: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, data: null }
  const cv = ctx.cv
  const w = src.cols
  const h = src.rows

  const focal = Number(params.focal) || 1.0
  const k1 = Number(params.k1) || 0
  const k2 = Number(params.k2) || 0
  const p1 = Number(params.p1) || 0
  const p2 = Number(params.p2) || 0
  const k3 = Number(params.k3) || 0
  const cropValid = !!params.crop_valid

  const fx = focal * w
  const fy = fx
  const cx = w / 2
  const cy = h / 2
  const cameraMatrix = cv.matFromArray(3, 3, cv.CV_64F, [fx, 0, cx, 0, fy, cy, 0, 0, 1])
  const distCoeffs = cv.matFromArray(1, 5, cv.CV_64F, [k1, k2, p1, p2, k3])

  const data = {
    camera_matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    dist_coeffs: [k1, k2, p1, p2, k3],
  }

  try {
    const dst = ctx.track(new cv.Mat())
    // getOptimalNewCameraMatrix isn't in this OpenCV.js build's calib3d bindings —
    // undistort with the original camera matrix still works, just without the
    // "crop to the valid pixels" step, so degrade to that instead of failing outright.
    if (cropValid && typeof cv.getOptimalNewCameraMatrix === 'function') {
      const newCameraMatrix = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, new cv.Size(w, h), 1, new cv.Size(w, h))
      cv.undistort(src, dst, cameraMatrix, distCoeffs, newCameraMatrix)
      newCameraMatrix.delete()
    } else {
      cv.undistort(src, dst, cameraMatrix, distCoeffs)
      if (cropValid) ctx.emit('error', 'Distortion Correction: crop-to-valid-ROI unsupported in this build, ignored.')
    }
    cameraMatrix.delete()
    distCoeffs.delete()
    return { main: dst, data }
  } catch (error) {
    cameraMatrix.delete()
    distCoeffs.delete()
    const message = error instanceof Error ? error.message : String(error)
    ctx.emit('error', `Distortion Correction: ${message}`)
    return { main: src, data }
  }
}

export const geomObb: NodeImpl = (inputs, params, ctx) => {
  const image = inputs.image as any
  const maskIn = inputs.mask as any
  if (!image) return { main: null, rotated: null, rotated_mask: null, angle: 0 }
  const cv = ctx.cv

  const overlay = ctx.track(toBgr(cv, image))
  const width = overlay.cols
  const height = overlay.rows

  let binary: any
  if (maskIn) {
    binary = ctx.track(toGray(cv, maskIn))
  } else {
    const gray = ctx.track(toGray(cv, overlay))
    binary = ctx.track(new cv.Mat())
    cv.threshold(gray, binary, 1, 255, cv.THRESH_BINARY)
  }

  const contours = new cv.MatVector()
  const hierarchy = ctx.track(new cv.Mat())
  cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  if (contours.size() === 0) {
    contours.delete()
    return { main: overlay, rotated: overlay, rotated_mask: binary, angle: 0 }
  }

  const target = String(params.target ?? 'largest')
  const groups: any[] = []
  const scratch: any[] = []
  if (target === 'largest') {
    let best = contours.get(0)
    let bestArea = cv.contourArea(best)
    for (let i = 1; i < contours.size(); i++) {
      const candidate = contours.get(i)
      const area = cv.contourArea(candidate)
      if (area > bestArea) {
        bestArea = area
        best = candidate
      }
    }
    groups.push(best)
  } else if (target === 'combined') {
    const all: number[] = []
    for (let i = 0; i < contours.size(); i++) {
      const data = contours.get(i).data32S as Int32Array
      for (let j = 0; j < data.length; j++) all.push(data[j])
    }
    const merged = cv.matFromArray(all.length / 2, 1, cv.CV_32SC2, all)
    scratch.push(merged)
    groups.push(merged)
  } else {
    for (let i = 0; i < contours.size(); i++) groups.push(contours.get(i))
  }

  const colour = parseColor(cv, String(params.color ?? '#00ff88'))
  const thickness = Math.max(1, Number(params.thickness) || 2)
  let lastRect: any = null
  for (const contour of groups) {
    const rect = cv.minAreaRect(contour)
    lastRect = rect
    if (params.draw_obb === false) continue
    drawPolyline(cv, overlay, rotatedRectPoints(rect), true, colour, thickness)
  }
  scratch.forEach((m) => m.delete())
  contours.delete()

  if (!lastRect) return { main: overlay, rotated: overlay, rotated_mask: binary, angle: 0 }

  let rectWidth = lastRect.size.width
  let rectHeight = lastRect.size.height
  let angle = lastRect.angle
  // Normalise so the long axis ends up horizontal, whichever way minAreaRect reported it.
  if (rectWidth < rectHeight) {
    angle += 90
    const swap = rectWidth
    rectWidth = rectHeight
    rectHeight = swap
  }

  let rotated = overlay
  let rotatedMask = binary
  if (params.auto_crop !== false && rectWidth > 0 && rectHeight > 0) {
    const pad = Math.round(Math.max(width, height) * 0.75)
    const cx = lastRect.center.x + pad
    const cy = lastRect.center.y + pad
    rotated = ctx.track(warpCrop(cv, overlay, cx, cy, angle, rectWidth, rectHeight, pad, cv.INTER_LINEAR))
    rotatedMask = ctx.track(warpCrop(cv, binary, cx, cy, angle, rectWidth, rectHeight, pad, cv.INTER_NEAREST))
  }

  return { main: overlay, rotated, rotated_mask: rotatedMask, angle }
}
