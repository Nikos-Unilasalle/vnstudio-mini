import type { NodeImpl } from '../types'
import { parseColor, toBgr, toGray } from '../cvUtils'

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
    const flat = points.flatMap((p) => [Math.round(p.x * w), Math.round(p.y * h)])
    const pointsMat = cv.matFromArray(points.length, 1, cv.CV_32SC2, flat)
    const vector = new cv.MatVector()
    vector.push_back(pointsMat)
    if (points.length >= 3 && params.filled !== false) {
      cv.fillPoly(mask, vector, new cv.Scalar(255))
    } else {
      cv.polylines(mask, vector, points.length >= 3, new cv.Scalar(255), Math.max(1, Number(params.thickness) || 2))
    }
    pointsMat.delete()
    vector.delete()
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
    const flat = points.flatMap((p) => [Math.round(p.x * w), Math.round(p.y * h)])
    const pointsMat = cv.matFromArray(points.length, 1, cv.CV_32SC2, flat)
    const vector = new cv.MatVector()
    vector.push_back(pointsMat)
    cv.polylines(overlay, vector, true, new cv.Scalar(0, 255, 136, 255), 2, cv.LINE_AA)
    pointsMat.delete()
    vector.delete()
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
    const corners = rotatedRectPoints(rect).flatMap((p) => [Math.round(p.x), Math.round(p.y)])
    const boxMat = cv.matFromArray(4, 1, cv.CV_32SC2, corners)
    const vector = new cv.MatVector()
    vector.push_back(boxMat)
    cv.polylines(overlay, vector, true, colour, thickness, cv.LINE_AA)
    boxMat.delete()
    vector.delete()
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
