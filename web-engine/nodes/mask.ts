import type { NodeImpl } from '../types'
import { toGray } from '../cvUtils'

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
