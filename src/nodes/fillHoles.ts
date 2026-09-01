import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const fillHolesNode: NodeDef = {
  typeId: 'fill_holes',
  label: 'Fill Holes',
  category: 'Mask',
  description:
    "Bouche les trous internes d'un masque binaire. Contour Fill: comble tout, rapide. Flood Fill: préserve les trous qui touchent le bord. Size Filter: ne comble que les trous plus petits que Max Hole.",
  inputs: [{ id: 'mask', label: 'mask', color: 'mask' }],
  outputs: [{ id: 'main', label: 'filled mask', color: 'mask' }],
  params: [
    {
      id: 'method',
      label: 'Method',
      type: 'select',
      default: 0,
      options: [
        { label: 'Contour Fill', value: 0 },
        { label: 'Flood Fill', value: 1 },
        { label: 'Size Filter', value: 2 },
      ],
    },
    { id: 'max_hole_px', label: 'Max Hole (px²)', type: 'number', default: 500, min: 1, max: 50000, step: 1 },
  ],
  process(inputs, params, ctx) {
    const src = inputs.mask as any
    if (!src) return { main: undefined }
    const cv = ctx.cv
    const method = Number(params.method)

    if (method === 0) {
      const contours = new cv.MatVector()
      const hierarchy = trackMat(new cv.Mat())
      cv.findContours(src, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
      const filled = trackMat(cv.Mat.zeros(src.rows, src.cols, cv.CV_8U))
      cv.drawContours(filled, contours, -1, new cv.Scalar(255), -1)
      contours.delete()
      return { main: filled }
    }

    // flood-fill the background from (0,0) on a padded copy, then invert to get the holes
    const padded = trackMat(new cv.Mat())
    cv.copyMakeBorder(src, padded, 1, 1, 1, 1, cv.BORDER_CONSTANT, new cv.Scalar(0))
    const floodFilled = trackMat(new cv.Mat())
    padded.copyTo(floodFilled)
    const ffMask = trackMat(cv.Mat.zeros(padded.rows + 2, padded.cols + 2, cv.CV_8U))
    cv.floodFill(floodFilled, ffMask, new cv.Point(0, 0), new cv.Scalar(255))

    const floodInv = trackMat(new cv.Mat())
    cv.bitwise_not(floodFilled, floodInv)
    const holes = trackMat(floodInv.roi(new cv.Rect(1, 1, src.cols, src.rows)))

    if (method === 1) {
      const filled = trackMat(new cv.Mat())
      cv.bitwise_or(src, holes, filled)
      return { main: filled }
    }

    // Size Filter: only fill holes smaller than max_hole_px
    const maxHolePx = Number(params.max_hole_px)
    const holeContours = new cv.MatVector()
    const hh = trackMat(new cv.Mat())
    cv.findContours(holes, holeContours, hh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    const smallHoles = trackMat(cv.Mat.zeros(src.rows, src.cols, cv.CV_8U))
    for (let i = 0; i < holeContours.size(); i++) {
      const c = holeContours.get(i)
      if (cv.contourArea(c) <= maxHolePx) {
        const vec = new cv.MatVector()
        vec.push_back(c)
        cv.drawContours(smallHoles, vec, -1, new cv.Scalar(255), -1)
        vec.delete()
      }
    }
    holeContours.delete()
    const filled = trackMat(new cv.Mat())
    cv.bitwise_or(src, smallHoles, filled)
    return { main: filled }
  },
}
