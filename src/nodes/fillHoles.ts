import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const fillHolesNode: NodeDef = {
  typeId: 'geo_fill_holes',
  label: 'Fill Holes',
  category: 'Mask',
  description: "Bouche les trous entièrement entourés de matière, sans souder deux objets voisins.",
  inputs: [{ id: 'main', label: 'mask', color: 'mask' }],
  outputs: [{ id: 'main', label: 'mask', color: 'mask' }],
  params: [],
  process(inputs, _params, ctx) {
    const src = inputs.main as any
    if (!src) return { main: undefined }
    const cv = ctx.cv

    // flood-fill the background from (0,0) on a padded copy, then invert to get the holes
    const padded = trackMat(new cv.Mat())
    cv.copyMakeBorder(src, padded, 1, 1, 1, 1, cv.BORDER_CONSTANT, new cv.Scalar(0))
    const floodFilled = trackMat(new cv.Mat())
    padded.copyTo(floodFilled)
    const mask = trackMat(new cv.Mat.zeros(padded.rows + 2, padded.cols + 2, cv.CV_8U))
    cv.floodFill(floodFilled, mask, new cv.Point(0, 0), new cv.Scalar(255))

    const floodInv = trackMat(new cv.Mat())
    cv.bitwise_not(floodFilled, floodInv)

    const cropped = trackMat(floodInv.roi(new cv.Rect(1, 1, src.cols, src.rows)))
    const filled = trackMat(new cv.Mat())
    cv.bitwise_or(src, cropped, filled)

    return { main: filled }
  },
}
