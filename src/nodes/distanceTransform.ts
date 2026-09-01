import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const distanceTransformNode: NodeDef = {
  typeId: 'sci_distance_transform',
  label: 'Distance Transform',
  category: 'Segmentation',
  description: 'Pour chaque pixel du masque, sa distance au bord le plus proche. main = normalisée 0-255, dist_map = brute en pixels.',
  inputs: [{ id: 'main', label: 'mask', color: 'mask' }],
  outputs: [
    { id: 'main', label: 'main (0-255)', color: 'image' },
    { id: 'dist_map', label: 'dist_map (px)', color: 'image' },
  ],
  params: [],
  process(inputs, _params, ctx) {
    const src = inputs.main as any
    if (!src) return { main: undefined, dist_map: undefined }
    const cv = ctx.cv
    const distRaw = trackMat(new cv.Mat())
    cv.distanceTransform(src, distRaw, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    const normalized = trackMat(new cv.Mat())
    cv.normalize(distRaw, normalized, 0, 255, cv.NORM_MINMAX)
    const normalized8u = trackMat(new cv.Mat())
    normalized.convertTo(normalized8u, cv.CV_8U)

    return { main: normalized8u, dist_map: distRaw }
  },
}
