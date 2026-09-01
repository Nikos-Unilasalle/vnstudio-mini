import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const distanceTransformNode: NodeDef = {
  typeId: 'feat_distance_transform',
  label: 'Distance Transform',
  category: 'Segmentation',
  description: 'Pour chaque pixel du masque, sa distance au bord le plus proche. main = normalisée 0-255, dist_map = brute en pixels.',
  inputs: [{ id: 'mask', label: 'mask', color: 'mask' }],
  outputs: [
    { id: 'main', label: 'main (0-255)', color: 'image' },
    { id: 'dist_map', label: 'dist_map (px)', color: 'image' },
  ],
  params: [
    {
      id: 'dist_type',
      label: 'Dist Type',
      type: 'select',
      default: 0,
      options: [
        { label: 'L2 (Euclidean)', value: 0 },
        { label: 'L1 (Manhattan)', value: 1 },
        { label: 'C (Chessboard)', value: 2 },
      ],
    },
    {
      id: 'mask_size',
      label: 'Mask Size',
      type: 'select',
      default: 1,
      options: [
        { label: '3', value: 0 },
        { label: '5', value: 1 },
        { label: 'Precise', value: 2 },
      ],
    },
  ],
  process(inputs, params, ctx) {
    const src = inputs.mask as any
    if (!src) return { main: undefined, dist_map: undefined }
    const cv = ctx.cv
    const distTypeMap = [cv.DIST_L2, cv.DIST_L1, cv.DIST_C]
    const maskSizeMap = [3, 5, cv.DIST_MASK_PRECISE]
    const distRaw = trackMat(new cv.Mat())
    cv.distanceTransform(src, distRaw, distTypeMap[Number(params.dist_type)], maskSizeMap[Number(params.mask_size)])

    const normalized = trackMat(new cv.Mat())
    cv.normalize(distRaw, normalized, 0, 255, cv.NORM_MINMAX)
    const normalized8u = trackMat(new cv.Mat())
    normalized.convertTo(normalized8u, cv.CV_8U)

    return { main: normalized8u, dist_map: distRaw }
  },
}
