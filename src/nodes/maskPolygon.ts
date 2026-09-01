import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const maskPolygonNode: NodeDef = {
  typeId: 'geo_mask_polygon',
  label: 'Mask Polygon',
  category: 'Mask',
  description: 'Entoure une zone à la souris (clic pour poser un point, "Fermer" pour boucler). masked_inv exclut cette zone.',
  interactive: true,
  inputs: [{ id: 'main', label: 'image', color: 'image' }],
  outputs: [
    { id: 'masked', label: 'masked', color: 'image' },
    { id: 'masked_inv', label: 'masked_inv', color: 'image' },
    { id: 'mask', label: 'mask', color: 'mask' },
  ],
  params: [],
  process(inputs, params, ctx) {
    const src = inputs.main as any
    const polygon = (params.__polygon as { x: number; y: number }[]) ?? []
    if (!src) return { masked: undefined, masked_inv: undefined, mask: undefined }
    const cv = ctx.cv
    const mask = trackMat(cv.Mat.zeros(src.rows, src.cols, cv.CV_8U))
    if (polygon.length >= 3) {
      const pts = polygon.flatMap((p) => [Math.round(p.x), Math.round(p.y)])
      const matVec = cv.matFromArray(polygon.length, 1, cv.CV_32SC2, pts)
      const vec = new cv.MatVector()
      vec.push_back(matVec)
      cv.fillPoly(mask, vec, new cv.Scalar(255))
      matVec.delete()
      vec.delete()
    }
    const masked = trackMat(new cv.Mat())
    src.copyTo(masked, mask)

    const maskInv = trackMat(new cv.Mat())
    cv.bitwise_not(mask, maskInv)
    const maskedInv = trackMat(new cv.Mat())
    src.copyTo(maskedInv, maskInv)

    return { masked, masked_inv: maskedInv, mask }
  },
}
