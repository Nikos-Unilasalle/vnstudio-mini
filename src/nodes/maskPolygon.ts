import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

interface NormPoint {
  x: number
  y: number
}

export const maskPolygonNode: NodeDef = {
  typeId: 'util_roi_polygon',
  label: 'Mask Polygon',
  category: 'Geometry',
  description: 'Région d\'intérêt polygonale interactive — entoure une zone à la souris. Sans polygone tracé : laisse tout passer.',
  interactive: true,
  inputs: [
    { id: 'image', label: 'image', color: 'image' },
    { id: 'mask_in', label: 'mask (optionnel)', color: 'mask' },
  ],
  outputs: [
    { id: 'main', label: 'image', color: 'image' },
    { id: 'mask', label: 'mask', color: 'mask' },
    { id: 'masked', label: 'masked', color: 'image' },
    { id: 'masked_inv', label: 'masked_inv', color: 'image' },
  ],
  params: [
    { id: 'points', label: 'Points', type: 'string', default: '[]' },
    { id: 'filled', label: 'Filled', type: 'boolean', default: true },
    { id: 'thickness', label: 'Thickness', type: 'number', default: 2, min: 1, max: 20, step: 1 },
  ],
  process(inputs, params, ctx) {
    const src = inputs.image as any
    const maskIn = inputs.mask_in as any
    if (!src) return { main: undefined, mask: undefined, masked: undefined, masked_inv: undefined }
    const cv = ctx.cv
    const w = src.cols
    const h = src.rows

    let ptsData: NormPoint[] = []
    try {
      ptsData = JSON.parse(String(params.points ?? '[]'))
    } catch {
      ptsData = []
    }

    const mask = trackMat(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(ptsData.length === 0 ? 255 : 0)))

    if (ptsData.length >= 3) {
      const pts = ptsData.flatMap((p) => [Math.round(p.x * w), Math.round(p.y * h)])
      const ptsMat = cv.matFromArray(ptsData.length, 1, cv.CV_32SC2, pts)
      const vec = new cv.MatVector()
      vec.push_back(ptsMat)
      if (params.filled) {
        cv.fillPoly(mask, vec, new cv.Scalar(255))
      } else {
        cv.polylines(mask, vec, true, new cv.Scalar(255), Number(params.thickness))
      }
      ptsMat.delete()
      vec.delete()
    }

    if (maskIn) {
      const mi = trackMat(new cv.Mat())
      if (maskIn.channels() === 1) maskIn.copyTo(mi)
      else cv.cvtColor(maskIn, mi, cv.COLOR_BGR2GRAY)
      cv.bitwise_and(mask, mi, mask)
    }

    const masked = trackMat(new cv.Mat())
    src.copyTo(masked, mask)

    const maskInv = trackMat(new cv.Mat())
    cv.bitwise_not(mask, maskInv)
    const maskedInv = trackMat(new cv.Mat())
    src.copyTo(maskedInv, maskInv)

    return { main: src, mask, masked, masked_inv: maskedInv }
  },
}
