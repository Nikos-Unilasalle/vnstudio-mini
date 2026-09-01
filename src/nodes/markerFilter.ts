import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'
import { computeLabelStats } from './labelStats'

export const markerFilterNode: NodeDef = {
  typeId: 'sci_marker_filter',
  label: 'Marker Filter',
  category: 'Segmentation',
  description: 'Élimine les graines trop petites avant le watershed, et renumérote les graines restantes.',
  inputs: [{ id: 'markers', label: 'markers', color: 'regions' }],
  outputs: [{ id: 'markers', label: 'markers', color: 'regions' }],
  params: [{ id: 'min_area', label: 'Min Area', type: 'number', default: 5, min: 0, max: 100000, step: 1 }],
  process(inputs, params, ctx) {
    const src = inputs.markers as any
    if (!src) return { markers: undefined }
    const cv = ctx.cv
    const minArea = Number(params.min_area)
    const stats = computeLabelStats(src)

    const remap = new Map<number, number>()
    let next = 1
    for (const [id, s] of stats) {
      if (s.area >= minArea) {
        remap.set(id, next)
        next++
      }
    }

    const dst = trackMat(new cv.Mat())
    src.copyTo(dst)
    const data = dst.data32S as Int32Array
    for (let i = 0; i < data.length; i++) {
      const label = data[i]
      if (label <= 0) continue
      data[i] = remap.get(label) ?? 0
    }
    return { markers: dst }
  },
}
