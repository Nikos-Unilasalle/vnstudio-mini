import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'
import { computeLabelStats } from './labelStats'
import { colorizeLabels } from './colorizeLabels'

export const connectedComponentsNode: NodeDef = {
  typeId: 'sci_connected_components',
  label: 'Connected Components',
  category: 'Segmentation',
  description: "Numérote les taches indépendantes d'un masque, avec filtre de surface.",
  inputs: [{ id: 'main', label: 'mask', color: 'mask' }],
  outputs: [
    { id: 'main', label: 'preview', color: 'image' },
    { id: 'regions', label: 'regions', color: 'regions' },
  ],
  params: [
    { id: 'min_area', label: 'Min Area', type: 'number', default: 200, min: 0, max: 1000000, step: 10 },
    { id: 'max_area', label: 'Max Area', type: 'number', default: 500000, min: 0, max: 5000000, step: 100 },
    {
      id: 'connectivity',
      label: 'Connectivity',
      type: 'select',
      default: 0,
      options: [
        { label: '8-connexe', value: 0 },
        { label: '4-connexe', value: 1 },
      ],
    },
  ],
  process(inputs, params, ctx) {
    const src = inputs.main as any
    if (!src) return { main: undefined, regions: undefined }
    const cv = ctx.cv
    const labels = trackMat(new cv.Mat())
    const connectivity = Number(params.connectivity) === 1 ? 4 : 8
    cv.connectedComponents(src, labels, connectivity, cv.CV_32S)

    const minArea = Number(params.min_area)
    const maxArea = Number(params.max_area)
    const stats = computeLabelStats(labels)
    const data = labels.data32S as Int32Array
    for (let i = 0; i < data.length; i++) {
      const label = data[i]
      if (label <= 0) continue
      const s = stats.get(label)
      if (!s || s.area < minArea || s.area > maxArea) data[i] = 0
    }
    // recompute stats after filtering so downstream count/area reflect the filter
    const filteredStats = computeLabelStats(labels)

    const preview = colorizeLabels(cv, labels)
    trackMat(preview)

    return { main: preview, regions: { labels, count: filteredStats.size, stats: filteredStats } }
  },
}
