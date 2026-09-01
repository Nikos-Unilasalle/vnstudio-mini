import type { NodeDef } from '../engine/types'
import { computeLabelStats } from './labelStats'

export const regionFilterNode: NodeDef = {
  typeId: 'sci_region_filter',
  label: 'Region Filter',
  category: 'Measure',
  description: 'Filtre des régions par surface, après le Watershed. Écarte le label du fond.',
  inputs: [{ id: 'main', label: 'regions', color: 'regions' }],
  outputs: [{ id: 'main', label: 'regions', color: 'regions' }],
  params: [
    { id: 'min_area', label: 'Min Area', type: 'number', default: 200, min: 0, max: 1000000, step: 10 },
    { id: 'max_area', label: 'Max Area', type: 'number', default: 150000, min: 0, max: 5000000, step: 100 },
  ],
  process(inputs, params) {
    const regions = inputs.main as { labels: any; stats: Map<number, any> } | undefined
    if (!regions) return { main: undefined }
    const minArea = Number(params.min_area)
    const maxArea = Number(params.max_area)
    const labels = regions.labels
    const data = labels.data32S as Int32Array
    for (let i = 0; i < data.length; i++) {
      const label = data[i]
      if (label <= 0) continue
      const s = regions.stats.get(label)
      if (!s || s.area < minArea || s.area > maxArea) data[i] = 0
    }
    const stats = computeLabelStats(labels)
    return { main: { labels, count: stats.size, stats } }
  },
}
