import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'
import { colorizeLabels } from './colorizeLabels'

export interface MeasuredRegion {
  id: number
  areaPx: number
  equivDiameterPx: number
  equivDiameterUm: number | null
  cx: number
  cy: number
}

export const regionPropsNode: NodeDef = {
  typeId: 'sci_region_props',
  label: 'Region Properties',
  category: 'Measure',
  description: 'Mesure chaque objet : aire, diamètre équivalent, centroïde. Applique la calibration µm/px.',
  inputs: [
    { id: 'labels_map', label: 'Label Map', color: 'regions' },
    { id: 'image', label: 'intensity (optionnel)', color: 'image' },
    { id: 'um_per_px', label: 'µm/pixel (calibration)', color: 'scalar' },
  ],
  outputs: [
    { id: 'regions', label: 'regions', color: 'regions' },
    { id: 'count', label: 'count', color: 'scalar' },
    { id: 'main', label: 'preview', color: 'image' },
  ],
  params: [
    { id: 'um_per_px', label: 'µm/pixel (manuel)', type: 'number', default: 1000, min: 0.001, max: 1000000, step: 1 },
    { id: 'show_ids', label: 'Show IDs', type: 'boolean', default: true },
  ],
  process(inputs, params, ctx) {
    const regions = inputs.labels_map as { labels: any; stats: Map<number, any> } | undefined
    if (!regions) return { main: undefined, regions: undefined, count: 0 }
    const cv = ctx.cv

    const umPerPixel = typeof inputs.um_per_px === 'number' ? (inputs.um_per_px as number) : Number(params.um_per_px)

    const measured: MeasuredRegion[] = []
    for (const [id, s] of regions.stats) {
      const equivDiameterPx = 2 * Math.sqrt(s.area / Math.PI)
      measured.push({
        id,
        areaPx: s.area,
        equivDiameterPx,
        equivDiameterUm: umPerPixel ? equivDiameterPx * umPerPixel : null,
        cx: s.cx,
        cy: s.cy,
      })
    }

    const intensity = inputs.image as any
    const preview = trackMat(colorizeLabels(cv, regions.labels))
    if (intensity) {
      const base = trackMat(new cv.Mat())
      if (intensity.channels() === 1) cv.cvtColor(intensity, base, cv.COLOR_GRAY2BGR)
      else intensity.copyTo(base)
      cv.addWeighted(base, 0.5, preview, 0.5, 0, preview)
    }

    if (params.show_ids) {
      for (const m of measured) {
        cv.putText(
          preview,
          String(m.id),
          new cv.Point(Math.round(m.cx) - 8, Math.round(m.cy) + 4),
          cv.FONT_HERSHEY_SIMPLEX,
          0.5,
          new cv.Scalar(255, 255, 255, 255),
          1,
          cv.LINE_AA
        )
      }
    }

    return { main: preview, regions: measured, count: measured.length }
  },
}
