import type { NodeDef } from '../engine/types'

export const pointTrackerNode: NodeDef = {
  typeId: 'geom_track_point',
  label: 'Point Tracker',
  category: 'Tracking',
  description: "Extrait les coordonnées précises d'un point de repère suivi (ex: un coin de bouche), par son numéro.",
  inputs: [
    { id: 'data', label: 'data', color: 'dict' },
    { id: 'image', label: 'image', color: 'image' },
  ],
  outputs: [
    { id: 'x', label: 'x', color: 'scalar' },
    { id: 'y', label: 'y', color: 'scalar' },
    { id: 'draw', label: 'draw', color: 'dict' },
  ],
  params: [
    { id: 'point_id', label: 'Point ID', type: 'number', default: 8, min: 0, max: 477, step: 1 },
    { id: 'absolute', label: 'Absolute Coords', type: 'boolean', default: false },
  ],
  process(inputs, params) {
    const data = inputs.data as { landmarks?: { x: number; y: number }[] } | null | undefined
    const out = { x: 0, y: 0, draw: null as any }
    if (!data || !Array.isArray(data.landmarks)) return out

    const ptId = Number(params.point_id)
    const lms = data.landmarks
    if (ptId < 0 || ptId >= lms.length) return out

    const lm = lms[ptId]
    let wMult = 1
    let hMult = 1
    if (params.absolute) {
      const image = inputs.image as any
      if (image) {
        wMult = image.cols
        hMult = image.rows
      } else {
        wMult = 640
        hMult = 480
      }
    }

    return {
      x: lm.x * wMult,
      y: lm.y * hMult,
      draw: { x: lm.x, y: lm.y, relative: !params.absolute },
    }
  },
}
