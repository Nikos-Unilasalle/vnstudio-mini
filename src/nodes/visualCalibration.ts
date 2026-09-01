import type { NodeDef } from '../engine/types'

const UM_PER_UNIT: Record<string, number> = { 'µm': 1, mm: 1000, cm: 10000, m: 1000000, in: 25400 }

export const visualCalibrationNode: NodeDef = {
  typeId: 'sci_interactive_calibration',
  label: 'Visual Calibration',
  category: 'Measure',
  description: "Trace une ligne de longueur connue sur l'image (ex: la pièce de 2€) pour en déduire l'échelle.",
  interactive: true,
  inputs: [{ id: 'image', label: 'image', color: 'image' }],
  outputs: [
    { id: 'factor', label: 'Px/Unit', color: 'scalar' },
    { id: 'um_per_px', label: 'µm/px', color: 'scalar' },
    { id: 'main', label: 'image', color: 'image' },
  ],
  params: [
    { id: 'points', label: 'Line Points', type: 'string', default: '[]' },
    { id: 'real_len', label: 'Known Length', type: 'number', default: 25.75, min: 0.001, max: 100000, step: 0.01 },
    {
      id: 'unit',
      label: 'Unit Name',
      type: 'select',
      default: 'mm',
      options: [
        { label: 'µm', value: 'µm' },
        { label: 'mm', value: 'mm' },
        { label: 'cm', value: 'cm' },
        { label: 'm', value: 'm' },
        { label: 'in', value: 'in' },
      ],
    },
  ],
  process(inputs, params) {
    const src = inputs.image as any
    if (!src) return { main: undefined, factor: 0, um_per_px: 0 }

    let points: { x: number; y: number }[] = []
    try {
      points = JSON.parse(String(params.points ?? '[]'))
    } catch {
      points = []
    }
    if (points.length !== 2) return { main: src, factor: 0, um_per_px: 0 }

    const w = src.cols
    const h = src.rows
    const p0 = { x: points[0].x * w, y: points[0].y * h }
    const p1 = { x: points[1].x * w, y: points[1].y * h }
    const pxLength = Math.hypot(p1.x - p0.x, p1.y - p0.y)
    const realLen = Number(params.real_len)
    const unit = params.unit as string
    const umPerUnit = UM_PER_UNIT[unit] ?? 0

    const factor = pxLength / realLen // px per unit
    const umPerPx = umPerUnit ? (realLen * umPerUnit) / pxLength : 0

    return { main: src, factor, um_per_px: umPerPx }
  },
}
