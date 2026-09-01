import type { NodeDef } from '../engine/types'

export const visualCalibrationNode: NodeDef = {
  typeId: 'geo_visual_calibration',
  label: 'Visual Calibration',
  category: 'Measure',
  description: "Trace une ligne de longueur connue sur l'image (ex: la pièce de 2€) pour en déduire l'échelle.",
  interactive: true,
  inputs: [{ id: 'main', label: 'image', color: 'image' }],
  outputs: [
    { id: 'main', label: 'image', color: 'image' },
    { id: 'px_per_unit', label: 'Px/Unit', color: 'scalar' },
    { id: 'um_per_px', label: 'µm/px', color: 'scalar' },
  ],
  params: [
    { id: 'known_length', label: 'Known Length', type: 'number', default: 25.75, min: 0.001, max: 100000, step: 0.01 },
    { id: 'unit_name', label: 'Unit Name', type: 'select', default: 'mm', options: [{ label: 'mm', value: 'mm' }, { label: 'cm', value: 'cm' }] },
  ],
  process(inputs, params) {
    const src = inputs.main as any
    const line = params.__line as { x1: number; y1: number; x2: number; y2: number } | undefined
    if (!src) return { main: undefined, px_per_unit: 0, um_per_px: 0 }
    if (!line) return { main: src, px_per_unit: 0, um_per_px: 0 }

    const pxLength = Math.hypot(line.x2 - line.x1, line.y2 - line.y1)
    const knownLength = Number(params.known_length)
    const unit = params.unit_name as string
    const toMm = unit === 'cm' ? 10 : 1

    const pxPerUnit = pxLength / knownLength
    const mmPerPx = (knownLength * toMm) / pxLength
    const umPerPx = mmPerPx * 1000

    return { main: src, px_per_unit: pxPerUnit, um_per_px: umPerPx }
  },
}
