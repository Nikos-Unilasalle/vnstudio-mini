import type { NodeDef } from '../engine/types'

export const coordCombineNode: NodeDef = {
  typeId: 'data_coord_combine',
  label: 'Coord Combine',
  category: 'Data',
  description: 'Rassemble 2 à 4 valeurs scalaires en un objet coordonnée unique.',
  inputs: [
    { id: 'x', label: 'x', color: 'scalar' },
    { id: 'y', label: 'y', color: 'scalar' },
    { id: 'w', label: 'w', color: 'scalar' },
    { id: 'h', label: 'h', color: 'scalar' },
  ],
  outputs: [{ id: 'dict_out', label: 'dict', color: 'dict' }],
  params: [],
  process(inputs) {
    const x = inputs.x
    const y = inputs.y
    if (typeof x !== 'number' || typeof y !== 'number') return { dict_out: null }
    return {
      dict_out: {
        x,
        y,
        xmin: x,
        ymin: y,
        width: typeof inputs.w === 'number' ? inputs.w : 0,
        height: typeof inputs.h === 'number' ? inputs.h : 0,
      },
    }
  },
}

export const coordSplitterNode: NodeDef = {
  typeId: 'data_coord_splitter',
  label: 'Coord Splitter',
  category: 'Data',
  description: 'Décompose un objet coordonnée en 4 scalaires x/y/w/h.',
  inputs: [{ id: 'data', label: 'coords', color: 'dict' }],
  outputs: [
    { id: 'x', label: 'x', color: 'scalar' },
    { id: 'y', label: 'y', color: 'scalar' },
    { id: 'w', label: 'w', color: 'scalar' },
    { id: 'h', label: 'h', color: 'scalar' },
  ],
  params: [],
  process(inputs) {
    const d = inputs.data as Record<string, number> | null | undefined
    if (!d) return { x: 0, y: 0, w: 0, h: 0 }
    return {
      x: d.x ?? d.xmin ?? 0,
      y: d.y ?? d.ymin ?? 0,
      w: d.width ?? 0,
      h: d.height ?? 0,
    }
  },
}
