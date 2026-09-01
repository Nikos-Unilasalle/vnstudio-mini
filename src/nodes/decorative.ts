import type { NodeDef } from '../engine/types'

export const canvasFrameNode: NodeDef = {
  typeId: 'canvas_frame',
  label: 'Frame',
  category: 'Decorative',
  decorative: true,
  inputs: [],
  outputs: [],
  params: [
    { id: 'title', label: 'Titre', type: 'string', default: 'Groupe' },
    { id: 'color_index', label: 'Couleur', type: 'number', default: 0 },
  ],
}

export const canvasNoteNode: NodeDef = {
  typeId: 'canvas_note',
  label: 'Note',
  category: 'Decorative',
  decorative: true,
  inputs: [],
  outputs: [],
  params: [
    { id: 'text', label: 'Texte', type: 'string', default: '' },
    { id: 'color_index', label: 'Couleur', type: 'number', default: 0 },
  ],
}
