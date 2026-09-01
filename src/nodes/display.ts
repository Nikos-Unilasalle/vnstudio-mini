import type { NodeDef } from '../engine/types'

export const displayNode: NodeDef = {
  typeId: 'output_display',
  label: 'Display',
  category: 'Output',
  description: "Affiche une image. À poser partout où tu veux comprendre ce qui se passe.",
  inputs: [{ id: 'main', label: 'image', color: 'image' }],
  outputs: [{ id: 'main', label: 'image', color: 'image' }],
  params: [],
  process(inputs) {
    return { main: inputs.main }
  },
}
