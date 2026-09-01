import type { NodeDef } from '../engine/types'

export const listSelectorNode: NodeDef = {
  typeId: 'data_list_selector',
  label: 'List Selector',
  category: 'Data',
  description: "Récupère un élément d'une liste par son index — sert par ex. à relire une ligne précalculée d'un CSV Import à l'image courante.",
  inputs: [{ id: 'list_in', label: 'list', color: 'list' }],
  outputs: [{ id: 'item_out', label: 'item', color: 'dict' }],
  params: [{ id: 'index', label: 'Index', type: 'number', default: 0, min: 0, max: 1000000, step: 1 }],
  process(inputs, params) {
    const list = inputs.list_in
    const index = Number(params.index)
    if (!Array.isArray(list) || index < 0 || index >= list.length) return { item_out: null }
    return { item_out: list[index] }
  },
}
