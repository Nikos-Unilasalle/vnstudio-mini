import type { NodeDef } from '../engine/types'

export const dictGetNode: NodeDef = {
  typeId: 'plugin_dict_get',
  label: 'Dict Get',
  category: 'Data',
  description: 'Extrait 1 à 3 valeurs d\'un dictionnaire par nom de clé. scalar_1 force val_1 en nombre.',
  inputs: [{ id: 'dict_in', label: 'dict', color: 'dict' }],
  outputs: [
    { id: 'val_1', label: 'val_1', color: 'dict' },
    { id: 'val_2', label: 'val_2', color: 'dict' },
    { id: 'val_3', label: 'val_3', color: 'dict' },
    { id: 'scalar_1', label: 'scalar_1', color: 'scalar' },
    { id: 'dict_1', label: 'dict_1', color: 'dict' },
  ],
  params: [
    { id: 'key_1', label: 'Key 1', type: 'string', default: 'xmin' },
    { id: 'key_2', label: 'Key 2', type: 'string', default: '' },
    { id: 'key_3', label: 'Key 3', type: 'string', default: '' },
  ],
  process(inputs, params) {
    const d = inputs.dict_in as Record<string, unknown> | undefined
    if (!d || typeof d !== 'object') {
      return { val_1: null, val_2: null, val_3: null, scalar_1: 0, dict_1: null }
    }
    const k1 = String(params.key_1 ?? '').trim()
    const k2 = String(params.key_2 ?? '').trim()
    const k3 = String(params.key_3 ?? '').trim()
    const v1 = k1 ? d[k1] : undefined
    const v2 = k2 ? d[k2] : undefined
    const v3 = k3 ? d[k3] : undefined

    const s1 = typeof v1 === 'number' ? v1 : typeof v1 === 'string' && v1 !== '' && !Number.isNaN(Number(v1)) ? Number(v1) : 0
    const dict1 = v1 && typeof v1 === 'object' && !Array.isArray(v1) ? v1 : null

    return { val_1: v1 ?? null, val_2: v2 ?? null, val_3: v3 ?? null, scalar_1: s1, dict_1: dict1 }
  },
}
