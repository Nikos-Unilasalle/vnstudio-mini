import type { NodeDef } from '../engine/types'

function formatValue(v: unknown): string {
  if (v == null) return '—'
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2)
  if (typeof v === 'string') return v
  if (Array.isArray(v)) return `[${v.length} éléments]`
  if (v && typeof v === 'object' && 'stats' in (v as any)) {
    const d = v as { d10?: number; d50?: number; d90?: number; unit?: string; count?: number }
    if (d.d50 != null) return `D10 ${d.d10?.toFixed(1)} · D50 ${d.d50?.toFixed(1)} · D90 ${d.d90?.toFixed(1)} ${d.unit ?? ''}`
  }
  try {
    return JSON.stringify(v)
  } catch {
    return String(v)
  }
}

export const inspectorNode: NodeDef = {
  typeId: 'data_inspector',
  label: 'Inspector',
  category: 'Visualize',
  description: 'Affiche une valeur chiffrée en clair sur le canvas.',
  inputs: [{ id: 'main', label: 'value', color: 'scalar' }],
  outputs: [],
  params: [],
  process(inputs) {
    return { __text: formatValue(inputs.main) } as any
  },
}
