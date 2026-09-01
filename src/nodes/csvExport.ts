import type { NodeDef } from '../engine/types'
import type { MeasuredRegion } from './regionProps'

interface RecordState {
  rows: Record<string, unknown>[]
  lastRecord: boolean
}

export const csvExportNode: NodeDef = {
  typeId: 'util_csv_export',
  label: 'CSV Export',
  category: 'Output',
  description:
    "Sort des données vers un CSV téléchargeable. Branche 'regions' pour exporter une table entière d'un coup (granulométrie, empreintes…), ou active Recording pour empiler une ligne à chaque exécution — utile pour enregistrer une mesure qui évolue image après image sur une vidéo.",
  inputs: [
    { id: 'main', label: 'regions (table complète)', color: 'regions' },
    { id: 'value_1', label: 'value_1', color: 'scalar' },
    { id: 'value_2', label: 'value_2', color: 'scalar' },
    { id: 'value_3', label: 'value_3', color: 'scalar' },
  ],
  outputs: [],
  params: [
    { id: 'filename', label: 'Filename', type: 'string', default: 'export.csv' },
    { id: 'name_1', label: 'Nom colonne 1', type: 'string', default: 'value_1' },
    { id: 'name_2', label: 'Nom colonne 2', type: 'string', default: 'value_2' },
    { id: 'name_3', label: 'Nom colonne 3', type: 'string', default: 'value_3' },
    { id: 'record', label: 'Recording', type: 'boolean', default: false },
  ],
  process(inputs, params, ctx) {
    const regions = inputs.main as MeasuredRegion[] | undefined
    if (regions && regions.length) {
      const header = 'id,area_px,equiv_diameter_px,equiv_diameter_um,cx,cy'
      const rows = regions.map(
        (r) => `${r.id},${r.areaPx},${r.equivDiameterPx.toFixed(3)},${r.equivDiameterUm?.toFixed(3) ?? ''},${r.cx.toFixed(1)},${r.cy.toFixed(1)}`
      )
      const csv = [header, ...rows].join('\n')
      return { __csv: csv, __filename: (params.filename as string) || 'export.csv' } as any
    }

    let state: RecordState = ctx.nodeState.get(ctx.nodeId)
    if (!state) {
      state = { rows: [], lastRecord: false }
      ctx.nodeState.set(ctx.nodeId, state)
    }
    if (params.record && !state.lastRecord) state.rows = []
    state.lastRecord = !!params.record

    const cols = [String(params.name_1 || 'value_1'), String(params.name_2 || 'value_2'), String(params.name_3 || 'value_3')]
    const vals = [inputs.value_1, inputs.value_2, inputs.value_3]
    const hasAny = vals.some((v) => typeof v === 'number')

    if (params.record && hasAny) {
      const row: Record<string, unknown> = { tick: state.rows.length }
      cols.forEach((c, i) => (row[c] = typeof vals[i] === 'number' ? vals[i] : ''))
      state.rows.push(row)
    }

    if (state.rows.length === 0) return {}
    const header = ['tick', ...cols].join(',')
    const csv = [header, ...state.rows.map((r) => ['tick', ...cols].map((c) => r[c] ?? '').join(','))].join('\n')
    return { __csv: csv, __filename: (params.filename as string) || 'export.csv' } as any
  },
}
