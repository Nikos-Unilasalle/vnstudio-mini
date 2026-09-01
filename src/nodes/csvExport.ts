import type { NodeDef } from '../engine/types'
import type { MeasuredRegion } from './regionProps'

export const csvExportNode: NodeDef = {
  typeId: 'output_csv_export',
  label: 'CSV Export',
  category: 'Output',
  description: 'Sort le tableau de mesures vers un fichier CSV téléchargeable.',
  inputs: [{ id: 'main', label: 'regions', color: 'regions' }],
  outputs: [],
  params: [{ id: 'filename', label: 'Filename', type: 'string', default: 'mesures.csv' }],
  process(inputs, params) {
    const regions = inputs.main as MeasuredRegion[] | undefined
    if (!regions || !regions.length) return {}
    const header = 'id,area_px,equiv_diameter_px,equiv_diameter_um,cx,cy'
    const rows = regions.map(
      (r) => `${r.id},${r.areaPx},${r.equivDiameterPx.toFixed(3)},${r.equivDiameterUm?.toFixed(3) ?? ''},${r.cx.toFixed(1)},${r.cy.toFixed(1)}`
    )
    const csv = [header, ...rows].join('\n')
    return { __csv: csv, __filename: (params.filename as string) || 'mesures.csv' } as any
  },
}
