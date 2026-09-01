import type { NodeDef } from '../engine/types'

export const analysisReportNode: NodeDef = {
  typeId: 'sci_analysis_report',
  label: 'Analysis Report',
  category: 'Measure',
  description: 'Affiche un tableau récapitulatif de toutes les variables. Branche un dictionnaire (stats, report, etc.).',
  inputs: [{ id: 'data', label: 'dict', color: 'dict' }],
  outputs: [{ id: 'report', label: 'report', color: 'dict' }],
  params: [{ id: 'title', label: 'Report Title', type: 'string', default: 'Analysis Report' }],
  process(inputs) {
    return { report: (inputs.data as Record<string, unknown>) ?? {} }
  },
}
