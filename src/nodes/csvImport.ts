import type { NodeDef } from '../engine/types'

function parseCsv(text: string): Record<string, unknown>[] {
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0)
  if (lines.length === 0) return []
  const header = lines[0].split(',').map((h) => h.trim())
  const rows: Record<string, unknown>[] = []
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(',')
    const row: Record<string, unknown> = {}
    header.forEach((key, j) => {
      const raw = (cells[j] ?? '').trim()
      const num = Number(raw)
      row[key] = raw !== '' && !Number.isNaN(num) ? num : raw
    })
    rows.push(row)
  }
  return rows
}

export const csvImportNode: NodeDef = {
  typeId: 'util_csv_import',
  label: 'CSV Import',
  category: 'Input',
  description:
    "Charge un fichier CSV (import) et le transforme en liste de lignes, une par objet. Sert à relire n'importe quel jeu de données précalculé — mesures, courbes, prédictions — pas seulement les tables produites par les autres nodes.",
  inputs: [],
  outputs: [
    { id: 'rows', label: 'rows', color: 'list' },
    { id: 'count', label: 'count', color: 'scalar' },
  ],
  params: [{ id: 'file', label: 'Fichier CSV', type: 'file', default: '' }],
  process(_inputs, params) {
    const text = params.__csvText as string | undefined
    if (!text) return { rows: [], count: 0 }
    const rows = parseCsv(text)
    return { rows, count: rows.length }
  },
}
