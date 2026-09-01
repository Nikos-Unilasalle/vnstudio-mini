import { useMemo } from 'react'
import { getNodeDef } from '../engine/registry'
import { useGraph } from './GraphContext'
import { matToDataUrl } from '../engine/imageIO'
import type { MeasuredRegion } from '../nodes/regionProps'

function isMat(v: unknown): boolean {
  return !!v && typeof v === 'object' && typeof (v as any).delete === 'function' && typeof (v as any).cols === 'number'
}

function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function PreviewPanel() {
  const { nodes, previewId, lastRun, cv } = useGraph()
  const node = nodes.find((n) => n.id === previewId)
  const def = node ? getNodeDef(node.data.typeId) : undefined
  const outputs = previewId ? lastRun?.outputsByNode[previewId] : undefined
  const error = previewId ? lastRun?.errors[previewId] : undefined

  const imageUrl = useMemo(() => {
    const mainOut = outputs?.main as any
    if (!cv || !mainOut) return null
    if (isMat(mainOut)) {
      try {
        return matToDataUrl(cv, mainOut)
      } catch {
        return null
      }
    }
    if (typeof mainOut === 'string' && mainOut.startsWith('data:image')) return mainOut
    return null
  }, [outputs, cv])

  if (!previewId || !node || !def) {
    return <div className="preview-panel preview-panel--empty">Sélectionne une node puis appuie sur Entrée pour voir son résultat.</div>
  }

  const regions = outputs?.regions as MeasuredRegion[] | undefined
  const text = (outputs as any)?.__text as string | undefined
  const csv = (outputs as any)?.__csv as string | undefined
  const filename = (outputs as any)?.__filename as string | undefined
  const stats = outputs?.stats as { d10: number; d50: number; d90: number; unit: string; count: number } | undefined
  const count = typeof outputs?.count === 'number' ? outputs.count : undefined

  const dict = useMemo(() => {
    if (!outputs) return null
    for (const v of Object.values(outputs)) {
      if (v && typeof v === 'object' && !Array.isArray(v) && !isMat(v) && !('labels' in (v as any)) && !('d50' in (v as any))) {
        return v as Record<string, unknown>
      }
    }
    return null
  }, [outputs])

  return (
    <div className="preview-panel">
      <div className="preview-panel__title">{node.data.title || def.label}</div>
      {error && <div className="preview-panel__error">Erreur: {error}</div>}
      {imageUrl && <img className="preview-panel__image" src={imageUrl} alt="" />}
      {text && <div className="preview-panel__text">{text}</div>}
      {stats && (
        <div className="preview-panel__stats">
          <div>D10: {stats.d10.toFixed(1)} {stats.unit}</div>
          <div>D50: {stats.d50.toFixed(1)} {stats.unit}</div>
          <div>D90: {stats.d90.toFixed(1)} {stats.unit}</div>
          <div>n = {stats.count}</div>
        </div>
      )}
      {count !== undefined && <div className="preview-panel__count">Objets: {count}</div>}
      {csv && (
        <button className="primary" onClick={() => downloadText(filename ?? 'export.csv', csv)}>
          Télécharger {filename ?? 'export.csv'}
        </button>
      )}
      {regions && regions.length > 0 && (
        <table className="preview-panel__table">
          <thead>
            <tr>
              <th>id</th>
              <th>aire (px)</th>
              <th>Ø équiv.</th>
            </tr>
          </thead>
          <tbody>
            {regions.slice(0, 200).map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{r.areaPx}</td>
                <td>{r.equivDiameterUm != null ? `${(r.equivDiameterUm / 1000).toFixed(1)} mm` : `${r.equivDiameterPx.toFixed(1)} px`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {dict && Object.keys(dict).length > 0 && (
        <table className="preview-panel__table">
          <tbody>
            {Object.entries(dict).map(([k, v]) => (
              <tr key={k}>
                <td>{k}</td>
                <td>{typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : String(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {!imageUrl && !text && !regions && !stats && !csv && !dict && count === undefined && !error && (
        <div className="preview-panel__empty">Pas de sortie (relance le graphe ?)</div>
      )}
    </div>
  )
}
