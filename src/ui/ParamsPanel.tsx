import { useMemo, useState } from 'react'
import { getNodeDef } from '../engine/registry'
import { useGraph } from './GraphContext'
import { InteractiveImageEditor } from './InteractiveImageEditor'
import { matToDataUrl } from '../engine/imageIO'

function parseNormPoints(raw: unknown): { x: number; y: number }[] {
  try {
    const arr = JSON.parse(String(raw ?? '[]'))
    return Array.isArray(arr) ? arr : []
  } catch {
    return []
  }
}

export function ParamsPanel() {
  const { nodes, edges, selectedId, updateNodeParams, lastRun, cv } = useGraph()
  const [editing, setEditing] = useState(false)

  const node = nodes.find((n) => n.id === selectedId)
  const def = node ? getNodeDef(node.data.typeId) : undefined

  const upstream = useMemo(() => {
    if (!node || !def || !cv) return null
    const inputPortId = def.inputs.find((p) => p.color === 'image')?.id
    if (!inputPortId) return null
    const edge = edges.find((e) => e.target === node.id && e.targetHandle === inputPortId)
    if (!edge || !lastRun) return null
    const mat = lastRun.outputsByNode[edge.source]?.[edge.sourceHandle ?? 'main'] as any
    if (!mat || typeof mat.delete !== 'function') return null
    try {
      return { url: matToDataUrl(cv, mat), w: mat.cols as number, h: mat.rows as number }
    } catch {
      return null
    }
  }, [node, def, edges, lastRun, cv])

  if (!node || !def) {
    return <div className="params-panel params-panel--empty">Sélectionne une node pour voir ses paramètres.</div>
  }

  function setParam(id: string, value: unknown) {
    updateNodeParams(node!.id, { [id]: value })
  }

  return (
    <div className="params-panel">
      <div className="params-panel__title">{node.data.title || def.label}</div>
      {def.description && <div className="params-panel__desc">{def.description}</div>}

      {def.params.map((p) => {
        const value = node.data.params[p.id] ?? p.default
        if (p.type === 'number') {
          return (
            <label key={p.id} className="params-panel__field">
              <span>{p.label}</span>
              <input
                type="number"
                value={value as number}
                min={p.min}
                max={p.max}
                step={p.step ?? 1}
                onChange={(e) => setParam(p.id, Number(e.target.value))}
              />
            </label>
          )
        }
        if (p.type === 'select') {
          return (
            <label key={p.id} className="params-panel__field">
              <span>{p.label}</span>
              <select value={String(value)} onChange={(e) => {
                const opt = p.options?.find((o) => String(o.value) === e.target.value)
                setParam(p.id, opt ? opt.value : e.target.value)
              }}>
                {p.options?.map((o) => (
                  <option key={String(o.value)} value={String(o.value)}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
          )
        }
        if (p.type === 'boolean') {
          return (
            <label key={p.id} className="params-panel__field params-panel__field--checkbox">
              <input type="checkbox" checked={!!value} onChange={(e) => setParam(p.id, e.target.checked)} />
              <span>{p.label}</span>
            </label>
          )
        }
        if (p.type === 'file') {
          const isCsv = def.typeId === 'util_csv_import'
          const isVideo = def.typeId === 'input_movie'
          return (
            <label key={p.id} className="params-panel__field">
              <span>{p.label}</span>
              <input
                type="file"
                accept={isCsv ? '.csv,text/csv' : isVideo ? 'video/*' : 'image/*'}
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (!file) return
                  if (isCsv) {
                    file.text().then((text) => {
                      setParam(p.id, file.name)
                      setParam('__csvText', text)
                    })
                    return
                  }
                  const reader = new FileReader()
                  reader.onload = () => {
                    setParam(p.id, reader.result as string)
                    setParam('source', '__upload__')
                  }
                  reader.readAsDataURL(file)
                }}
              />
            </label>
          )
        }
        if (p.id === 'code') {
          return (
            <label key={p.id} className="params-panel__field">
              <span>{p.label}</span>
              <textarea
                rows={8}
                value={String(value ?? '')}
                onChange={(e) => setParam(p.id, e.target.value)}
              />
            </label>
          )
        }
        // string
        return (
          <label key={p.id} className="params-panel__field">
            <span>{p.label}</span>
            <input type="text" value={String(value ?? '')} onChange={(e) => setParam(p.id, e.target.value)} />
          </label>
        )
      })}

      {(def.typeId === 'util_roi_polygon' || def.typeId === 'sci_interactive_calibration') && (
        <div className="params-panel__interactive">
          <button disabled={!upstream} onClick={() => setEditing(true)}>
            {upstream ? (def.typeId === 'util_roi_polygon' ? 'Éditer le polygone' : 'Tracer la ligne de calibration') : "Lance le graphe pour voir l'image"}
          </button>
          {editing && upstream && (
            <InteractiveImageEditor
              imageDataUrl={upstream.url}
              mode={def.typeId === 'util_roi_polygon' ? 'polygon' : 'line'}
              initialPoints={parseNormPoints(node.data.params.points).map((p) => ({ x: p.x * upstream.w, y: p.y * upstream.h }))}
              onSave={(pts) => setParam('points', JSON.stringify(pts.map((p) => ({ x: p.x / upstream.w, y: p.y / upstream.h }))))}
              onClose={() => setEditing(false)}
            />
          )}
        </div>
      )}
    </div>
  )
}
