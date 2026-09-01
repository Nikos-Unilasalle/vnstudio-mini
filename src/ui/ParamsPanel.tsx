import { useMemo, useState } from 'react'
import { getNodeDef } from '../engine/registry'
import { useGraph } from './GraphContext'
import { InteractiveImageEditor } from './InteractiveImageEditor'
import { matToDataUrl } from '../engine/imageIO'

export function ParamsPanel() {
  const { nodes, edges, selectedId, updateNodeParams, lastRun, cv } = useGraph()
  const [editing, setEditing] = useState(false)

  const node = nodes.find((n) => n.id === selectedId)
  const def = node ? getNodeDef(node.data.typeId) : undefined

  const upstreamImageUrl = useMemo(() => {
    if (!node || !def || !cv) return null
    const inputPortId = def.inputs.find((p) => p.color === 'image')?.id
    if (!inputPortId) return null
    const edge = edges.find((e) => e.target === node.id && e.targetHandle === inputPortId)
    if (!edge || !lastRun) return null
    const mat = lastRun.outputsByNode[edge.source]?.[edge.sourceHandle ?? 'main'] as any
    if (!mat || typeof mat.delete !== 'function') return null
    try {
      return matToDataUrl(cv, mat)
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
          return (
            <label key={p.id} className="params-panel__field">
              <span>{p.label}</span>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (!file) return
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
        // string
        return (
          <label key={p.id} className="params-panel__field">
            <span>{p.label}</span>
            <input type="text" value={String(value ?? '')} onChange={(e) => setParam(p.id, e.target.value)} />
          </label>
        )
      })}

      {def.typeId === 'geo_mask_polygon' && (
        <div className="params-panel__interactive">
          <button disabled={!upstreamImageUrl} onClick={() => setEditing(true)}>
            {upstreamImageUrl ? 'Éditer le polygone' : 'Lance le graphe pour voir l\'image'}
          </button>
          {editing && upstreamImageUrl && (
            <InteractiveImageEditor
              imageDataUrl={upstreamImageUrl}
              mode="polygon"
              initialPoints={(node.data.params.__polygon as any) ?? []}
              onSave={(pts) => setParam('__polygon', pts)}
              onClose={() => setEditing(false)}
            />
          )}
        </div>
      )}

      {def.typeId === 'geo_visual_calibration' && (
        <div className="params-panel__interactive">
          <button disabled={!upstreamImageUrl} onClick={() => setEditing(true)}>
            {upstreamImageUrl ? 'Tracer la ligne de calibration' : 'Lance le graphe pour voir l\'image'}
          </button>
          {editing && upstreamImageUrl && (
            <InteractiveImageEditor
              imageDataUrl={upstreamImageUrl}
              mode="line"
              initialPoints={
                node.data.params.__line
                  ? [
                      { x: (node.data.params.__line as any).x1, y: (node.data.params.__line as any).y1 },
                      { x: (node.data.params.__line as any).x2, y: (node.data.params.__line as any).y2 },
                    ]
                  : []
              }
              onSave={(pts) => setParam('__line', { x1: pts[0].x, y1: pts[0].y, x2: pts[1].x, y2: pts[1].y })}
              onClose={() => setEditing(false)}
            />
          )}
        </div>
      )}
    </div>
  )
}
