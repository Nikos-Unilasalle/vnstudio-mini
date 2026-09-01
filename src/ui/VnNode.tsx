import { memo } from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'
import type { GraphNodeData } from '../engine/types'
import { getNodeDef } from '../engine/registry'
import { PORT_COLORS } from './portColors'
import { useGraph } from './GraphContext'

function VnNodeImpl({ id, data, selected }: NodeProps<GraphNodeData>) {
  const def = getNodeDef(data.typeId)
  const { lastRun, previewId } = useGraph()
  const error = lastRun?.errors[id]
  const hasRun = lastRun?.outputsByNode[id] !== undefined

  if (!def) {
    return <div className="vn-node vn-node--error">Type inconnu: {data.typeId}</div>
  }

  if (def.decorative) {
    if (data.typeId === 'canvas_note') {
      return (
        <div className={`vn-note vn-color-${data.colorIndex ?? 0}`}>
          <div className="vn-note__text">{(data.params.text as string) || 'Note vide'}</div>
        </div>
      )
    }
    return (
      <div className={`vn-frame vn-color-${data.colorIndex ?? 0}`}>
        <div className="vn-frame__title">{(data.params.title as string) || 'Groupe'}</div>
      </div>
    )
  }

  return (
    <div className={`vn-node ${selected ? 'vn-node--selected' : ''} ${previewId === id ? 'vn-node--preview' : ''}`}>
      <div className="vn-node__header">
        <span className="vn-node__title">{data.title || def.label}</span>
        {error && <span className="vn-node__badge vn-node__badge--error" title={error}>!</span>}
        {!error && hasRun && <span className="vn-node__badge vn-node__badge--ok">✓</span>}
      </div>
      <div className="vn-node__body">
        <div className="vn-node__ports vn-node__ports--in">
          {def.inputs.map((p, i) => (
            <div key={p.id} className="vn-port">
              <Handle
                type="target"
                position={Position.Left}
                id={p.id}
                style={{ top: 34 + i * 18, background: PORT_COLORS[p.color] }}
              />
              <span className="vn-port__label">{p.label ?? p.id}</span>
            </div>
          ))}
        </div>
        <div className="vn-node__ports vn-node__ports--out">
          {def.outputs.map((p, i) => (
            <div key={p.id} className="vn-port vn-port--out">
              <span className="vn-port__label">{p.label ?? p.id}</span>
              <Handle
                type="source"
                position={Position.Right}
                id={p.id}
                style={{ top: 34 + i * 18, background: PORT_COLORS[p.color] }}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export const VnNode = memo(VnNodeImpl)
