import { CATEGORIES, NODE_DEFS } from '../engine/registry'

export function NodePalette() {
  return (
    <div className="palette">
      <div className="palette__heading">Nodes</div>
      {CATEGORIES.map((cat) => (
        <div key={cat} className="palette__group">
          <div className="palette__group-title">{cat}</div>
          {NODE_DEFS.filter((d) => d.category === cat).map((d) => (
            <div
              key={d.typeId}
              className="palette__item"
              draggable
              title={d.description}
              onDragStart={(e) => {
                e.dataTransfer.setData('application/vn-node-type', d.typeId)
                e.dataTransfer.effectAllowed = 'move'
              }}
            >
              {d.label}
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
