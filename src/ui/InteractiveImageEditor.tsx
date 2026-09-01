import { useRef, useState } from 'react'

interface Point {
  x: number
  y: number
}

interface Props {
  imageDataUrl: string
  mode: 'polygon' | 'line'
  initialPoints: Point[]
  onSave: (points: Point[]) => void
  onClose: () => void
}

const DISPLAY_MAX_W = 420

export function InteractiveImageEditor({ imageDataUrl, mode, initialPoints, onSave, onClose }: Props) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [points, setPoints] = useState<Point[]>(initialPoints)
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null)

  const scale = naturalSize ? Math.min(1, DISPLAY_MAX_W / naturalSize.w) : 1
  const displayW = naturalSize ? naturalSize.w * scale : DISPLAY_MAX_W
  const displayH = naturalSize ? naturalSize.h * scale : 300

  function handleClick(e: React.MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = (e.clientX - rect.left) / scale
    const y = (e.clientY - rect.top) / scale
    if (mode === 'line') {
      setPoints((prev) => (prev.length >= 2 ? [{ x, y }] : [...prev, { x, y }]))
    } else {
      setPoints((prev) => [...prev, { x, y }])
    }
  }

  const svgPoints = points.map((p) => `${p.x * scale},${p.y * scale}`).join(' ')

  return (
    <div className="interactive-editor">
      <div
        className="interactive-editor__canvas"
        style={{ width: displayW, height: displayH }}
        onClick={handleClick}
      >
        <img
          ref={imgRef}
          src={imageDataUrl}
          alt=""
          style={{ width: displayW, height: displayH }}
          onLoad={(e) => setNaturalSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
          draggable={false}
        />
        <svg className="interactive-editor__overlay" width={displayW} height={displayH}>
          {mode === 'polygon' && points.length >= 2 && (
            <polygon points={svgPoints} fill="rgba(79,140,255,0.25)" stroke="#4f8cff" strokeWidth={2} />
          )}
          {mode === 'line' && points.length === 2 && (
            <line x1={points[0].x * scale} y1={points[0].y * scale} x2={points[1].x * scale} y2={points[1].y * scale} stroke="#ff5c5c" strokeWidth={2} />
          )}
          {points.map((p, i) => (
            <circle key={i} cx={p.x * scale} cy={p.y * scale} r={4} fill="#fff" stroke="#000" strokeWidth={1} />
          ))}
        </svg>
      </div>
      <div className="interactive-editor__hint">
        {mode === 'polygon' ? 'Clique pour poser les sommets du polygone.' : 'Clique deux points pour tracer la ligne de calibration.'}
      </div>
      <div className="interactive-editor__actions">
        <button onClick={() => setPoints([])}>Effacer</button>
        <button
          className="primary"
          disabled={mode === 'polygon' ? points.length < 3 : points.length !== 2}
          onClick={() => {
            onSave(points)
            onClose()
          }}
        >
          Valider
        </button>
        <button onClick={onClose}>Annuler</button>
      </div>
    </div>
  )
}
