import { useCallback, useEffect, useRef, useState } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type Node,
  type ReactFlowInstance,
} from 'reactflow'
import 'reactflow/dist/style.css'
import './App.css'

import type { GraphNodeData } from './engine/types'
import { getNodeDef } from './engine/registry'
import { runGraph } from './engine/executor'
import { useOpenCv } from './engine/useOpenCv'
import { parseVnFile, serializeVnFile } from './engine/vnFile'
import { GraphContext, type GraphContextValue } from './ui/GraphContext'
import { VnNode } from './ui/VnNode'
import { NodePalette } from './ui/NodePalette'
import { ParamsPanel } from './ui/ParamsPanel'
import { PreviewPanel } from './ui/PreviewPanel'
import type { ExecResult } from './engine/executor'

const nodeTypes = { vnNode: VnNode }

let idCounter = 1
function nextId(typeId: string) {
  return `${typeId}-${idCounter++}`
}

function makeDefaultNode(typeId: string, position: { x: number; y: number }): Node<GraphNodeData> {
  const def = getNodeDef(typeId)!
  return {
    id: nextId(typeId),
    type: 'vnNode',
    position,
    data: {
      typeId,
      params: Object.fromEntries(def.params.map((p) => [p.id, p.default])),
    },
  }
}

export default function App() {
  const { cv, ready, error: cvError } = useOpenCv()
  const [nodes, setNodes, onNodesChange] = useNodesState<GraphNodeData>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [previewId, setPreviewId] = useState<string | null>(null)
  const [lastRun, setLastRun] = useState<ExecResult | null>(null)
  const [running, setRunning] = useState(false)
  const [warnings, setWarnings] = useState<string[]>([])
  const rfInstance = useRef<ReactFlowInstance | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const onConnect = useCallback((c: Connection) => setEdges((eds) => addEdge(c, eds)), [setEdges])

  const onSelectionChange = useCallback(({ nodes: sel }: { nodes: Node[] }) => {
    setSelectedId(sel[0]?.id ?? null)
  }, [])

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Enter' && selectedId) setPreviewId(selectedId)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selectedId])

  const updateNodeParams = useCallback(
    (id: string, patch: Record<string, unknown>) => {
      setNodes((nds) =>
        nds.map((n) => (n.id === id ? { ...n, data: { ...n.data, params: { ...n.data.params, ...patch } } } : n))
      )
    },
    [setNodes]
  )

  const run = useCallback(() => {
    if (!ready) return
    setRunning(true)
    runGraph(nodes, edges, { cv })
      .then(setLastRun)
      .finally(() => setRunning(false))
  }, [nodes, edges, cv, ready])

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    const typeId = e.dataTransfer.getData('application/vn-node-type')
    if (!typeId || !rfInstance.current) return
    const position = rfInstance.current.screenToFlowPosition({ x: e.clientX, y: e.clientY })
    setNodes((nds) => [...nds, makeDefaultNode(typeId, position)])
  }

  function loadVn(json: unknown) {
    const parsed = parseVnFile(json)
    setNodes(parsed.nodes)
    setEdges(parsed.edges)
    setWarnings(parsed.warnings)
    setLastRun(null)
    setSelectedId(null)
    setPreviewId(null)
  }

  function loadSampleTd1() {
    fetch(`${import.meta.env.BASE_URL}samples/M1.1_reference.vn`)
      .then((r) => r.json())
      .then(loadVn)
  }

  function onImportFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    file.text().then((txt) => loadVn(JSON.parse(txt)))
    e.target.value = ''
  }

  function exportVn() {
    const data = serializeVnFile(nodes, edges)
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'graphe.vn'
    a.click()
    URL.revokeObjectURL(url)
  }

  const ctxValue: GraphContextValue = {
    nodes,
    edges,
    selectedId,
    previewId,
    setSelectedId,
    setPreviewId,
    updateNodeParams,
    lastRun,
    cvReady: ready,
    cv,
    run,
    running,
  }

  return (
    <GraphContext.Provider value={ctxValue}>
      <div className="app">
        <header className="app__header">
          <span className="app__brand">vnstudio-mini</span>
          <button onClick={loadSampleTd1}>Charger TD I</button>
          <button onClick={() => fileInputRef.current?.click()}>Importer .vn</button>
          <input ref={fileInputRef} type="file" accept=".vn,application/json" hidden onChange={onImportFile} />
          <button onClick={exportVn}>Exporter .vn</button>
          <div className="app__spacer" />
          {!ready && !cvError && <span className="app__status">Chargement OpenCV.js…</span>}
          {cvError && <span className="app__status app__status--error">{cvError}</span>}
          <button className="primary" disabled={!ready || running} onClick={run}>
            {running ? 'Exécution…' : '▶ Lancer'}
          </button>
        </header>

        {warnings.length > 0 && (
          <div className="app__warnings">
            {warnings.map((w, i) => (
              <div key={i}>{w}</div>
            ))}
            <button onClick={() => setWarnings([])}>OK</button>
          </div>
        )}

        <div className="app__body">
          <NodePalette />
          <div className="app__canvas" onDrop={onDrop} onDragOver={(e) => e.preventDefault()}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onSelectionChange={onSelectionChange}
              onInit={(inst) => (rfInstance.current = inst)}
              nodeTypes={nodeTypes}
              fitView
            >
              <Background />
              <Controls />
              <MiniMap />
            </ReactFlow>
          </div>
          <div className="app__side">
            <ParamsPanel />
            <PreviewPanel />
          </div>
        </div>
      </div>
    </GraphContext.Provider>
  )
}
