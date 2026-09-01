import { createContext, useContext } from 'react'
import type { Node, Edge } from 'reactflow'
import type { GraphNodeData } from '../engine/types'
import type { ExecResult } from '../engine/executor'

export interface GraphContextValue {
  nodes: Node<GraphNodeData>[]
  edges: Edge[]
  selectedId: string | null
  previewId: string | null
  setSelectedId: (id: string | null) => void
  setPreviewId: (id: string | null) => void
  updateNodeParams: (id: string, patch: Record<string, unknown>) => void
  lastRun: ExecResult | null
  cvReady: boolean
  cv: any
  run: () => void
  running: boolean
}

export const GraphContext = createContext<GraphContextValue | null>(null)

export function useGraph(): GraphContextValue {
  const ctx = useContext(GraphContext)
  if (!ctx) throw new Error('useGraph must be used within GraphContext.Provider')
  return ctx
}
