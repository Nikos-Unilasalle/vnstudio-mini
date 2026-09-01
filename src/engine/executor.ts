import type { Node, Edge } from 'reactflow'
import type { GraphNodeData, NodeOutputs, RuntimeCtx } from './types'
import { getNodeDef } from './registry'

export interface ExecResult {
  outputsByNode: Record<string, NodeOutputs>
  errors: Record<string, string>
  order: string[]
}

/** Mats created during a run are tracked here so the next run can free them. */
const matPool: any[] = []

export function trackMat(m: any) {
  if (m && typeof m.delete === 'function') matPool.push(m)
  return m
}

function freeMatPool() {
  for (const m of matPool) {
    try {
      if (!m.isDeleted?.()) m.delete()
    } catch {
      /* already freed */
    }
  }
  matPool.length = 0
}

function topoSort(nodes: Node<GraphNodeData>[], edges: Edge[]): string[] {
  const execNodes = nodes.filter((n) => !getNodeDef(n.data.typeId)?.decorative)
  const ids = new Set(execNodes.map((n) => n.id))
  const inDeg = new Map<string, number>()
  const adj = new Map<string, string[]>()
  for (const id of ids) {
    inDeg.set(id, 0)
    adj.set(id, [])
  }
  for (const e of edges) {
    if (!ids.has(e.source) || !ids.has(e.target)) continue
    adj.get(e.source)!.push(e.target)
    inDeg.set(e.target, (inDeg.get(e.target) ?? 0) + 1)
  }
  const queue: string[] = [...ids].filter((id) => inDeg.get(id) === 0)
  const order: string[] = []
  while (queue.length) {
    const id = queue.shift()!
    order.push(id)
    for (const next of adj.get(id) ?? []) {
      inDeg.set(next, (inDeg.get(next) ?? 0) - 1)
      if (inDeg.get(next) === 0) queue.push(next)
    }
  }
  return order
}

export async function runGraph(
  nodes: Node<GraphNodeData>[],
  edges: Edge[],
  ctx: RuntimeCtx
): Promise<ExecResult> {
  freeMatPool()

  const order = topoSort(nodes, edges)
  const nodeById = new Map(nodes.map((n) => [n.id, n]))
  const outputsByNode: Record<string, NodeOutputs> = {}
  const errors: Record<string, string> = {}

  for (const id of order) {
    const node = nodeById.get(id)!
    const def = getNodeDef(node.data.typeId)
    if (!def?.process) continue

    const inputs: Record<string, unknown> = {}
    for (const e of edges) {
      if (e.target !== id) continue
      const sourceOut = outputsByNode[e.source]
      if (!sourceOut) continue
      const handleOut = e.sourceHandle ?? 'main'
      const handleIn = e.targetHandle ?? 'main'
      inputs[handleIn] = sourceOut[handleOut]
    }

    try {
      const result = await def.process(inputs, node.data.params, { ...ctx, nodeId: id })
      outputsByNode[id] = result
    } catch (err) {
      errors[id] = err instanceof Error ? err.message : String(err)
    }
  }

  return { outputsByNode, errors, order }
}
