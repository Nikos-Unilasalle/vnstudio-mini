import type { RunContext } from './types'
import { getSchema, IMPLEMENTATIONS } from './registry'
import { isMat, matToBase64 } from './cvUtils'

export interface GraphNode {
  id: string
  type: string
  data?: { params?: Record<string, any> }
}

export interface GraphEdge {
  source: string
  sourceHandle?: string | null
  target: string
  targetHandle?: string | null
}

export interface RunResult {
  /** Flat `${nodeId}:${field}` map, matching what the desktop engine publishes. */
  nodesData: Record<string, unknown>
  /** Base64 JPEG of the preview node's image output, or null. */
  frame: string | null
  errors: Record<string, string>
}

/** Nodes that exist purely on the canvas and never execute. */
const NON_EXECUTING = new Set(['canvas_frame', 'canvas_ink', 'canvas_reroute', 'canvas_ribbon', 'canvas_teleport'])

/**
 * Handles are serialised as "{portColor}__{portId}" by the desktop UI. The port
 * id is the part that matters; the colour prefix only drives edge styling.
 */
function portIdOf(handle: string | null | undefined, fallback: string): string {
  if (!handle) return fallback
  const index = handle.indexOf('__')
  return index >= 0 ? handle.slice(index + 2) : handle
}

function topologicalOrder(nodes: GraphNode[], edges: GraphEdge[]): string[] {
  const executable = nodes.filter((n) => !NON_EXECUTING.has(n.type))
  const ids = new Set(executable.map((n) => n.id))

  const indegree = new Map<string, number>()
  const outgoing = new Map<string, string[]>()
  for (const id of ids) {
    indegree.set(id, 0)
    outgoing.set(id, [])
  }
  for (const edge of edges) {
    if (!ids.has(edge.source) || !ids.has(edge.target) || edge.source === edge.target) continue
    outgoing.get(edge.source)!.push(edge.target)
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1)
  }

  const queue = [...ids].filter((id) => indegree.get(id) === 0)
  const order: string[] = []
  while (queue.length > 0) {
    const id = queue.shift()!
    order.push(id)
    for (const next of outgoing.get(id) ?? []) {
      const remaining = (indegree.get(next) ?? 0) - 1
      indegree.set(next, remaining)
      if (remaining === 0) queue.push(next)
    }
  }

  // A cycle leaves nodes unvisited; run them anyway so the user sees partial
  // results rather than a silently dead branch.
  for (const id of ids) if (!order.includes(id)) order.push(id)
  return order
}

export class GraphExecutor {
  private readonly cv: any
  /** Survives across runs: video elements, MediaPipe detectors, plot history. */
  private readonly nodeState = new Map<string, any>()
  private matPool: any[] = []

  constructor(cv: any) {
    this.cv = cv
  }

  /** Frees the Mats allocated by the previous run — WASM heap is not garbage collected. */
  private releaseMats(): void {
    for (const mat of this.matPool) {
      try {
        if (!mat.isDeleted?.()) mat.delete()
      } catch {
        // Already released.
      }
    }
    this.matPool = []
  }

  dispose(): void {
    this.releaseMats()
    for (const value of this.nodeState.values()) {
      if (value?.stream) for (const track of value.stream.getTracks()) track.stop()
    }
    this.nodeState.clear()
  }

  /** Drops state for nodes that no longer exist, so deleting a node also stops its webcam. */
  pruneState(liveIds: Set<string>): void {
    for (const key of [...this.nodeState.keys()]) {
      const nodeId = key.includes(':') ? key.slice(0, key.indexOf(':')) : key
      if (liveIds.has(nodeId)) continue
      const value = this.nodeState.get(key)
      if (value?.stream) for (const track of value.stream.getTracks()) track.stop()
      this.nodeState.delete(key)
    }
  }

  async run(nodes: GraphNode[], edges: GraphEdge[], previewNodeId: string | null): Promise<RunResult> {
    this.releaseMats()
    this.pruneState(new Set(nodes.map((n) => n.id)))

    const nodeById = new Map(nodes.map((n) => [n.id, n]))
    const outputsByNode = new Map<string, Record<string, unknown>>()
    const nodesData: Record<string, unknown> = {}
    const errors: Record<string, string> = {}

    for (const nodeId of topologicalOrder(nodes, edges)) {
      const node = nodeById.get(nodeId)!
      const implementation = IMPLEMENTATIONS[node.type]
      if (!implementation) continue

      const schema = getSchema(node.type)
      const defaultInput = schema?.inputs[0]?.id ?? 'main'
      const defaultOutput = 'main'

      const inputs: Record<string, unknown> = {}
      for (const edge of edges) {
        if (edge.target !== nodeId) continue
        const upstream = outputsByNode.get(edge.source)
        if (!upstream) continue
        inputs[portIdOf(edge.targetHandle, defaultInput)] = upstream[portIdOf(edge.sourceHandle, defaultOutput)]
      }

      const params = { ...(node.data?.params ?? {}) }
      // Params the user never touched arrive absent; fill from the schema so
      // implementations can trust that every documented key is present.
      for (const spec of schema?.params ?? []) {
        if (params[spec.id] === undefined && spec.default !== undefined) params[spec.id] = spec.default
      }

      const context: RunContext = {
        cv: this.cv,
        state: this.nodeState,
        nodeId,
        track: (mat: any) => {
          if (mat && typeof mat.delete === 'function') this.matPool.push(mat)
          return mat
        },
        emit: (field: string, value: unknown) => {
          nodesData[`${nodeId}:${field}`] = value
        },
      }

      try {
        const outputs = (await implementation(inputs, params, context)) ?? {}
        outputsByNode.set(nodeId, outputs)

        const main = outputs.main
        if (isMat(main)) {
          nodesData[`${nodeId}:main_preview`] = matToBase64(this.cv, main)
        }
        for (const [port, value] of Object.entries(outputs)) {
          if (typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') {
            nodesData[`${nodeId}:${port}`] = value
          }
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        errors[nodeId] = message
        nodesData[`${nodeId}:error`] = message
      }
    }

    let frame: string | null = null
    if (previewNodeId) {
      const outputs = outputsByNode.get(previewNodeId)
      const image = outputs && (isMat(outputs.main) ? outputs.main : Object.values(outputs).find(isMat))
      if (image) frame = matToBase64(this.cv, image, 1280, 0.85)
      else {
        const preview = nodesData[`${previewNodeId}:main_preview`]
        if (typeof preview === 'string') frame = preview
      }
    }

    return { nodesData, frame, errors }
  }
}
