/** Shape of the schema entries baked into schemas.json (a subset of the desktop registry). */
export interface WebNodeSchema {
  type: string
  label: string
  category: string | string[]
  icon: string
  description?: string
  inputs: { id: string; color: string; label?: string }[]
  outputs: { id: string; color: string; label?: string }[]
  params: {
    id: string
    label?: string
    type?: string
    default?: unknown
    min?: number
    max?: number
    step?: number
    options?: string[]
  }[]
  resizable?: boolean
  colorable?: boolean
}

export interface RunContext {
  /** OpenCV.js module. */
  cv: any
  /** Per-node state that outlives a single run — video elements, detectors, accumulated series. */
  state: Map<string, any>
  /** Id of the node being processed, for keying into `state`. */
  nodeId: string
  /** Registers a Mat for cleanup after the next run. */
  track: (mat: any) => any
  /** Publishes a live field for this node, surfacing as `${nodeId}:${field}` in nodesData. */
  emit: (field: string, value: unknown) => void
}

export type NodeInputs = Record<string, unknown>
export type NodeOutputs = Record<string, unknown>

export type NodeImpl = (
  inputs: NodeInputs,
  params: Record<string, any>,
  ctx: RunContext
) => NodeOutputs | Promise<NodeOutputs>
