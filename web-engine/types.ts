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

/** A video/webcam frame the main thread grabbed before dispatching a run — see worker.ts. */
export interface CapturedFrame {
  bitmap: ImageBitmap
  /** Extra fields the node should emit alongside `main` (e.g. `frame`/`total_frames`, `fps`). */
  extra?: Record<string, unknown>
}

export interface RunContext {
  /** OpenCV.js module. */
  cv: any
  /** Per-node state that outlives a single run — detectors, accumulated series. */
  state: Map<string, any>
  /** Id of the node being processed, for keying into `state`. */
  nodeId: string
  /** Registers a Mat for cleanup after the next run. */
  track: (mat: any) => any
  /** Publishes a live field for this node, surfacing as `${nodeId}:${field}` in nodesData. */
  emit: (field: string, value: unknown) => void
  /**
   * Movie/webcam frames, keyed by node id. The worker has no DOM, so it can't own
   * `<video>` elements or call getUserMedia itself — the main thread captures these
   * and transfers the bitmaps in with each run request (see shims/useVisionEngine.ts).
   */
  frames?: Record<string, CapturedFrame>
}

export type NodeInputs = Record<string, unknown>
export type NodeOutputs = Record<string, unknown>

export type NodeImpl = (
  inputs: NodeInputs,
  params: Record<string, any>,
  ctx: RunContext
) => NodeOutputs | Promise<NodeOutputs>
