export type PortColor =
  | 'image'
  | 'mask'
  | 'scalar'
  | 'dict'
  | 'regions'
  | 'points'
  | 'string'

export interface PortSpec {
  id: string
  label?: string
  color: PortColor
}

export type ParamType = 'number' | 'select' | 'boolean' | 'string' | 'file' | 'polygon' | 'calibration-line'

export interface ParamOption {
  label: string
  value: number | string
}

export interface ParamSpec {
  id: string
  label: string
  type: ParamType
  default: unknown
  min?: number
  max?: number
  step?: number
  options?: ParamOption[]
}

export interface RuntimeCtx {
  /** cv is the loaded OpenCV.js module (window.cv) */
  cv: any
}

export type NodeInputs = Record<string, unknown>
export type NodeOutputs = Record<string, unknown>

export interface NodeDef {
  typeId: string
  label: string
  category: string
  description?: string
  inputs: PortSpec[]
  outputs: PortSpec[]
  params: ParamSpec[]
  /** decorative nodes (canvas_frame, canvas_note) skip execution */
  decorative?: boolean
  /** requires user interaction (polygon draw, calibration line) before it produces real output */
  interactive?: boolean
  process?: (inputs: NodeInputs, params: Record<string, unknown>, ctx: RuntimeCtx) => NodeOutputs | Promise<NodeOutputs>
}

export interface GraphNodeData {
  typeId: string
  title?: string
  params: Record<string, unknown>
  colorIndex?: number
  /** interactive state not persisted as a "param" but edited via canvas (polygon points, calibration line) */
  interactiveState?: Record<string, unknown>
}
