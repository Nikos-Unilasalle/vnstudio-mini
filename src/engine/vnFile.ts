import type { Node, Edge } from 'reactflow'
import type { GraphNodeData } from './types'
import { getNodeDef } from './registry'

/**
 * The desktop VNStudio app names handles "{colorType}__{portId}". Ambiguous cases
 * (nodes with several ports of the same color) need an explicit legacy->local alias.
 */
const LEGACY_ALIASES: Record<string, { in?: Record<string, string>; out?: Record<string, string> }> = {
  sci_connected_components: {
    out: { markers__labels_map: 'regions', image__main: 'main' },
  },
}

function resolvePort(typeId: string, side: 'in' | 'out', rawHandle: string | null | undefined): string {
  const def = getNodeDef(typeId)
  const ports = side === 'in' ? def?.inputs : def?.outputs
  if (!ports || ports.length === 0) return 'main'
  if (ports.length === 1) return ports[0].id

  const alias = LEGACY_ALIASES[typeId]?.[side]
  if (rawHandle && alias?.[rawHandle]) return alias[rawHandle]

  const guess = rawHandle?.split('__').pop()
  if (guess && ports.some((p) => p.id === guess)) return guess

  // plotter_pro has desktop-side dynamic_inputs: every value port beyond "ticks"
  // gets an arbitrary user-typed name that can't be known ahead of time.
  if (typeId === 'plotter_pro' && side === 'in' && guess !== 'ticks') return 'value'

  return ports[0].id
}

interface RawVnNode {
  id: string
  type: string
  position?: { x: number; y: number }
  data?: { label?: string; params?: Record<string, unknown> }
}

interface RawVnEdge {
  id?: string
  source: string
  sourceHandle?: string | null
  target: string
  targetHandle?: string | null
}

export interface ParsedVn {
  nodes: Node<GraphNodeData>[]
  edges: Edge[]
  warnings: string[]
}

export function parseVnFile(json: unknown): ParsedVn {
  const raw = json as { nodes: RawVnNode[]; edges: RawVnEdge[] }
  const warnings: string[] = []
  const nodes: Node<GraphNodeData>[] = []

  raw.nodes.forEach((n, i) => {
    const def = getNodeDef(n.type)
    if (!def) {
      warnings.push(`Node type inconnu ignoré: "${n.type}" (${n.id})`)
      return
    }
    const params = { ...(n.data?.params ?? {}) }

    // input_image/input_movie legacy files reference an absolute filesystem path; remap to a bundled sample by filename.
    if ((n.type === 'input_image' || n.type === 'input_movie') && typeof params.path === 'string') {
      const filename = params.path.split('/').pop() ?? ''
      const known = def.params.find((p) => p.id === 'source')?.options?.find((o) => String(o.value).endsWith(filename))
      params.source = known ? known.value : '__upload__'
      if (!known) warnings.push(`Fichier "${filename}" non fourni: importe-le manuellement sur la node ${n.id}.`)
      delete params.path
    }

    nodes.push({
      id: n.id,
      type: 'vnNode',
      position: n.position ?? { x: (i % 5) * 260, y: Math.floor(i / 5) * 220 },
      data: {
        typeId: n.type,
        title: n.data?.label,
        params: { ...Object.fromEntries(def.params.map((p) => [p.id, p.default])), ...params },
        colorIndex: (params.color_index as number) ?? 0,
      },
    })
  })

  const nodeTypeById = new Map(raw.nodes.map((n) => [n.id, n.type]))
  const edges: Edge[] = raw.edges
    .filter((e) => nodeTypeById.has(e.source) && nodeTypeById.has(e.target))
    .map((e, i) => ({
      id: e.id ?? `e${i}`,
      source: e.source,
      target: e.target,
      sourceHandle: resolvePort(nodeTypeById.get(e.source)!, 'out', e.sourceHandle),
      targetHandle: resolvePort(nodeTypeById.get(e.target)!, 'in', e.targetHandle),
    }))

  return { nodes, edges, warnings }
}

export function serializeVnFile(nodes: Node<GraphNodeData>[], edges: Edge[]) {
  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.typeId,
      position: n.position,
      data: { label: n.data.title, params: n.data.params },
    })),
    edges: edges.map((e) => ({ id: e.id, source: e.source, sourceHandle: e.sourceHandle, target: e.target, targetHandle: e.targetHandle })),
  }
}
