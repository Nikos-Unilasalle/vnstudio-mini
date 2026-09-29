/**
 * Dict / list nodes ported from the desktop plugins dict_scale.py,
 * dict_report_panel.py and list_natural_break.py.
 */
import type { NodeImpl } from '../types'

// ── Scale Dict Values ────────────────────────────────────────────────────────

export const dictScale: NodeImpl = (inputs, params) => {
  const d = inputs.dict
  if (!d || typeof d !== 'object' || Array.isArray(d)) return { dict: {} }
  const factor = Number(params.factor ?? 1)
  const scale = factor ** Number(params.power ?? 2)
  if (!scale || !Number.isFinite(scale)) return { dict: { ...(d as object) } }
  const divide = Number(params.operation ?? 1) === 1
  const skip = new Set(String(params.skip_keys ?? '').split(',').map((k) => k.trim()).filter(Boolean))
  const digits = Math.round(Number(params.digits ?? -1))

  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(d as Record<string, unknown>)) {
    if (skip.has(k) || typeof v !== 'number') {
      out[k] = v
      continue
    }
    const val = divide ? v / scale : v * scale
    out[k] = digits >= 0 ? Number(val.toFixed(digits)) : val
  }
  return { dict: out }
}

// ── Report Panel ─────────────────────────────────────────────────────────────

interface ReportGroup { label: string; keys: string[] }

function parseGroups(text: unknown): ReportGroup[] {
  const groups: ReportGroup[] = []
  for (const line of String(text ?? '').replace(/\|/g, '\n').split('\n')) {
    const cut = line.indexOf(':')
    if (cut < 0) continue
    const label = line.slice(0, cut).trim()
    const keys = line.slice(cut + 1).split(',').map((k) => k.trim()).filter(Boolean)
    if (label && keys.length) groups.push({ label, keys })
  }
  return groups
}

export const dictReportPanel: NodeImpl = (inputs, params) => {
  const merged: Record<string, unknown> = {}
  for (const key of Object.keys(inputs).sort()) {
    const v = inputs[key]
    if (key.startsWith('dict') && v && typeof v === 'object' && !Array.isArray(v)) Object.assign(merged, v)
  }
  const parsed = parseGroups(params.groups ?? 'Values: *')
  const listed = new Set(parsed.flatMap((g) => g.keys).filter((k) => k !== '*'))
  const rest = Object.keys(merged).filter((k) => !listed.has(k))
  const groups = parsed.map((g) => ({ label: g.label, keys: g.keys.flatMap((k) => (k === '*' ? rest : [k])) }))

  const ordered: Record<string, unknown> = {}
  for (const g of groups) for (const k of g.keys) if (k in merged) ordered[k] = merged[k]
  for (const [k, v] of Object.entries(merged)) if (!(k in ordered)) ordered[k] = v
  return { report: ordered, groups, title: String(params.title ?? 'Report'), digits: Math.round(Number(params.digits ?? 4)) }
}

// ── List Natural Break ───────────────────────────────────────────────────────

/** Index k maximising the between-class variance for sorted[:k] | sorted[k:] (1-D Otsu). */
function otsuSplit(sorted: number[]): number {
  const n = sorted.length
  const total = sorted.reduce((a, b) => a + b, 0)
  let cum = 0
  let best = 1
  let bestVar = -Infinity
  for (let k = 1; k < n; k++) {
    cum += sorted[k - 1]
    const w0 = k / n
    const w1 = 1 - w0
    const m0 = cum / k
    const m1 = (total - cum) / (n - k)
    const between = w0 * w1 * (m0 - m1) ** 2
    if (between > bestVar) {
      bestVar = between
      best = k
    }
  }
  return best
}

export const listNaturalBreak: NodeImpl = (inputs, params) => {
  const raw = Array.isArray(inputs.values) ? (inputs.values as unknown[]) : []
  const useLog = params.log_scale !== false
  const floor = Number(params.min_value ?? 0)
  let vals = raw.map(Number).filter((v) => Number.isFinite(v) && v >= floor)
  if (useLog) vals = vals.filter((v) => v > 0)
  if (vals.length === 0) return { threshold: null, count_above: 0, count_below: 0, above: [], below: [] }

  const ordered = [...vals].sort((a, b) => a - b)
  let threshold: number
  if (ordered.length === 1 || ordered[0] === ordered[ordered.length - 1]) {
    threshold = ordered[0]
  } else {
    const space = useLog ? ordered.map(Math.log) : ordered
    const k = otsuSplit(space)
    const mid = (space[k - 1] + space[k]) / 2
    threshold = useLog ? Math.exp(mid) : mid
  }
  const above = ordered.filter((v) => v >= threshold)
  const below = ordered.filter((v) => v < threshold)
  return { threshold, count_above: above.length, count_below: below.length, above, below }
}
