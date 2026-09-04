import type { NodeImpl } from '../types'
import { downloadFile } from '../../shims/vfs'
import {
  aggregate,
  cellText,
  column,
  compareValues,
  DataFrame,
  dfMeta,
  emptyDf,
  isDf,
  isNa,
  isNumericColumn,
  makeDf,
  metaAndPreview,
  parseRenameMap,
  previewSize,
  renderDfTable,
  resolveColumn,
  seededRandom,
  splitList,
  toCsv,
} from '../dataframe'

/** Every table node takes its input on `table`; a couple of components spell it `main`. */
function tableIn(inputs: Record<string, unknown>): DataFrame | null {
  const v = inputs.table ?? inputs.data ?? inputs.main
  return isDf(v) ? v : null
}

/** Row-wise copy — enough isolation, since node code only ever replaces whole cells. */
function copyRows(df: DataFrame): Record<string, unknown>[] {
  return df.rows.map((r) => ({ ...r }))
}

/* ------------------------------------------------------------------ builders */

/** Flattens a list/array input into a column of plain values. */
function toColumn(v: unknown): unknown[] | null {
  if (!Array.isArray(v)) return null
  return v.map((x) => (Array.isArray(x) ? (x.length ? x[0] : null) : x))
}

// Ports carrying engine plumbing rather than a series worth plotting.
const RESERVED_PORTS = new Set(['raw_frame', 'img_size'])

export const dfFromList: NodeImpl = (inputs, params) => {
  const raw = new Map<string, unknown[]>()
  for (const [port, value] of Object.entries(inputs)) {
    if (RESERVED_PORTS.has(port)) continue
    const col = toColumn(value)
    if (col && col.length) raw.set(port, col)
  }
  if (raw.size === 0) return {}

  const requested = splitList(params.name)
  // Dynamic ports carry a random suffix, so sort them for a stable column order;
  // the base `list` port always comes first.
  const order = [...(raw.has('list') ? ['list'] : []), ...[...raw.keys()].filter((k) => k !== 'list').sort()]

  const named = new Map<string, unknown[]>()
  order.forEach((port, i) => {
    let name: string
    if (i < requested.length) name = requested[i]
    else if (port !== 'list' && /^[A-Za-z_$][\w$]*$/.test(port)) name = port
    else name = `serie_${i + 1}`
    while (named.has(name)) name += '_'
    named.set(name, raw.get(port)!)
  })

  const length = Math.max(...[...named.values()].map((v) => v.length))

  const head = new Map<string, unknown[]>()
  if (params.add_index !== false) head.set('image', Array.from({ length }, (_, i) => i))
  const fps = Math.round(Number(params.fps) || 0)
  if (fps > 0) head.set('temps', Array.from({ length }, (_, i) => i / fps))

  const all = new Map([...head, ...named])
  const columns = [...all.keys()]
  const rows: Record<string, unknown>[] = []
  for (let i = 0; i < length; i++) {
    const row: Record<string, unknown> = {}
    // Shorter lists are padded rather than dropped, so a wiring mistake shows up.
    for (const [name, values] of all) row[name] = i < values.length ? values[i] : null
    rows.push(row)
  }

  const df = makeDf(columns, rows)
  return { table: df, df_meta: dfMeta(df), rows: length }
}

export const dfToDataframe: NodeImpl = (inputs, params) => {
  const dict = inputs.dict_in as Record<string, unknown> | null | undefined
  if (!dict || typeof dict !== 'object' || Array.isArray(dict) || Object.keys(dict).length === 0) return {}

  if (Math.round(Number(params.orient) || 0) === 1) {
    const df = makeDf(
      ['metric', 'value'],
      Object.entries(dict).map(([k, v]) => ({ metric: String(k), value: v as unknown }))
    )
    return { data: df, df_meta: dfMeta(df) }
  }

  const lengths = Object.values(dict).filter(Array.isArray).map((v) => (v as unknown[]).length)
  const n = lengths.length ? Math.max(...lengths) : 1
  const columns = Object.keys(dict).map(String)
  const rows: Record<string, unknown>[] = []
  for (let i = 0; i < n; i++) {
    const row: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(dict)) {
      if (Array.isArray(v)) {
        // A ragged list stays whole in the first cell so nothing is silently dropped.
        row[String(k)] = v.length === n ? v[i] : i === 0 ? v : null
      } else {
        row[String(k)] = v
      }
    }
    rows.push(row)
  }
  const df = makeDf(columns, rows)
  return { data: df, df_meta: dfMeta(df) }
}

/* --------------------------------------------------------------- extraction */

export const dfColumnList: NodeImpl = (inputs, params) => {
  const df = tableIn(inputs)
  if (!df) return {}

  const meta = { shape: [df.rows.length, df.columns.length], columns: [...df.columns], dtypes: dfMeta(df).dtypes }
  if (df.columns.length === 0) return { df_meta: meta }

  const requested = String(params.column ?? '').trim()
  // Falling back to the first column keeps the node useful before it is configured.
  const col = resolveColumn(df, requested) ?? (requested ? null : df.columns[0])
  if (col === null) return { df_meta: meta }

  let values = column(df, col)
  if (params.numeric) {
    values = values.map((v) => {
      if (typeof v === 'number') return v
      const n = Number.parseFloat(String(v))
      return Number.isNaN(n) ? null : n
    })
  }
  if (params.dropna !== false) values = values.filter((v) => !isNa(v))

  return {
    list: values,
    table: makeDf([col], values.map((v) => ({ [col]: v }))),
    count: values.length,
    df_meta: meta,
  }
}

/* ------------------------------------------------------------- manipulation */

export const dfSelect: NodeImpl = (inputs, params, ctx) => {
  const df = tableIn(inputs)
  if (!df) return {}

  const keep = splitList(params.keep)
  const drop = splitList(params.drop)
  const rename = parseRenameMap(params.rename)

  let columns = keep.length ? keep.filter((c) => df.columns.includes(c)) : [...df.columns]
  if (drop.length) columns = columns.filter((c) => !drop.includes(c))

  const finalNames = columns.map((c) => rename.get(c) ?? c)
  const rows = df.rows.map((row) => {
    const out: Record<string, unknown> = {}
    columns.forEach((c, i) => {
      out[finalNames[i]] = row[c]
    })
    return out
  })

  const out = makeDf(finalNames, rows)
  const [w, h] = previewSize(inputs.img_size, params)
  return {
    table: out,
    row_count: out.rows.length,
    col_count: out.columns.length,
    img_size: [w, h],
    ...metaAndPreview(ctx.cv, out, w, h, 'Select', ctx.track),
  }
}

export const dfSort: NodeImpl = (inputs, params, ctx) => {
  const df = tableIn(inputs)
  if (!df) return {}
  const [w, h] = previewSize(inputs.img_size, params)

  const by = splitList(params.by)
  const valid = by.filter((c) => df.columns.includes(c))
  if (!valid.length) {
    const title = by.length ? 'Sort (column not found)' : 'Sort (no column)'
    return { table: df, img_size: [w, h], ...metaAndPreview(ctx.cv, df, w, h, title, ctx.track) }
  }

  const ascending = params.ascending !== false
  const naFirst = Math.round(Number(params.na_pos) || 0) === 1
  const rows = [...df.rows].sort((a, b) => {
    for (const col of valid) {
      const aNa = isNa(a[col])
      const bNa = isNa(b[col])
      // NaN position is independent of direction, as in pandas.
      if (aNa || bNa) {
        if (aNa && bNa) continue
        return (aNa ? 1 : -1) * (naFirst ? -1 : 1)
      }
      const cmp = compareValues(a[col], b[col])
      if (cmp !== 0) return ascending ? cmp : -cmp
    }
    return 0
  })

  const out = makeDf(df.columns, rows)
  return { table: out, img_size: [w, h], ...metaAndPreview(ctx.cv, out, w, h, `Sort by ${valid[0]}`, ctx.track) }
}

const SAMPLE_MODES = ['head', 'tail', 'sample', 'slice']

export const dfSample: NodeImpl = (inputs, params, ctx) => {
  const df = tableIn(inputs)
  if (!df) return {}

  const mode = SAMPLE_MODES[Math.round(Number(params.mode) || 0)] ?? 'head'
  const n = Math.max(1, Math.round(Number(params.n) || 10))

  let rows: Record<string, unknown>[]
  if (mode === 'head') rows = df.rows.slice(0, n)
  else if (mode === 'tail') rows = df.rows.slice(Math.max(0, df.rows.length - n))
  else if (mode === 'sample') {
    // Partial Fisher–Yates over an index list, so the seed reproduces the draw.
    const rand = seededRandom(Math.round(Number(params.seed) || 42))
    const idx = df.rows.map((_, i) => i)
    const take = Math.min(n, idx.length)
    for (let i = 0; i < take; i++) {
      const j = i + Math.floor(rand() * (idx.length - i))
      ;[idx[i], idx[j]] = [idx[j], idx[i]]
    }
    rows = idx.slice(0, take).map((i) => df.rows[i])
  } else {
    rows = df.rows.slice(Math.round(Number(params.start) || 0), Math.round(Number(params.end) ?? 10))
  }

  const out = makeDf(df.columns, rows)
  const [w, h] = previewSize(inputs.img_size, params)
  return {
    table: out,
    row_count: out.rows.length,
    img_size: [w, h],
    ...metaAndPreview(ctx.cv, out, w, h, mode, ctx.track),
  }
}

/** A negative row position counts from the end, as in `df.iloc[-100:]`. */
function absolutePosition(value: number, n: number): number {
  return value >= 0 ? value : n + value
}

/** A wired scalar wins over the typed parameter. */
function scalarOrParam(wired: unknown, param: unknown, fallback: number): number {
  if (typeof wired === 'number' && Number.isFinite(wired)) return Math.round(wired)
  const n = Number(param)
  return Number.isFinite(n) ? Math.round(n) : fallback
}

export const dfSlice: NodeImpl = (inputs, params, ctx) => {
  const df = tableIn(inputs)
  if (!df) return {}

  const n = df.rows.length
  let start = absolutePosition(scalarOrParam(inputs.start, params.start, 0), n)
  let end = absolutePosition(scalarOrParam(inputs.end, params.end, -1), n)
  const step = Math.max(1, Math.round(Number(params.step) || 1))

  start = Math.max(0, Math.min(start, n))
  end = Math.min(end, n - 1)

  const rows: Record<string, unknown>[] = []
  // End is inclusive here, unlike the half-open slice everywhere else.
  for (let i = start; i <= end; i += step) rows.push(df.rows[i])

  const out = makeDf(df.columns, rows)
  const [w, h] = previewSize(inputs.img_size, params)
  return {
    table: out,
    preview: ctx.track(renderDfTable(ctx.cv, out, w, h, `Slice ${start}–${end} (${rows.length} rows)`)),
    row_count: rows.length,
    df_meta: { shape: [rows.length, out.columns.length], columns: [...out.columns], dtypes: dfMeta(out).dtypes },
    img_size: [w, h],
  }
}

const AGG_LABELS = ['mean', 'sum', 'count', 'min', 'max', 'std', 'median', 'nunique']

export const dfGroupby: NodeImpl = (inputs, params, ctx) => {
  const df = tableIn(inputs)
  if (!df) return {}
  const by = String(params.by ?? '').trim()
  if (!by || !df.columns.includes(by)) return {}

  const agg = AGG_LABELS[Math.round(Number(params.agg) || 0)] ?? 'mean'
  const requested = splitList(params.cols).filter((c) => df.columns.includes(c) && c !== by)
  let target = requested.length ? requested : df.columns.filter((c) => c !== by && isNumericColumn(df, c))
  // mean/median/std are only defined on numbers, whatever the user asked for.
  if (agg === 'mean' || agg === 'median' || agg === 'std') target = target.filter((c) => isNumericColumn(df, c))

  const groups = new Map<string, { key: unknown; rows: Record<string, unknown>[] }>()
  for (const row of df.rows) {
    const key = row[by]
    const id = isNa(key) ? ' nan' : String(key)
    let group = groups.get(id)
    if (!group) {
      group = { key, rows: [] }
      groups.set(id, group)
    }
    group.rows.push(row)
  }

  // pandas sorts group keys by default.
  const ordered = [...groups.values()].sort((a, b) => compareValues(a.key, b.key))
  const rows = ordered.map((group) => {
    const out: Record<string, unknown> = { [by]: group.key }
    for (const col of target) out[col] = aggregate(group.rows.map((r) => r[col]), agg)
    return out
  })

  const out = makeDf([by, ...target], rows)
  const [w, h] = previewSize(inputs.img_size, params)
  return {
    table: out,
    row_count: rows.length,
    img_size: [w, h],
    ...metaAndPreview(ctx.cv, out, w, h, `GroupBy ${by} / ${agg}`, ctx.track),
  }
}

const HOW_LABELS = ['inner', 'left', 'right', 'outer']

export const dfMerge: NodeImpl = (inputs, params, ctx) => {
  const left = isDf(inputs.left) ? (inputs.left as DataFrame) : null
  const right = isDf(inputs.right) ? (inputs.right as DataFrame) : null
  if (!left || !right) return {}

  const how = HOW_LABELS[Math.round(Number(params.how) || 0)] ?? 'inner'
  const leftKey = String(params.left_on ?? '').trim()
  const rightKey = String(params.right_on ?? '').trim() || leftKey

  let keys: [string, string][]
  if (leftKey && left.columns.includes(leftKey) && right.columns.includes(rightKey)) {
    keys = [[leftKey, rightKey]]
  } else {
    // No usable key given: fall back to joining on every shared column name.
    const common = left.columns.filter((c) => right.columns.includes(c))
    if (!common.length) return {}
    keys = common.map((c) => [c, c] as [string, string])
  }

  const keyOf = (row: Record<string, unknown>, side: 0 | 1) =>
    keys.map(([l, r]) => (isNa(row[side === 0 ? l : r]) ? ' nan' : String(row[side === 0 ? l : r]))).join('')

  const rightIndex = new Map<string, Record<string, unknown>[]>()
  for (const row of right.rows) {
    const id = keyOf(row, 1)
    const bucket = rightIndex.get(id)
    if (bucket) bucket.push(row)
    else rightIndex.set(id, [row])
  }

  // Shared non-key columns get pandas' _x/_y suffixes rather than overwriting.
  const keyRightNames = new Set(keys.map(([, r]) => r))
  const overlapping = new Set(left.columns.filter((c) => right.columns.includes(c) && !keyRightNames.has(c)))
  const columns = [
    ...left.columns.map((c) => (overlapping.has(c) ? `${c}_x` : c)),
    ...right.columns.filter((c) => !keyRightNames.has(c)).map((c) => (overlapping.has(c) ? `${c}_y` : c)),
  ]

  const blend = (l: Record<string, unknown> | null, r: Record<string, unknown> | null) => {
    const out: Record<string, unknown> = {}
    for (const c of left.columns) out[overlapping.has(c) ? `${c}_x` : c] = l ? l[c] : null
    for (const c of right.columns) {
      if (keyRightNames.has(c)) continue
      out[overlapping.has(c) ? `${c}_y` : c] = r ? r[c] : null
    }
    // A right-only row still has to carry the key values into the key columns.
    if (!l && r) keys.forEach(([lc, rc]) => (out[overlapping.has(lc) ? `${lc}_x` : lc] = r[rc]))
    return out
  }

  let rows: Record<string, unknown>[] = []
  if (how === 'right') {
    // A right join walks the right frame in its own order, which is how pandas
    // orders the result — not left order with the leftovers appended.
    const leftIndex = new Map<string, Record<string, unknown>[]>()
    for (const row of left.rows) {
      const id = keyOf(row, 0)
      const bucket = leftIndex.get(id)
      if (bucket) bucket.push(row)
      else leftIndex.set(id, [row])
    }
    for (const rrow of right.rows) {
      const matches = leftIndex.get(keyOf(rrow, 1))
      if (matches && matches.length) for (const lrow of matches) rows.push(blend(lrow, rrow))
      else rows.push(blend(null, rrow))
    }
  } else {
    const matchedRight = new Set<string>()
    const keyed: { id: string; row: Record<string, unknown> }[] = []
    for (const lrow of left.rows) {
      const id = keyOf(lrow, 0)
      const matches = rightIndex.get(id)
      if (matches && matches.length) {
        matchedRight.add(id)
        for (const rrow of matches) keyed.push({ id, row: blend(lrow, rrow) })
      } else if (how === 'outer') {
        keyed.push({ id, row: blend(lrow, null) })
      } else if (how === 'left') {
        keyed.push({ id, row: blend(lrow, null) })
      }
    }
    if (how === 'outer') {
      for (const [id, bucket] of rightIndex) {
        if (matchedRight.has(id)) continue
        for (const rrow of bucket) keyed.push({ id, row: blend(null, rrow) })
      }
      // An outer join comes back sorted by key; a left join keeps left order.
      keyed.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
    }
    rows = keyed.map((k) => k.row)
  }

  const out = makeDf(columns, rows)
  const [w, h] = previewSize(inputs.img_size, params)
  return {
    table: out,
    row_count: rows.length,
    img_size: [w, h],
    ...metaAndPreview(ctx.cv, out, w, h, `Merge (${how})`, ctx.track),
  }
}

const FILL_STRATEGIES = ['value', 'mean', 'median', 'mode', 'ffill', 'bfill', 'drop_rows', 'drop_cols']

export const dfFillna: NodeImpl = (inputs, params, ctx) => {
  const df = tableIn(inputs)
  if (!df) return {}

  const strategy = FILL_STRATEGIES[Math.round(Number(params.strategy) || 0)] ?? 'value'
  const requested = splitList(params.columns).filter((c) => df.columns.includes(c))
  const cols = requested.length ? requested : [...df.columns]
  const nullBefore = df.columns.reduce((n, c) => n + df.rows.filter((r) => isNa(r[c])).length, 0)

  let columns = [...df.columns]
  let rows = copyRows(df)

  if (strategy === 'drop_rows') {
    rows = rows.filter((row) => cols.every((c) => !isNa(row[c])))
  } else if (strategy === 'drop_cols') {
    const thresh = Math.max(1, Math.round(Number(params.thresh) || 1))
    columns = columns.filter((c) => rows.filter((r) => !isNa(r[c])).length >= thresh)
    rows = rows.map((row) => {
      const out: Record<string, unknown> = {}
      for (const c of columns) out[c] = row[c]
      return out
    })
  } else if (strategy === 'ffill' || strategy === 'bfill') {
    const order = strategy === 'ffill' ? rows : [...rows].reverse()
    for (const c of cols) {
      let carry: unknown = null
      for (const row of order) {
        if (isNa(row[c])) {
          if (!isNa(carry)) row[c] = carry
        } else carry = row[c]
      }
    }
  } else if (strategy === 'mode') {
    for (const c of cols) {
      const counts = new Map<string, { value: unknown; n: number }>()
      for (const row of rows) {
        if (isNa(row[c])) continue
        const id = String(row[c])
        const seen = counts.get(id)
        if (seen) seen.n++
        else counts.set(id, { value: row[c], n: 1 })
      }
      let best: { value: unknown; n: number } | null = null
      for (const entry of counts.values()) if (!best || entry.n > best.n) best = entry
      if (best) for (const row of rows) if (isNa(row[c])) row[c] = best.value
    }
  } else if (strategy === 'mean' || strategy === 'median') {
    for (const c of cols.filter((c) => isNumericColumn(df, c))) {
      const fill = aggregate(rows.map((r) => r[c]), strategy)
      if (fill === null) continue
      for (const row of rows) if (isNa(row[c])) row[c] = fill
    }
  } else {
    const raw = String(params.value ?? '0')
    const asNumber = Number(raw)
    const fill: unknown = raw.trim() !== '' && Number.isFinite(asNumber) ? asNumber : raw
    for (const c of cols) for (const row of rows) if (isNa(row[c])) row[c] = fill
  }

  const out = makeDf(columns, rows)
  const [w, h] = previewSize(inputs.img_size, params)
  return {
    table: out,
    null_count: nullBefore,
    img_size: [w, h],
    ...metaAndPreview(ctx.cv, out, w, h, `Fill NA (${strategy})`, ctx.track),
  }
}

export const dfNewCol: NodeImpl = (inputs, params, ctx) => {
  const df = tableIn(inputs)
  if (!df) return {}

  const name = String(params.name ?? 'new_col').trim() || 'new_col'
  const expr = String(params.expr ?? '').trim()
  const [w, h] = previewSize(inputs.img_size, params)
  if (!expr) return { table: df, img_size: [w, h], ...metaAndPreview(ctx.cv, df, w, h, name, ctx.track) }

  // The desktop evaluates the expression once against whole pandas Series. There
  // are no vectorised Series here, so the same `df['a'] + df['b']` source is run
  // per row with `df` bound to that row — which is what these expressions mean.
  let compute: (row: Record<string, unknown>, i: number) => unknown
  try {
    compute = new Function('df', 'i', 'Math', `return (${expr})`) as typeof compute
  } catch {
    return { table: df, img_size: [w, h], ...metaAndPreview(ctx.cv, df, w, h, `${name} (bad expression)`, ctx.track) }
  }

  const rows = copyRows(df)
  try {
    rows.forEach((row, i) => {
      // Missing cells are handed over as NaN, not null: JS would evaluate
      // `null * 2` to 0, where pandas propagates the missing value.
      const view: Record<string, unknown> = {}
      for (const c of df.columns) view[c] = isNa(row[c]) ? NaN : row[c]
      const value = compute(view, i)
      row[name] = typeof value === 'number' && !Number.isFinite(value) ? null : value
    })
  } catch {
    return { table: df, img_size: [w, h], ...metaAndPreview(ctx.cv, df, w, h, `${name} (eval error)`, ctx.track) }
  }

  const columns = df.columns.includes(name) ? [...df.columns] : [...df.columns, name]
  const out = makeDf(columns, rows)
  return { table: out, img_size: [w, h], ...metaAndPreview(ctx.cv, out, w, h, name, ctx.track) }
}

export const dfRename: NodeImpl = (inputs, params, ctx) => {
  const df = tableIn(inputs)
  if (!df) return {}

  const strip = !!params.strip_spaces
  const source = df.columns.map((c) => (strip ? c.trim() : c))
  const map = parseRenameMap(params.map)
  const target = source.map((c) => map.get(c) ?? c)

  const rows = df.rows.map((row) => {
    const out: Record<string, unknown> = {}
    df.columns.forEach((original, i) => {
      out[target[i]] = row[original]
    })
    return out
  })

  const out = makeDf(target, rows)
  const [w, h] = previewSize(inputs.img_size, params)
  return { table: out, img_size: [w, h], ...metaAndPreview(ctx.cv, out, w, h, 'Rename', ctx.track) }
}

/* -------------------------------------------------------------- accumulate */

interface CollectState {
  buffer: DataFrame[]
  lastSeq: unknown
  lastCapture: number
  lastReset: number
}

export const dfCollect: NodeImpl = (inputs, params, ctx) => {
  let state = ctx.state.get(ctx.nodeId) as CollectState | undefined
  if (!state) {
    state = { buffer: [], lastSeq: null, lastCapture: 0, lastReset: 0 }
    ctx.state.set(ctx.nodeId, state)
  }

  const resetTrigger = Number(params.reset) ? 1 : 0
  if (resetTrigger && !state.lastReset) {
    state.buffer = []
    state.lastSeq = null
  }
  state.lastReset = resetTrigger

  const table = tableIn(inputs)
  const seq = inputs.seq
  const captureTrigger = Number(params.capture) ? 1 : 0

  let shouldCapture = false
  if (seq !== null && seq !== undefined) {
    // A wired sequence (e.g. a folder iterator index) drives capture on its own.
    if (seq !== state.lastSeq) {
      shouldCapture = true
      state.lastSeq = seq
    }
  } else if (captureTrigger && !state.lastCapture) {
    shouldCapture = true
  }
  state.lastCapture = captureTrigger

  if (shouldCapture && table && table.rows.length) state.buffer.push(table)

  if (!state.buffer.length) return { data: emptyDf(), captured: 0, rows: 0 }

  // Concatenate on the union of columns, filling gaps — pandas' concat default.
  const columns: string[] = []
  for (const df of state.buffer) for (const c of df.columns) if (!columns.includes(c)) columns.push(c)
  const rows: Record<string, unknown>[] = []
  for (const df of state.buffer) {
    for (const row of df.rows) {
      const out: Record<string, unknown> = {}
      for (const c of columns) out[c] = c in row ? row[c] : null
      rows.push(out)
    }
  }

  return { data: makeDf(columns, rows), captured: state.buffer.length, rows: rows.length }
}

/* ------------------------------------------------------------------- editor */

interface CellEdit {
  row?: number
  __row_index__?: number
  col?: string
  column?: string
  value?: unknown
}

export const dfEditor: NodeImpl = (inputs, params, ctx) => {
  const df = tableIn(inputs)
  if (!df) return {}

  let edits: CellEdit[] = []
  const raw = params.edits
  if (Array.isArray(raw)) edits = raw as CellEdit[]
  else if (typeof raw === 'string' && raw.trim()) {
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) edits = parsed
    } catch {
      // A malformed edit list must not take the whole table down.
    }
  }

  const rows = copyRows(df)
  for (const edit of edits) {
    const index = Number(edit.row ?? edit.__row_index__)
    const col = String(edit.col ?? edit.column ?? '')
    if (!Number.isInteger(index) || index < 0 || index >= rows.length || !df.columns.includes(col)) continue
    // Cell values arrive as strings from the editor; numeric columns keep their type.
    const value = edit.value
    if (isNumericColumn(df, col) && typeof value === 'string') {
      const n = Number(value.trim())
      rows[index][col] = value.trim() === '' ? null : Number.isFinite(n) ? n : value
    } else {
      rows[index][col] = value === '' ? null : value
    }
  }

  const out = makeDf(df.columns, rows)
  const offset = Math.max(0, Math.min(Math.round(Number(params.row_offset) || 0), Math.max(0, out.rows.length - 1)))
  const maxRows = Math.max(1, Math.round(Number(params.max_rows) || 500))
  const window = out.rows.slice(offset, offset + maxRows)

  const base = dfMeta(out)
  // The row window travels as a JSON string: the engine drops any output holding
  // a list of more than 2000 items, which would silently empty the editor.
  const payload = {
    columns: [...out.columns],
    dtypes: base.dtypes,
    nulls: base.nulls,
    rows: window.map((row, i) => {
      const entry: Record<string, unknown> = {}
      for (const c of out.columns) entry[c] = isNa(row[c]) ? null : (row[c] as unknown)
      entry.__row_index__ = offset + i
      return entry
    }),
  }

  return {
    table: out,
    preview: ctx.track(renderDfTable(ctx.cv, out, 420, 200, 'Editor')),
    df_meta: {
      shape: [out.rows.length, out.columns.length],
      columns: [...out.columns],
      row_count: out.rows.length,
      offset,
      window: window.length,
      truncated: window.length < out.rows.length,
      table_json: JSON.stringify(payload),
    },
  }
}

/* ------------------------------------------------------------------- export */

export const dfExport: NodeImpl = (inputs, params) => {
  if (!Number(params.save)) return {}
  const df = tableIn(inputs)
  if (!df) return {}

  let path = String(params.path ?? 'output.csv').trim() || 'output.csv'
  const override = inputs.filename
  if (override) {
    // Keep the folder and extension from the path, take the base name from the input.
    const base = String(override).trim().split('/').pop()!.replace(/\.[^.]*$/, '')
    const ext = path.includes('.') ? path.slice(path.lastIndexOf('.')) : '.csv'
    const dir = path.includes('/') ? path.slice(0, path.lastIndexOf('/') + 1) : ''
    path = `${dir}${base}${ext}`
  }

  // Excel and Parquet are binary formats with no browser-native writer, so both
  // fall back to the closest text format rather than producing a corrupt file.
  const format = Math.round(Number(params.format) || 0)
  if (format === 2) {
    const records = df.rows.map((row) => {
      const out: Record<string, unknown> = {}
      for (const c of df.columns) out[c] = isNa(row[c]) ? null : (row[c] as unknown)
      return out
    })
    downloadFile(path.replace(/\.[^.]*$/, '') + '.json', JSON.stringify(records, null, 2), 'application/json')
  } else {
    const separator = String(params.separator ?? ',') || ','
    downloadFile(path.replace(/\.[^.]*$/, '') + '.csv', toCsv(df, separator), 'text/csv')
  }
  return {}
}

/* --------------------------------------------------------------------- plot */

const CHART_TYPES = ['line', 'bar', 'scatter', 'histogram', 'box', 'area', 'pie']

const PLOT_PALETTE: [number, number, number][] = [
  [246, 130, 59],
  [80, 175, 76],
  [60, 76, 231],
  [180, 100, 200],
  [40, 190, 230],
  [200, 80, 120],
  [90, 200, 180],
  [130, 130, 240],
]

function numericSeries(values: unknown[]): number[] {
  return values.map((v) => {
    if (typeof v === 'number') return v
    const n = Number.parseFloat(String(v))
    return Number.isNaN(n) ? NaN : n
  })
}

interface PlotFrame {
  left: number
  top: number
  width: number
  height: number
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  logX: boolean
  logY: boolean
}

function project(frame: PlotFrame, x: number, y: number): [number, number] {
  const fx = frame.logX ? Math.log10(Math.max(1e-12, x)) : x
  const fy = frame.logY ? Math.log10(Math.max(1e-12, y)) : y
  const xMin = frame.logX ? Math.log10(Math.max(1e-12, frame.xMin)) : frame.xMin
  const xMax = frame.logX ? Math.log10(Math.max(1e-12, frame.xMax)) : frame.xMax
  const yMin = frame.logY ? Math.log10(Math.max(1e-12, frame.yMin)) : frame.yMin
  const yMax = frame.logY ? Math.log10(Math.max(1e-12, frame.yMax)) : frame.yMax
  const px = frame.left + ((fx - xMin) / (xMax - xMin || 1)) * frame.width
  const py = frame.top + frame.height - ((fy - yMin) / (yMax - yMin || 1)) * frame.height
  return [Math.round(px), Math.round(py)]
}

export const dfPlot: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const [w, h] = previewSize(inputs.img_size, { width: 480, height: 320, ...params })
  const width = Math.max(160, Math.round(Number(params.out_w) || w))
  const height = Math.max(120, Math.round(Number(params.out_h) || h))

  const transparentBg = Math.round(Number(params.export_bg) || 0) === 1
  const bg = transparentBg ? new cv.Scalar(255, 255, 255, 255) : new cv.Scalar(255, 255, 255, 255)
  const img = ctx.track(new cv.Mat(height, width, cv.CV_8UC3, bg))

  const font = cv.FONT_HERSHEY_SIMPLEX
  const ink = new cv.Scalar(60, 60, 60, 255)
  const gridColour = new cv.Scalar(220, 220, 220, 255)

  const df = tableIn(inputs)
  const meta = df ? { shape: [df.rows.length, df.columns.length], columns: [...df.columns] } : {}

  // Raw X/Y arrays wired in take precedence over the table + column names.
  const wiredX = Array.isArray(inputs.x) ? numericSeries(inputs.x as unknown[]) : null
  const wiredY = Array.isArray(inputs.y) ? numericSeries(inputs.y as unknown[]) : null

  let labels: string[] = []
  let series: { name: string; values: number[] }[] = []

  if (wiredY && wiredY.length) {
    series = [{ name: 'y', values: wiredY }]
    labels = (wiredX ?? wiredY.map((_, i) => i)).map((v) => cellText(v))
  } else if (df) {
    const yCols = splitList(params.y_cols).filter((c) => df.columns.includes(c))
    const chosen = yCols.length ? yCols : df.columns.filter((c) => isNumericColumn(df, c)).slice(0, 3)
    series = chosen.map((c) => ({ name: c, values: numericSeries(column(df, c)) }))
    const xCol = resolveColumn(df, params.x_col)
    labels = xCol ? column(df, xCol).map(cellText) : df.rows.map((_, i) => String(i))
  }

  if (!series.length || !series[0].values.length) {
    cv.putText(img, 'No data', new cv.Point(12, Math.round(height / 2)), font, 0.5, ink, 1, cv.LINE_AA)
    return { main: img, df_meta: meta }
  }

  const maxPoints = Math.round(Number(params.max_points) ?? 5000)
  if (maxPoints > 0) {
    // Decimate rather than draw more points than the plot has pixels for.
    const stride = Math.max(1, Math.ceil(series[0].values.length / maxPoints))
    if (stride > 1) {
      series = series.map((s) => ({ name: s.name, values: s.values.filter((_, i) => i % stride === 0) }))
      labels = labels.filter((_, i) => i % stride === 0)
    }
  }

  if (params.sort_x && labels.length === series[0].values.length) {
    const order = labels.map((_, i) => i).sort((a, b) => compareValues(labels[a], labels[b]))
    labels = order.map((i) => labels[i])
    series = series.map((s) => ({ name: s.name, values: order.map((i) => s.values[i]) }))
  }

  const chart = CHART_TYPES[Math.round(Number(params.chart_type) || 0)] ?? 'line'

  // Histogram bins its single series before anything is measured or drawn.
  if (chart === 'histogram') {
    const bins = Math.max(5, Math.round(Number(params.bins) || 30))
    const values = series[0].values.filter((v) => Number.isFinite(v))
    if (!values.length) {
      cv.putText(img, 'No numeric data', new cv.Point(12, Math.round(height / 2)), font, 0.5, ink, 1, cv.LINE_AA)
      return { main: img, df_meta: meta }
    }
    const lo = Math.min(...values)
    const hi = Math.max(...values)
    const span = hi - lo || 1
    const counts = new Array<number>(bins).fill(0)
    for (const v of values) counts[Math.min(bins - 1, Math.floor(((v - lo) / span) * bins))]++
    series = [{ name: series[0].name, values: counts }]
    labels = counts.map((_, i) => String(Number((lo + (i * span) / bins).toPrecision(4))))
  }

  const title = String(params.title ?? '').trim()
  const marginLeft = 46
  const marginRight = 10
  const marginTop = title ? 26 : 12
  const marginBottom = 26
  const frameW = width - marginLeft - marginRight
  const frameH = height - marginTop - marginBottom

  const allValues = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v))
  let yMin = Math.min(...allValues)
  let yMax = Math.max(...allValues)
  if (chart === 'bar' || chart === 'histogram' || chart === 'area') yMin = Math.min(0, yMin)
  if (yMin === yMax) {
    yMin -= 1
    yMax += 1
  }

  const frame: PlotFrame = {
    left: marginLeft,
    top: marginTop,
    width: frameW,
    height: frameH,
    xMin: 0,
    xMax: Math.max(1, series[0].values.length - 1),
    yMin,
    yMax,
    logX: !!params.x_log,
    logY: !!params.y_log,
  }

  if (title) cv.putText(img, title.slice(0, 60), new cv.Point(marginLeft, 18), font, 0.45, ink, 1, cv.LINE_AA)

  // Axes + horizontal gridlines with value labels.
  const ticks = 5
  for (let i = 0; i <= ticks; i++) {
    const value = yMin + ((yMax - yMin) * i) / ticks
    const [, py] = project(frame, frame.xMin, value)
    if (params.grid !== false) {
      cv.line(img, new cv.Point(marginLeft, py), new cv.Point(marginLeft + frameW, py), gridColour, 1)
    }
    cv.putText(img, String(Number(value.toPrecision(4))), new cv.Point(2, py + 4), font, 0.3, ink, 1, cv.LINE_AA)
  }
  cv.line(img, new cv.Point(marginLeft, marginTop), new cv.Point(marginLeft, marginTop + frameH), ink, 1)
  cv.line(img, new cv.Point(marginLeft, marginTop + frameH), new cv.Point(marginLeft + frameW, marginTop + frameH), ink, 1)

  const thickness = Math.max(1, Math.round(Number(params.marker_size) || 40) / 20)
  const colourOf = (i: number) => {
    const [b, g, r] = PLOT_PALETTE[i % PLOT_PALETTE.length]
    return new cv.Scalar(b, g, r, 255)
  }

  if (chart === 'pie') {
    // Pie ignores the axes entirely: one wedge per value of the first series.
    const values = series[0].values.map((v) => (Number.isFinite(v) ? Math.abs(v) : 0))
    const total = values.reduce((a, b) => a + b, 0) || 1
    const centre = new cv.Point(Math.round(width / 2), Math.round(marginTop + frameH / 2))
    const radius = Math.max(10, Math.round(Math.min(frameW, frameH) / 2) - 6)
    let angle = 0
    values.forEach((v, i) => {
      const sweep = (v / total) * 360
      cv.ellipse(img, centre, new cv.Size(radius, radius), 0, angle, angle + sweep, colourOf(i), -1, cv.LINE_AA)
      angle += sweep
    })
  } else if (chart === 'bar' || chart === 'histogram') {
    const n = series[0].values.length
    const groupW = frameW / Math.max(1, n)
    const stacked = !!params.stacked && series.length > 1
    const barW = Math.max(1, Math.floor((groupW * 0.8) / (stacked ? 1 : series.length)))
    for (let i = 0; i < n; i++) {
      let stackBase = 0
      series.forEach((s, k) => {
        const v = Number.isFinite(s.values[i]) ? s.values[i] : 0
        const from = stacked ? stackBase : 0
        const to = stacked ? stackBase + v : v
        if (stacked) stackBase = to
        const x0 = Math.round(marginLeft + i * groupW + groupW * 0.1 + (stacked ? 0 : k * barW))
        const [, y0] = project(frame, 0, Math.max(from, to))
        const [, y1] = project(frame, 0, Math.min(from, to))
        cv.rectangle(img, new cv.Point(x0, y0), new cv.Point(x0 + barW, y1), colourOf(k), -1)
      })
    }
  } else if (chart === 'scatter') {
    const radius = Math.max(1, Math.round(Math.sqrt(Number(params.marker_size) || 40) / 2))
    series.forEach((s, k) => {
      const colour = colourOf(k)
      s.values.forEach((v, i) => {
        if (!Number.isFinite(v)) return
        const [px, py] = project(frame, i, v)
        cv.circle(img, new cv.Point(px, py), radius, colour, -1, cv.LINE_AA)
      })
    })
  } else if (chart === 'box') {
    // One box-and-whisker per series, laid out left to right.
    const slotW = frameW / series.length
    series.forEach((s, k) => {
      const sorted = s.values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b)
      if (!sorted.length) return
      const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))]
      const [q1, med, q3] = [at(0.25), at(0.5), at(0.75)]
      const cx = Math.round(marginLeft + slotW * (k + 0.5))
      const halfW = Math.max(4, Math.round(slotW * 0.25))
      const [, yq1] = project(frame, 0, q1)
      const [, yq3] = project(frame, 0, q3)
      const [, ymed] = project(frame, 0, med)
      const [, ylo] = project(frame, 0, sorted[0])
      const [, yhi] = project(frame, 0, sorted[sorted.length - 1])
      const colour = colourOf(k)
      cv.line(img, new cv.Point(cx, ylo), new cv.Point(cx, yhi), colour, 1)
      cv.rectangle(img, new cv.Point(cx - halfW, yq3), new cv.Point(cx + halfW, yq1), colour, 1)
      cv.line(img, new cv.Point(cx - halfW, ymed), new cv.Point(cx + halfW, ymed), colour, 2)
    })
  } else {
    // line / area
    series.forEach((s, k) => {
      const colour = colourOf(k)
      let previous: [number, number] | null = null
      s.values.forEach((v, i) => {
        if (!Number.isFinite(v)) {
          previous = null
          return
        }
        const point = project(frame, i, v)
        if (chart === 'area') {
          const [, yBase] = project(frame, 0, Math.max(frame.yMin, 0))
          cv.line(img, new cv.Point(point[0], point[1]), new cv.Point(point[0], yBase), colour, 1)
        }
        if (previous) {
          cv.line(img, new cv.Point(previous[0], previous[1]), new cv.Point(point[0], point[1]), colour, thickness, cv.LINE_AA)
        }
        previous = point
      })
    })
  }

  // X labels: only as many as fit without overlapping.
  if (labels.length) {
    const every = Math.max(1, Math.ceil(labels.length / Math.max(1, Math.floor(frameW / 44))))
    labels.forEach((label, i) => {
      if (i % every !== 0) return
      const [px] = project(frame, i, frame.yMin)
      cv.putText(img, label.slice(0, 7), new cv.Point(px - 10, height - 8), font, 0.3, ink, 1, cv.LINE_AA)
    })
  }

  if (params.legend !== false && series.length > 1) {
    series.forEach((s, k) => {
      const y = marginTop + 10 + k * 12
      cv.rectangle(img, new cv.Point(width - 74, y - 6), new cv.Point(width - 64, y + 1), colourOf(k), -1)
      cv.putText(img, s.name.slice(0, 9), new cv.Point(width - 60, y), font, 0.3, ink, 1, cv.LINE_AA)
    })
  }

  return { main: img, df_meta: meta }
}
