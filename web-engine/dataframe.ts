/**
 * A minimal row-oriented DataFrame, standing in for pandas.
 *
 * The desktop `df_*` nodes pass real `pandas.DataFrame` objects between them.
 * There is no pandas in the browser and pulling in a full dataframe library for
 * fifteen nodes is not worth 200 kB, so the family agrees on this shape instead:
 * an ordered column list plus plain row objects. Every operation the desktop
 * nodes perform — select, sort, slice, group, merge, fill, derive — is a short
 * array transform over that.
 *
 * Row order is meaningful (sort and slice rely on it) and `columns` is the
 * authority on column order, since object key order is not something to lean on
 * once rows have been rebuilt by a merge or a groupby.
 */

export interface DataFrame {
  columns: string[]
  rows: Record<string, unknown>[]
}

export function makeDf(columns: string[], rows: Record<string, unknown>[]): DataFrame {
  return { columns: [...columns], rows }
}

export function emptyDf(): DataFrame {
  return { columns: [], rows: [] }
}

export function isDf(v: unknown): v is DataFrame {
  return !!v && typeof v === 'object' && Array.isArray((v as DataFrame).columns) && Array.isArray((v as DataFrame).rows)
}

/** True for pandas' idea of "missing": null, undefined, or NaN. */
export function isNa(v: unknown): boolean {
  return v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))
}

export function column(df: DataFrame, name: string): unknown[] {
  return df.rows.map((r) => r[name])
}

/**
 * Resolves a user-typed column name: exact match first, then trimmed and
 * case-insensitive, mirroring the desktop's `_resolve_col`.
 */
export function resolveColumn(df: DataFrame, name: unknown): string | null {
  const wanted = String(name ?? '').trim()
  if (!wanted) return null
  if (df.columns.includes(wanted)) return wanted
  const lowered = new Map(df.columns.map((c) => [c.trim().toLowerCase(), c]))
  return lowered.get(wanted.toLowerCase()) ?? null
}

/** Parses a "a, b, c" parameter into a trimmed, non-empty list. */
export function splitList(raw: unknown): string[] {
  return String(raw ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
}

/** Parses an "old:new, old2:new2" parameter into a rename map. */
export function parseRenameMap(raw: unknown): Map<string, string> {
  const map = new Map<string, string>()
  for (const pair of String(raw ?? '').split(',')) {
    const at = pair.indexOf(':')
    if (at < 0) continue
    const from = pair.slice(0, at).trim()
    const to = pair.slice(at + 1).trim()
    if (from) map.set(from, to)
  }
  return map
}

/** pandas-style dtype label, inferred from the non-missing values in a column. */
export function dtypeOf(df: DataFrame, name: string): string {
  let sawFloat = false
  let sawNumber = false
  let sawBool = false
  let sawOther = false
  let sawMissing = false
  for (const row of df.rows) {
    const v = row[name]
    if (isNa(v)) {
      sawMissing = true
      continue
    }
    if (typeof v === 'boolean') sawBool = true
    else if (typeof v === 'number') {
      sawNumber = true
      if (!Number.isInteger(v)) sawFloat = true
    } else sawOther = true
  }
  if (sawOther || (sawBool && sawNumber)) return 'object'
  if (sawBool) return 'bool'
  // A missing value forces a numeric column to float, since pandas stores it as NaN.
  if (sawNumber) return sawFloat || sawMissing ? 'float64' : 'int64'
  // An all-missing column is float64 for the same reason.
  return 'float64'
}

export function isNumericColumn(df: DataFrame, name: string): boolean {
  const dtype = dtypeOf(df, name)
  return dtype === 'int64' || dtype === 'float64'
}

function serialize(v: unknown): unknown {
  if (isNa(v)) return null
  if (typeof v === 'number' || typeof v === 'boolean') return v
  return String(v)
}

/** The `df_meta` payload every DataFrame node emits; the inspector reads it. */
export function dfMeta(df: DataFrame, headRows = 8): Record<string, unknown> {
  const nulls: Record<string, number> = {}
  const dtypes: Record<string, string> = {}
  for (const col of df.columns) {
    dtypes[col] = dtypeOf(df, col)
    nulls[col] = df.rows.reduce((n, row) => n + (isNa(row[col]) ? 1 : 0), 0)
  }
  return {
    shape: [df.rows.length, df.columns.length],
    columns: [...df.columns],
    dtypes,
    nulls,
    head: df.rows.slice(0, headRows).map((row) => {
      const out: Record<string, unknown> = {}
      for (const col of df.columns) out[col] = serialize(row[col])
      return out
    }),
  }
}

/** Formats a cell the way the desktop's matplotlib table does: short and flat. */
export function cellText(v: unknown): string {
  if (v === null || v === undefined) return 'None'
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return 'NaN'
    if (Number.isInteger(v)) return String(v)
    return String(Number(v.toPrecision(6)))
  }
  return String(v)
}

const PREVIEW_ROWS = 8
const PREVIEW_COLS = 7

/**
 * Dark-theme table image of the first rows/columns, replacing the desktop's
 * matplotlib `ax.table`. Drawn with rectangles and putText since there is no
 * plotting library in the worker.
 */
export function renderDfTable(cv: any, df: DataFrame, width: number, height: number, title = ''): any {
  const w = Math.max(120, Math.round(width))
  const h = Math.max(80, Math.round(height))
  const img = new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(22, 22, 22, 255))

  const cols = df.columns.slice(0, PREVIEW_COLS)
  const rows = df.rows.slice(0, PREVIEW_ROWS)
  const labels = cols.length > 0 ? cols : ['(no data)']

  const titleH = title ? 20 : 4
  const headerH = 20
  const bodyH = Math.max(0, h - titleH - headerH - 4)
  const rowCount = Math.max(1, rows.length)
  const rowH = Math.max(12, Math.min(24, Math.floor(bodyH / rowCount)))
  const colW = Math.floor((w - 8) / labels.length)

  const font = cv.FONT_HERSHEY_SIMPLEX
  const scale = 0.32
  const grey = new cv.Scalar(204, 204, 204, 255)
  const indigo = new cv.Scalar(252, 180, 165, 255) // #a5b4fc in BGR
  const headerBg = new cv.Scalar(58, 42, 42, 255)
  const edge = new cv.Scalar(64, 42, 42, 255)

  if (title) {
    cv.putText(img, title.slice(0, Math.floor(w / 6)), new cv.Point(6, 14), font, scale, grey, 1, cv.LINE_AA)
  }

  // Header row.
  const headerTop = titleH
  cv.rectangle(img, new cv.Point(4, headerTop), new cv.Point(4 + colW * labels.length, headerTop + headerH), headerBg, -1)
  labels.forEach((label, j) => {
    const x = 4 + j * colW
    cv.rectangle(img, new cv.Point(x, headerTop), new cv.Point(x + colW, headerTop + headerH), edge, 1)
    cv.putText(img, label.slice(0, 16), new cv.Point(x + 4, headerTop + 14), font, scale, indigo, 1, cv.LINE_AA)
  })

  // Body: alternating stripes, same as the desktop preview.
  const bodyTop = headerTop + headerH
  const stripeA = new cv.Scalar(32, 24, 24, 255)
  const stripeB = new cv.Scalar(40, 26, 26, 255)
  const shown = rows.length > 0 ? rows : [null]
  shown.forEach((row, i) => {
    const y = bodyTop + i * rowH
    if (y + rowH > h) return
    cv.rectangle(img, new cv.Point(4, y), new cv.Point(4 + colW * labels.length, y + rowH), i % 2 === 0 ? stripeA : stripeB, -1)
    labels.forEach((label, j) => {
      const x = 4 + j * colW
      cv.rectangle(img, new cv.Point(x, y), new cv.Point(x + colW, y + rowH), edge, 1)
      const text = row && cols.length > 0 ? cellText((row as Record<string, unknown>)[label]).slice(0, 16) : '—'
      cv.putText(img, text, new cv.Point(x + 4, y + rowH - 5), font, scale, grey, 1, cv.LINE_AA)
    })
  })

  return img
}

/** Resolves the preview size from a wired `img_size` list, falling back to params. */
export function previewSize(imgSize: unknown, params: Record<string, any>): [number, number] {
  if (Array.isArray(imgSize) && imgSize.length >= 2) {
    return [Math.max(1, Math.round(Number(imgSize[0]))), Math.max(1, Math.round(Number(imgSize[1])))]
  }
  return [Math.round(Number(params.width) || 420), Math.round(Number(params.height) || 200)]
}

/** The `{df_meta, preview}` pair the desktop's `_meta_and_preview` returns. */
export function metaAndPreview(
  cv: any,
  df: DataFrame,
  width: number,
  height: number,
  title: string,
  track: (mat: any) => any
): Record<string, unknown> {
  return { df_meta: dfMeta(df), preview: track(renderDfTable(cv, df, width, height, title)) }
}

/**
 * RFC-4180 escaping: quote whenever the value contains a delimiter, quote or
 * newline. Whole numbers in a float column are written with a trailing `.0`, as
 * pandas' `to_csv` does, so a file exported here round-trips to the same dtypes
 * as one exported from the desktop.
 */
export function toCsv(df: DataFrame, separator = ','): string {
  const floatColumns = new Set(df.columns.filter((c) => dtypeOf(df, c) === 'float64'))
  const escape = (v: unknown, asFloat: boolean) => {
    let text: string
    if (isNa(v)) text = ''
    else if (asFloat && typeof v === 'number' && Number.isInteger(v)) text = `${v}.0`
    else text = String(v)
    return text.includes(separator) || /["\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
  }
  const lines = [df.columns.map((c) => escape(c, false)).join(separator)]
  for (const row of df.rows) {
    lines.push(df.columns.map((c) => escape(row[c], floatColumns.has(c))).join(separator))
  }
  return lines.join('\n')
}

/**
 * Ascending comparator with pandas' missing-value handling: NaN sorts to one
 * end regardless of direction, numbers compare numerically, everything else
 * compares as text.
 */
export function compareValues(a: unknown, b: unknown): number {
  const aNa = isNa(a)
  const bNa = isNa(b)
  if (aNa && bNa) return 0
  if (aNa) return 1
  if (bNa) return -1
  if (typeof a === 'number' && typeof b === 'number') return a - b
  const at = String(a)
  const bt = String(b)
  return at < bt ? -1 : at > bt ? 1 : 0
}

/** Deterministic PRNG so `df_sample`'s seed parameter reproduces a selection. */
export function seededRandom(seed: number): () => number {
  let state = (seed >>> 0) || 1
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Numeric aggregations `df_groupby` offers, matching pandas' names. */
export function aggregate(values: unknown[], how: string): unknown {
  const nums = values.filter((v): v is number => typeof v === 'number' && !Number.isNaN(v))
  switch (how) {
    case 'count':
      return values.filter((v) => !isNa(v)).length
    case 'nunique':
      return new Set(values.filter((v) => !isNa(v)).map((v) => String(v))).size
    case 'sum':
      return nums.reduce((a, b) => a + b, 0)
    case 'min':
      return nums.length ? Math.min(...nums) : null
    case 'max':
      return nums.length ? Math.max(...nums) : null
    case 'mean':
      return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : null
    case 'median': {
      if (!nums.length) return null
      const sorted = [...nums].sort((a, b) => a - b)
      const mid = sorted.length >> 1
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }
    case 'std': {
      // pandas defaults to the sample standard deviation (ddof=1).
      if (nums.length < 2) return null
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length
      const variance = nums.reduce((a, b) => a + (b - mean) ** 2, 0) / (nums.length - 1)
      return Math.sqrt(variance)
    }
    default:
      return null
  }
}
