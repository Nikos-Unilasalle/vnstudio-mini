import type { NodeImpl } from '../types'
import { applyColormap, jetColor, plasmaColor, viridisColor } from '../colormaps'
import {
  cellText,
  compareValues,
  DataFrame,
  dfMeta,
  isDf,
  isNa,
  isNumericColumn,
  makeDf,
  metaAndPreview,
  previewSize,
  renderDfTable,
  resolveColumn,
  splitList,
} from '../dataframe'

function tableIn(inputs: Record<string, unknown>): DataFrame | null {
  const v = inputs.table ?? inputs.data ?? inputs.main
  return isDf(v) ? v : null
}

function numericColumns(df: DataFrame): string[] {
  return df.columns.filter((c) => isNumericColumn(df, c))
}

function asNumbers(df: DataFrame, column: string): number[] {
  return df.rows.map((r) => Number(r[column])).filter((v) => Number.isFinite(v))
}

/* ------------------------------------------------------------------ filter */

const FILTER_OPS = ['==', '!=', '>', '<', '>=', '<=', 'contains', 'is null', 'is not null']

export const mlDfFilter: NodeImpl = (inputs, params, ctx) => {
  const df = tableIn(inputs)
  if (!df) return {}

  let rows = df.rows
  if (params.dropna) rows = rows.filter((row) => df.columns.every((c) => !isNa(row[c])))

  if (params.enabled !== false) {
    const column = String(params.column ?? '').trim()
    const op = FILTER_OPS[Math.round(Number(params.operator) || 0)] ?? '=='
    // A wired value overrides the typed one.
    const raw = inputs.value !== null && inputs.value !== undefined ? String(inputs.value) : String(params.value ?? '')

    if (column && df.columns.includes(column)) {
      if (op === 'is null') rows = rows.filter((row) => isNa(row[column]))
      else if (op === 'is not null') rows = rows.filter((row) => !isNa(row[column]))
      else if (op === 'contains') rows = rows.filter((row) => String(row[column] ?? '').includes(raw))
      else {
        const asNumber = Number(raw)
        // A numeric comparison when the value parses as a number, otherwise a
        // string comparison — the same fallback pandas' filter takes.
        if (raw.trim() !== '' && Number.isFinite(asNumber)) {
          rows = rows.filter((row) => {
            const v = Number(row[column])
            if (!Number.isFinite(v)) return false
            switch (op) {
              case '==': return v === asNumber
              case '!=': return v !== asNumber
              case '>': return v > asNumber
              case '<': return v < asNumber
              case '>=': return v >= asNumber
              default: return v <= asNumber
            }
          })
        } else if (op === '==' || op === '!=') {
          rows = rows.filter((row) => (String(row[column] ?? '') === raw) === (op === '=='))
        }
      }
    }
  }

  const out = makeDf(df.columns, rows)
  const [w, h] = previewSize(inputs.img_size, params)
  return {
    table: out,
    row_count: out.rows.length,
    img_size: [w, h],
    ...metaAndPreview(ctx.cv, out, w, h, 'Filter', ctx.track),
  }
}

/* ------------------------------------------------------------------- stats */

function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return 0
  // Linear interpolation between order statistics, numpy's default.
  const pos = (sorted.length - 1) * q
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
}

export const mlDfStats: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const df = tableIn(inputs)
  if (!df) return {}

  const requested = splitList(params.columns).filter((c) => df.columns.includes(c))
  const scoped = requested.length ? makeDf(requested, df.rows) : df
  const mode = Math.round(Number(params.mode) || 0)

  let table: DataFrame
  const stats: Record<string, unknown> = {}

  if (mode === 0) {
    // describe(): the eight summary rows pandas reports for numeric columns.
    const cols = numericColumns(scoped)
    const labels = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    const rows = labels.map((label) => {
      const row: Record<string, unknown> = { stat: label }
      for (const c of cols) {
        const values = asNumbers(scoped, c)
        const sorted = [...values].sort((a, b) => a - b)
        const mean = values.reduce((a, b) => a + b, 0) / (values.length || 1)
        let value: number
        if (label === 'count') value = values.length
        else if (label === 'mean') value = mean
        // pandas' describe uses the sample standard deviation, ddof = 1.
        else if (label === 'std') value = values.length > 1 ? Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1)) : 0
        else if (label === 'min') value = sorted[0] ?? 0
        else if (label === '25%') value = quantile(sorted, 0.25)
        else if (label === '50%') value = quantile(sorted, 0.5)
        else if (label === '75%') value = quantile(sorted, 0.75)
        else value = sorted[sorted.length - 1] ?? 0
        row[c] = Math.round(value * 1e4) / 1e4
      }
      return row
    })
    table = makeDf(['stat', ...cols], rows)
    stats.describe = rows
  } else if (mode === 1) {
    table = makeDf(scoped.columns, scoped.rows.slice(0, 10))
    stats.head = table.rows
  } else if (mode === 2) {
    const meta = dfMeta(scoped)
    const dtypes = meta.dtypes as Record<string, string>
    const nulls = meta.nulls as Record<string, number>
    table = makeDf(
      ['column', 'dtype', 'nulls'],
      scoped.columns.map((c) => ({ column: c, dtype: dtypes[c], nulls: nulls[c] }))
    )
    stats.dtypes = dtypes
    stats.nulls = nulls
  } else {
    // value_counts on one column.
    const column = resolveColumn(scoped, params.col_vc) ?? scoped.columns[0]
    const counts = new Map<string, number>()
    for (const row of scoped.rows) {
      const key = isNa(row[column]) ? '(null)' : String(row[column])
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
    const ordered = [...counts.entries()].sort((a, b) => b[1] - a[1])
    table = makeDf(['value', 'count'], ordered.map(([value, count]) => ({ value, count })))
    stats.value_counts = Object.fromEntries(ordered)
  }

  stats.shape = [scoped.rows.length, scoped.columns.length]
  const [w, h] = previewSize(inputs.img_size, { width: 580, height: 320, ...params })
  return { preview: ctx.track(renderDfTable(cv, table, w, h, ['describe', 'head', 'dtypes', 'value counts'][mode] ?? '')), stats_data: stats }
}

/* -------------------------------------------------------------------- join */

export const mlDataframeJoin: NodeImpl = (inputs, params, ctx) => {
  const tables = ['table_a', 'table_b', 'table_c', 'table_d']
    .map((slot) => inputs[slot])
    .filter(isDf) as DataFrame[]
  if (!tables.length) return {}

  const [w, h] = previewSize(inputs.img_size, { width: 420, height: 220, ...params })
  if (tables.length === 1) {
    const only = tables[0]
    return {
      table: only,
      preview: ctx.track(renderDfTable(ctx.cv, only, w, h, 'Join (one table)')),
      row_count: only.rows.length,
      col_count: only.columns.length,
    }
  }

  const key = String(params.join_key ?? '__px_idx').trim() || '__px_idx'
  const joinType = ['inner', 'outer', 'left'][Math.round(Number(params.join_type) ?? 1)] ?? 'outer'
  const dropDuplicates = params.drop_dupes !== false

  const missing = tables.some((t) => !t.columns.includes(key))
  if (missing) return { table: tables[0], preview: ctx.track(renderDfTable(ctx.cv, tables[0], w, h, `Join: key "${key}" missing`)), row_count: tables[0].rows.length, col_count: tables[0].columns.length }

  // Index every table by the key, then walk the key set the join type calls for.
  const indexed = tables.map((t) => {
    const map = new Map<string, Record<string, unknown>>()
    for (const row of t.rows) map.set(String(row[key]), row)
    return map
  })

  let keys: string[]
  if (joinType === 'inner') {
    keys = [...indexed[0].keys()].filter((k) => indexed.every((m) => m.has(k)))
  } else if (joinType === 'left') {
    keys = [...indexed[0].keys()]
  } else {
    const all = new Set<string>()
    for (const m of indexed) for (const k of m.keys()) all.add(k)
    keys = [...all]
  }
  keys.sort(compareValues)

  const columns: string[] = [key]
  const columnSource: { table: number; name: string; as: string }[] = []
  tables.forEach((t, i) => {
    for (const c of t.columns) {
      if (c === key) continue
      let name = c
      if (columns.includes(name)) {
        // A duplicate column name either takes a table suffix or is dropped.
        if (dropDuplicates) continue
        name = `${c}_${String.fromCharCode(97 + i)}`
      }
      columns.push(name)
      columnSource.push({ table: i, name: c, as: name })
    }
  })

  const rows = keys.map((k) => {
    const row: Record<string, unknown> = { [key]: indexed.find((m) => m.has(k))!.get(k)![key] }
    for (const source of columnSource) {
      row[source.as] = indexed[source.table].get(k)?.[source.name] ?? null
    }
    return row
  })

  const out = makeDf(columns, rows)
  return {
    table: out,
    preview: ctx.track(renderDfTable(ctx.cv, out, w, h, `Join (${joinType})`)),
    row_count: out.rows.length,
    col_count: out.columns.length,
  }
}

/* ------------------------------------------------------------------- plots */

const PLOT_BG: [number, number, number] = [26, 26, 26]
const PLOT_INK: [number, number, number] = [200, 200, 200]
const PLOT_GRID: [number, number, number] = [55, 55, 55]

interface Axes {
  left: number
  top: number
  width: number
  height: number
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

/** Draws the plot frame with gridlines and tick labels, and returns the axes. */
function drawAxes(cv: any, img: any, xMin: number, xMax: number, yMin: number, yMax: number, grid: boolean, title: string): Axes {
  const marginLeft = 54
  const marginRight = 14
  const marginTop = title ? 28 : 14
  const marginBottom = 30
  const axes: Axes = {
    left: marginLeft,
    top: marginTop,
    width: img.cols - marginLeft - marginRight,
    height: img.rows - marginTop - marginBottom,
    xMin,
    xMax: xMax === xMin ? xMin + 1 : xMax,
    yMin,
    yMax: yMax === yMin ? yMin + 1 : yMax,
  }

  const ink = new cv.Scalar(PLOT_INK[0], PLOT_INK[1], PLOT_INK[2], 255)
  const gridColour = new cv.Scalar(PLOT_GRID[0], PLOT_GRID[1], PLOT_GRID[2], 255)
  const font = cv.FONT_HERSHEY_SIMPLEX

  if (title) cv.putText(img, title.slice(0, 44), new cv.Point(marginLeft, 19), font, 0.45, ink, 1, cv.LINE_AA)

  for (let i = 0; i <= 4; i++) {
    const y = axes.top + Math.round((axes.height * i) / 4)
    const value = axes.yMax - ((axes.yMax - axes.yMin) * i) / 4
    if (grid) cv.line(img, new cv.Point(axes.left, y), new cv.Point(axes.left + axes.width, y), gridColour, 1)
    cv.putText(img, String(Number(value.toPrecision(4))), new cv.Point(3, y + 4), font, 0.3, ink, 1, cv.LINE_AA)

    const x = axes.left + Math.round((axes.width * i) / 4)
    const xValue = axes.xMin + ((axes.xMax - axes.xMin) * i) / 4
    if (grid) cv.line(img, new cv.Point(x, axes.top), new cv.Point(x, axes.top + axes.height), gridColour, 1)
    cv.putText(img, String(Number(xValue.toPrecision(4))), new cv.Point(x - 14, img.rows - 10), font, 0.3, ink, 1, cv.LINE_AA)
  }

  cv.rectangle(img, new cv.Point(axes.left, axes.top), new cv.Point(axes.left + axes.width, axes.top + axes.height), ink, 1)
  return axes
}

function project(axes: Axes, x: number, y: number): [number, number] {
  return [
    axes.left + Math.round(((x - axes.xMin) / (axes.xMax - axes.xMin)) * axes.width),
    axes.top + axes.height - Math.round(((y - axes.yMin) / (axes.yMax - axes.yMin)) * axes.height),
  ]
}

/** Evenly spaced categorical colours, so classes stay distinguishable. */
function classColour(cv: any, index: number, total: number): any {
  const t = total > 1 ? index / (total - 1) : 0
  const [r, g, b] = viridisColor(Math.round(t * 255))
  return new cv.Scalar(b, g, r, 255)
}

export const mlScatterPlot: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const df = tableIn(inputs)
  if (!df) return {}
  const meta = { shape: [df.rows.length, df.columns.length], columns: [...df.columns] }

  const numeric = numericColumns(df)
  const xCol = resolveColumn(df, params.x_col) ?? numeric[0]
  const yCol = resolveColumn(df, params.y_col) ?? numeric[1] ?? numeric[0]
  if (!xCol || !yCol) return { df_meta: meta }

  const [w, h] = previewSize(inputs.img_size, { width: 540, height: 400, ...params })
  const img = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(PLOT_BG[0], PLOT_BG[1], PLOT_BG[2], 255)))

  let rows = df.rows.filter((r) => Number.isFinite(Number(r[xCol])) && Number.isFinite(Number(r[yCol])))
  const maxPoints = Math.round(Number(params.max_points) ?? 2000)
  if (maxPoints > 0 && rows.length > maxPoints) {
    // Decimate evenly so the visible distribution is unchanged.
    const stride = Math.ceil(rows.length / maxPoints)
    rows = rows.filter((_, i) => i % stride === 0)
  }
  if (!rows.length) return { df_meta: meta, main: img }

  const xs = rows.map((r) => Number(r[xCol]))
  const ys = rows.map((r) => Number(r[yCol]))
  const axes = drawAxes(cv, img, Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys), params.grid !== false, `${yCol} vs ${xCol}`)

  const hueCol = resolveColumn(df, params.hue_col)
  const classes = hueCol ? [...new Set(rows.map((r) => String(r[hueCol])))].sort() : []
  const radius = Math.max(1, Math.round(Math.sqrt(Number(params.dot_size) || 40) / 2))

  rows.forEach((row, i) => {
    const [px, py] = project(axes, xs[i], ys[i])
    const colour = hueCol ? classColour(cv, classes.indexOf(String(row[hueCol])), classes.length) : new cv.Scalar(230, 160, 60, 255)
    cv.circle(img, new cv.Point(px, py), radius, colour, -1, cv.LINE_AA)
  })

  if (params.regression) {
    // Ordinary least squares, drawn across the full x range.
    const n = xs.length
    const meanX = xs.reduce((a, b) => a + b, 0) / n
    const meanY = ys.reduce((a, b) => a + b, 0) / n
    let sxy = 0
    let sxx = 0
    for (let i = 0; i < n; i++) {
      sxy += (xs[i] - meanX) * (ys[i] - meanY)
      sxx += (xs[i] - meanX) ** 2
    }
    if (sxx > 0) {
      const slope = sxy / sxx
      const intercept = meanY - slope * meanX
      const a = project(axes, axes.xMin, slope * axes.xMin + intercept)
      const b = project(axes, axes.xMax, slope * axes.xMax + intercept)
      cv.line(img, new cv.Point(a[0], a[1]), new cv.Point(b[0], b[1]), new cv.Scalar(80, 220, 80, 255), 2, cv.LINE_AA)
    }
  }

  if (classes.length) {
    classes.slice(0, 8).forEach((name, i) => {
      const y = axes.top + 12 + i * 14
      cv.circle(img, new cv.Point(w - 74, y - 4), 4, classColour(cv, i, classes.length), -1)
      cv.putText(img, name.slice(0, 9), new cv.Point(w - 64, y), cv.FONT_HERSHEY_SIMPLEX, 0.32, new cv.Scalar(PLOT_INK[0], PLOT_INK[1], PLOT_INK[2], 255), 1, cv.LINE_AA)
    })
  }

  return { main: img, df_meta: meta }
}

export const mlHistogram: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const df = tableIn(inputs)
  if (!df) return {}

  const numeric = numericColumns(df)
  const column = resolveColumn(df, params.column) ?? numeric[0]
  if (!column) return {}

  const [w, h] = previewSize(inputs.img_size, { width: 540, height: 380, ...params })
  const img = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(PLOT_BG[0], PLOT_BG[1], PLOT_BG[2], 255)))

  const bins = Math.max(5, Math.round(Number(params.bins) || 30))
  const hueCol = resolveColumn(df, params.hue_col)
  const groups = hueCol ? [...new Set(df.rows.map((r) => String(r[hueCol])))].sort() : ['']

  const all = asNumbers(df, column)
  if (!all.length) return { main: img }
  const lo = Math.min(...all)
  const hi = Math.max(...all)
  const span = hi - lo || 1

  const counts = groups.map((group) => {
    const bucket = new Float64Array(bins)
    for (const row of df.rows) {
      if (hueCol && String(row[hueCol]) !== group) continue
      const v = Number(row[column])
      if (!Number.isFinite(v)) continue
      bucket[Math.min(bins - 1, Math.floor(((v - lo) / span) * bins))]++
    }
    if (params.density) {
      // Normalise to a density: the bars then integrate to one.
      const total = bucket.reduce((a, b) => a + b, 0) * (span / bins)
      if (total > 0) for (let i = 0; i < bins; i++) bucket[i] /= total
    }
    return bucket
  })

  let peak = 0
  for (const bucket of counts) for (const v of bucket) if (v > peak) peak = v
  const axes = drawAxes(cv, img, lo, hi, 0, peak, params.grid !== false, column)

  const barWidth = Math.max(1, Math.floor(axes.width / bins / groups.length))
  counts.forEach((bucket, g) => {
    const colour = groups.length > 1 ? classColour(cv, g, groups.length) : new cv.Scalar(230, 160, 60, 255)
    for (let i = 0; i < bins; i++) {
      if (bucket[i] <= 0) continue
      const x0 = axes.left + Math.round((axes.width * i) / bins) + g * barWidth
      const [, y0] = project(axes, lo, bucket[i])
      cv.rectangle(img, new cv.Point(x0, y0), new cv.Point(x0 + barWidth, axes.top + axes.height), colour, -1)
    }
  })

  if (params.kde !== false) {
    // Gaussian KDE with Silverman's rule, drawn over the histogram.
    const n = all.length
    const mean = all.reduce((a, b) => a + b, 0) / n
    const sd = Math.sqrt(all.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, n - 1)) || span / 6
    const bandwidth = 1.06 * sd * Math.pow(n, -1 / 5) || span / 20
    const samples = 120
    let previous: [number, number] | null = null
    let maxDensity = 0
    const density = new Float64Array(samples)
    for (let s = 0; s < samples; s++) {
      const x = lo + (span * s) / (samples - 1)
      let sum = 0
      for (const v of all) {
        const z = (x - v) / bandwidth
        sum += Math.exp(-0.5 * z * z)
      }
      density[s] = sum / (n * bandwidth * Math.sqrt(2 * Math.PI))
      if (density[s] > maxDensity) maxDensity = density[s]
    }
    const scale = params.density ? 1 : maxDensity > 0 ? peak / maxDensity : 0
    for (let s = 0; s < samples; s++) {
      const x = lo + (span * s) / (samples - 1)
      const point = project(axes, x, density[s] * scale)
      if (previous) cv.line(img, new cv.Point(previous[0], previous[1]), new cv.Point(point[0], point[1]), new cv.Scalar(120, 220, 250, 255), 2, cv.LINE_AA)
      previous = point
    }
  }

  return { main: img }
}

/** Ranks with ties averaged, which is what Spearman's correlation needs. */
function rank(values: number[]): number[] {
  const order = values.map((v, i) => [v, i] as [number, number]).sort((a, b) => a[0] - b[0])
  const ranks = new Array<number>(values.length)
  let i = 0
  while (i < order.length) {
    let j = i
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++
    const average = (i + j) / 2 + 1
    for (let k = i; k <= j; k++) ranks[order[k][1]] = average
    i = j + 1
  }
  return ranks
}

function pearson(a: number[], b: number[]): number {
  const n = a.length
  if (n < 2) return 0
  const meanA = a.reduce((x, y) => x + y, 0) / n
  const meanB = b.reduce((x, y) => x + y, 0) / n
  let cov = 0
  let va = 0
  let vb = 0
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA
    const db = b[i] - meanB
    cov += da * db
    va += da * da
    vb += db * db
  }
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0
}

/** Kendall's tau-b, which corrects for ties in either variable. */
function kendall(a: number[], b: number[]): number {
  const n = a.length
  let concordant = 0
  let discordant = 0
  let tiesA = 0
  let tiesB = 0
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const da = Math.sign(a[i] - a[j])
      const db = Math.sign(b[i] - b[j])
      if (da === 0 && db === 0) continue
      if (da === 0) tiesA++
      else if (db === 0) tiesB++
      else if (da === db) concordant++
      else discordant++
    }
  }
  const denominator = Math.sqrt((concordant + discordant + tiesA) * (concordant + discordant + tiesB))
  return denominator > 0 ? (concordant - discordant) / denominator : 0
}

const CORR_COLOURS = [jetColor, viridisColor, viridisColor, plasmaColor]

export const mlCorrHeatmap: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const df = tableIn(inputs)
  if (!df) return {}

  const numeric = numericColumns(df)
  const requested = splitList(params.columns).filter((c) => numeric.includes(c))
  const cols = requested.length ? requested : numeric
  if (cols.length < 2) return {}

  const method = ['pearson', 'spearman', 'kendall'][Math.round(Number(params.method) || 0)] ?? 'pearson'
  // Only rows where every selected column is present take part, matching
  // pandas' pairwise-complete default closely enough for a display matrix.
  const usable = df.rows.filter((row) => cols.every((c) => Number.isFinite(Number(row[c]))))
  const series = cols.map((c) => usable.map((row) => Number(row[c])))
  const prepared = method === 'spearman' ? series.map(rank) : series

  const matrix = cols.map((_, i) =>
    cols.map((__, j) => (method === 'kendall' ? kendall(prepared[i], prepared[j]) : pearson(prepared[i], prepared[j])))
  )

  const [w, h] = previewSize(inputs.img_size, { width: 520, height: 480, ...params })
  const img = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(PLOT_BG[0], PLOT_BG[1], PLOT_BG[2], 255)))

  const labelWidth = 74
  const cell = Math.max(12, Math.floor(Math.min((w - labelWidth - 10) / cols.length, (h - labelWidth - 10) / cols.length)))
  const originX = labelWidth
  const originY = 20
  const colour = CORR_COLOURS[Math.min(3, Math.max(0, Math.round(Number(params.colormap) || 0)))]
  const font = cv.FONT_HERSHEY_SIMPLEX
  const ink = new cv.Scalar(PLOT_INK[0], PLOT_INK[1], PLOT_INK[2], 255)

  for (let i = 0; i < cols.length; i++) {
    for (let j = 0; j < cols.length; j++) {
      // Correlation runs [-1, 1]; map it onto the colormap's full range.
      const [r, g, b] = colour(Math.round(((matrix[i][j] + 1) / 2) * 255))
      const x = originX + j * cell
      const y = originY + i * cell
      cv.rectangle(img, new cv.Point(x, y), new cv.Point(x + cell, y + cell), new cv.Scalar(b, g, r, 255), -1)
      if (params.annot !== false && cell >= 26) {
        const text = matrix[i][j].toFixed(2)
        const luminance = 0.299 * r + 0.587 * g + 0.114 * b
        cv.putText(img, text, new cv.Point(x + 2, y + cell / 2 + 4), font, 0.3, new cv.Scalar(luminance > 140 ? 0 : 255, luminance > 140 ? 0 : 255, luminance > 140 ? 0 : 255, 255), 1, cv.LINE_AA)
      }
    }
    cv.putText(img, cols[i].slice(0, 10), new cv.Point(2, originY + i * cell + cell / 2 + 4), font, 0.32, ink, 1, cv.LINE_AA)
    cv.putText(img, cols[i].slice(0, 6), new cv.Point(originX + i * cell + 2, originY + cols.length * cell + 14), font, 0.3, ink, 1, cv.LINE_AA)
  }

  void cellText
  void applyColormap
  return { main: img, matrix: matrix.map((row) => row.map((v) => Math.round(v * 1e4) / 1e4)), columns: cols }
}
