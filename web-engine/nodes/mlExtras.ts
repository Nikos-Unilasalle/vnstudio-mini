/**
 * The Sklearn Dataset loader, the universal Bar Chart, and the SVM classifier.
 */
import type { NodeImpl, RunContext } from '../types'
import DATASETS from '../datasets.json'
import { DataFrame, dfMeta, isDf, isNumericColumn, makeDf, previewSize, renderDfTable, resolveColumn, splitList } from '../dataframe'
import { parseCsv } from '../csv'
import { listTextFiles, readTextFile } from '../textFiles'
import { classificationReport, confusionMatrix, makeMatrix, Matrix } from '../ml'
import { Kernel, svmFit, svmPredict, svmPredictOne, supportVectorCount } from '../svm'
import { Axes, classColour, drawAxes, PLOT_BG, PLOT_INK, project } from './mlData'

function canvas(cv: any, ctx: RunContext, w: number, h: number): any {
  return ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(PLOT_BG[0], PLOT_BG[1], PLOT_BG[2], 255)))
}

function text(cv: any, img: any, s: string, x: number, y: number, scale = 0.38, colour?: any): void {
  cv.putText(img, s, new cv.Point(x, y), cv.FONT_HERSHEY_SIMPLEX, scale,
    colour ?? new cv.Scalar(PLOT_INK[0], PLOT_INK[1], PLOT_INK[2], 255), 1, cv.LINE_AA)
}

/** The dark info panel the DataFrame nodes share. */
function infoPanel(cv: any, ctx: RunContext, lines: string[], w: number, h: number, title: string): any {
  const img = canvas(cv, ctx, w, h)
  cv.rectangle(img, new cv.Point(0, 0), new cv.Point(w, 28), new cv.Scalar(45, 45, 45, 255), -1)
  text(cv, img, title, 8, 19, 0.46)
  cv.line(img, new cv.Point(0, 28), new cv.Point(w, 28), new cv.Scalar(80, 80, 80, 255), 1)
  const lineHeight = 15
  lines.slice(0, Math.floor((h - 36) / lineHeight)).forEach((line, i) => {
    text(cv, img, line.slice(0, 68), 8, 44 + i * lineHeight, 0.37,
      i === 0 ? new cv.Scalar(255, 200, 140, 255) : new cv.Scalar(185, 185, 185, 255))
  })
  return img
}

/* ------------------------------------------------------------ sklearn data */

type Bundle = Record<string, { columns: string[]; data: Record<string, (number | string)[]> }>
const BUNDLE = DATASETS as Bundle

// The desktop offers six; digits and california_housing are not bundled — the
// first would add a quarter of a megabyte to the download, the second is
// fetched from the network at runtime, which a browser build cannot do offline.
const DATASET_KEYS = ['iris', 'wine', 'breast_cancer', 'diabetes', 'digits', 'california_housing']

export const mlSklearnDataset: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const index = Math.round(Number(params.dataset ?? 0))
  const key = DATASET_KEYS[index] ?? 'iris'
  const [w, h] = previewSize(inputs.img_size, { width: 320, height: 200, ...params })

  const bundled = BUNDLE[key]
  if (!bundled) {
    return {
      main: infoPanel(cv, ctx, [
        `"${key}" is not bundled in the web build.`,
        '',
        key === 'digits'
          ? 'digits would add ~250 KB to the download.'
          : 'california_housing is fetched over the network.',
        '',
        'Available: iris, wine, breast_cancer, diabetes.',
      ], w, h, 'sklearn - unavailable'),
      row_count: 0,
      col_count: 0,
    }
  }

  // Stored column-major; the DataFrame is row-major, so it is built once and
  // cached — the frame never changes, and rebuilding 569 rows every tick would
  // be pure waste.
  const cacheKey = `${ctx.nodeId}:dataset:${key}`
  let df = ctx.state.get(cacheKey) as DataFrame | undefined
  if (!df) {
    const rowCount = bundled.data[bundled.columns[0]]?.length ?? 0
    const rows: Record<string, unknown>[] = []
    for (let i = 0; i < rowCount; i++) {
      const record: Record<string, unknown> = {}
      for (const column of bundled.columns) record[column] = bundled.data[column][i]
      rows.push(record)
    }
    df = makeDf(bundled.columns, rows)
    ctx.state.set(cacheKey, df)
  }

  const numeric = df.columns.filter((c) => isNumericColumn(df!, c))
  const targets = new Set(df.rows.map((r) => String(r.target)))
  const lines = [
    `Dataset : ${key}`,
    `Shape   : ${df.rows.length} rows x ${df.columns.length} cols`,
    `Target  : target (${targets.size} unique values)`,
    `Features: ${numeric.length} numeric`,
    '',
    ...df.columns.slice(0, 12).map((c) => `  ${c}`),
  ]
  if (df.columns.length > 12) lines.push(`  ... +${df.columns.length - 12} more`)

  return {
    table: df,
    main: infoPanel(cv, ctx, lines, w, h, `sklearn - ${key}`),
    preview: infoPanel(cv, ctx, lines, w, h, `sklearn - ${key}`),
    row_count: df.rows.length,
    col_count: df.columns.length,
    img_size: [w, h],
    df_meta: dfMeta(df),
  }
}

/* --------------------------------------------------------------- CSV reader */

const SEPARATORS = [',', ';', '\t', '|']

export const mlCsvReader: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const path = String(params.path ?? 'data.csv').trim()
  const separator = SEPARATORS[Math.round(Number(params.separator ?? 0))] ?? ','
  const maxRows = Math.round(Number(params.max_rows ?? 0))
  const [w, h] = previewSize(inputs.img_size, { width: 420, height: 240, ...params })

  const text = readTextFile(path)
  if (text === null) {
    const known = listTextFiles()
    return {
      main: infoPanel(cv, ctx, [
        `File not found: ${path}`,
        '',
        'Drop a .csv onto the window, or open one from the',
        'file dialog, then set this path to match.',
        ...(known.length > 0 ? ['', 'Loaded files:', ...known.slice(0, 6).map((f) => `  ${f}`)] : []),
      ], w, h, 'CSV Reader'),
      row_count: 0,
      col_count: 0,
    }
  }

  // Reparsing a large file on every frame would dominate the run, so the
  // result is cached until the path, separator, row cap or contents change.
  const cacheKey = `${ctx.nodeId}:csv`
  const signature = `${path}|${separator}|${maxRows}|${text.length}`
  let cached = ctx.state.get(cacheKey) as { signature: string; df: DataFrame } | undefined
  if (!cached || cached.signature !== signature) {
    cached = { signature, df: parseCsv(text, { separator, maxRows }) }
    ctx.state.set(cacheKey, cached)
  }
  const df = cached.df
  const name = path.split('/').pop() || path

  return {
    table: df,
    main: ctx.track(renderDfTable(cv, df, w, h, name)),
    preview: ctx.track(renderDfTable(cv, df, w, h, name)),
    row_count: df.rows.length,
    col_count: df.columns.length,
    df_meta: dfMeta(df),
    img_size: [w, h],
  }
}

/* ---------------------------------------------------------------- bar chart */

/** Groups by category and averages, the way the desktop's `groupby().mean()` does. */
function groupMean(df: DataFrame, categoryCol: string, valueCol: string): { categories: string[]; values: number[] } {
  const sums = new Map<string, { total: number; count: number }>()
  for (const record of df.rows) {
    const key = String(record[categoryCol])
    const value = Number(record[valueCol])
    if (!Number.isFinite(value)) continue
    const entry = sums.get(key)
    if (entry) { entry.total += value; entry.count++ }
    else sums.set(key, { total: value, count: 1 })
  }
  // pandas' groupby sorts its keys.
  const categories = [...sums.keys()].sort()
  return { categories, values: categories.map((c) => sums.get(c)!.total / sums.get(c)!.count) }
}

/** Matplotlib's `%.3g`, which is what the value labels use. */
function threeSignificant(value: number): string {
  if (!Number.isFinite(value)) return String(value)
  if (value === 0) return '0'
  const formatted = value.toPrecision(3)
  // toPrecision keeps trailing zeros and %g does not.
  return String(Number(formatted))
}

export const mlBarChart: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const table = inputs.table
  const df = isDf(table) ? table : null
  const dictB = inputs.data_b && typeof inputs.data_b === 'object' && !Array.isArray(inputs.data_b)
    ? (inputs.data_b as Record<string, unknown>)
    : null

  let categories: string[] = []
  let valuesA: number[] = []
  let valuesB: number[] | null = null

  if (!df) {
    // Dict mode: `table` may itself carry a plain {label: value} mapping.
    const dictA = table && typeof table === 'object' && !Array.isArray(table) ? (table as Record<string, unknown>) : {}
    const merged = { ...dictA, ...(dictB ?? {}) }
    const keys = Object.keys(merged)
    if (keys.length === 0) return {}
    // The desktop sorts the merged keys by value, descending.
    keys.sort((a, b) => (Number(merged[b]) || 0) - (Number(merged[a]) || 0))
    categories = keys.map(String)
    valuesA = keys.map((k) => Number(dictA[k] ?? 0) || 0)
    if (dictB && Object.keys(dictB).length > 0) valuesB = keys.map((k) => Number(dictB[k] ?? 0) || 0)
  } else {
    const categoryCol = resolveColumn(df, params.x_col) ?? df.columns[0]
    const numeric = df.columns.filter((c) => isNumericColumn(df, c) && c !== categoryCol)
    const valueCol = resolveColumn(df, params.y_col) ?? numeric[0] ?? df.columns[df.columns.length - 1]
    const grouped = groupMean(df, categoryCol, valueCol)
    categories = grouped.categories
    valuesA = grouped.values
    if (dictB && Object.keys(dictB).length > 0) valuesB = categories.map((c) => Number(dictB[c] ?? 0) || 0)
  }
  if (categories.length === 0) return {}

  if (params.sorted) {
    const order = [...categories.keys()].sort((a, b) => valuesA[b] - valuesA[a])
    categories = order.map((i) => categories[i])
    const sortedA = order.map((i) => valuesA[i])
    if (valuesB) valuesB = order.map((i) => valuesB![i])
    valuesA = sortedA
  }

  const horizontal = Math.round(Number(params.orientation ?? 0)) === 1
  const showValues = params.show_values !== false
  const barWidth = Number(params.bar_width ?? 0.75)
  const title = String(params.title ?? '').trim()
  const count = categories.length
  const [w, h] = previewSize(inputs.img_size, {
    width: 540,
    height: horizontal ? Math.max(320, count * 28) : 400,
    ...params,
  })
  const img = canvas(cv, ctx, w, h)

  const allValues = valuesB ? [...valuesA, ...valuesB] : valuesA
  const lowest = Math.min(0, ...allValues)
  const highest = Math.max(...allValues, lowest + 1e-9)
  const headroom = (highest - lowest) * 0.12

  let axes: Axes
  if (horizontal) {
    axes = drawAxes(cv, img, lowest, highest + headroom, -0.5, count - 0.5, true, title)
  } else {
    axes = drawAxes(cv, img, -0.5, count - 0.5, lowest, highest + headroom, true, title)
  }

  const slot = horizontal ? axes.height / count : axes.width / count
  const thickness = Math.max(2, Math.round(slot * barWidth * (valuesB ? 0.45 : 1)))
  const seriesColours = [new cv.Scalar(246, 130, 59, 255), new cv.Scalar(22, 115, 249, 255)]

  const drawBar = (position: number, value: number, offset: number, colour: any) => {
    if (horizontal) {
      // Category axis runs bottom-to-top, so index 0 sits at the bottom.
      const [zeroX] = project(axes, Math.max(0, lowest), count - 1 - position)
      const [tipX, centreY] = project(axes, value, count - 1 - position)
      const y = centreY + offset
      cv.rectangle(img, new cv.Point(Math.min(zeroX, tipX), y - thickness / 2),
        new cv.Point(Math.max(zeroX, tipX), y + thickness / 2), colour, -1)
      if (showValues) text(cv, img, threeSignificant(value), Math.max(zeroX, tipX) + 4, y + 4, 0.3)
    } else {
      const [, zeroY] = project(axes, position, Math.max(0, lowest))
      const [centreX, tipY] = project(axes, position, value)
      const x = centreX + offset
      cv.rectangle(img, new cv.Point(x - thickness / 2, Math.min(zeroY, tipY)),
        new cv.Point(x + thickness / 2, Math.max(zeroY, tipY)), colour, -1)
      if (showValues) text(cv, img, threeSignificant(value), x - 10, Math.min(zeroY, tipY) - 4, 0.3)
    }
  }

  categories.forEach((label, i) => {
    if (valuesB) {
      const gap = Math.round(thickness / 2) + 1
      drawBar(i, valuesA[i], -gap, seriesColours[0])
      drawBar(i, valuesB[i], gap, seriesColours[1])
    } else {
      drawBar(i, valuesA[i], 0, classColour(cv, i, count))
    }
    // Category ticks, which drawAxes writes numerically for a numeric axis.
    if (horizontal) {
      const [, y] = project(axes, lowest, count - 1 - i)
      text(cv, img, label.slice(0, 9), 2, y + 4, 0.3)
    } else {
      const [x] = project(axes, i, lowest)
      text(cv, img, label.slice(0, 7), x - 18, h - 18, 0.3)
    }
  })

  if (valuesB) {
    const labels = [String(params.dict_key_a ?? 'Series A'), String(params.dict_key_b ?? 'Series B')]
    labels.forEach((label, i) => {
      const y = axes.top + 12 + i * 14
      cv.rectangle(img, new cv.Point(axes.left + axes.width - 100, y - 8),
        new cv.Point(axes.left + axes.width - 92, y), seriesColours[i], -1)
      text(cv, img, label.slice(0, 14), axes.left + axes.width - 88, y, 0.3)
    })
  }

  const xLabel = String(params.xlabel ?? '').trim()
  const yLabel = String(params.ylabel ?? '').trim()
  if (xLabel) text(cv, img, xLabel, axes.left + axes.width / 2 - 20, h - 4, 0.34)
  if (yLabel) text(cv, img, yLabel, 4, axes.top - 4, 0.34)

  return { main: img }
}

/* --------------------------------------------------------------------- SVM */

const KERNELS: Kernel[] = ['rbf', 'linear', 'poly', 'sigmoid']

function encodeLabels(values: unknown[]): { codes: Int32Array; classes: string[] } {
  const classes = [...new Set(values.map((v) => String(v)))].sort()
  const index = new Map(classes.map((c, i) => [c, i]))
  const codes = new Int32Array(values.length)
  values.forEach((v, i) => { codes[i] = index.get(String(v)) ?? 0 })
  return { codes, classes }
}

function featureRows(df: DataFrame, features: string[], target: string, encoder: Map<string, number>): { X: Matrix; y: Int32Array } {
  const rows: number[][] = []
  const labels: number[] = []
  for (const record of df.rows) {
    const values = features.map((f) => Number(record[f]))
    const code = encoder.get(String(record[target]))
    if (values.every((v) => Number.isFinite(v)) && code !== undefined) {
      rows.push(values)
      labels.push(code)
    }
  }
  return { X: makeMatrix(rows), y: Int32Array.from(labels) }
}

function standardizeWith(rows: number[][]): { mean: number[]; scale: number[] } {
  const d = rows[0]?.length ?? 0
  const mean = new Array(d).fill(0)
  const scale = new Array(d).fill(1)
  for (let j = 0; j < d; j++) {
    let sum = 0
    for (const r of rows) sum += r[j]
    mean[j] = rows.length > 0 ? sum / rows.length : 0
    let variance = 0
    for (const r of rows) variance += (r[j] - mean[j]) ** 2
    variance = rows.length > 0 ? variance / rows.length : 0
    scale[j] = variance > 1e-12 ? Math.sqrt(variance) : 1
  }
  return { mean, scale }
}

export const mlSvmClassifier: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const trainDf = isDf(inputs.train) ? inputs.train : isDf(inputs.table) ? inputs.table : null
  const testDf = isDf(inputs.test) ? inputs.test : null
  if (!trainDf) return {}

  const target = resolveColumn(trainDf, params.target) ?? trainDf.columns[trainDf.columns.length - 1]
  const requested = splitList(params.features)
    .map((name) => resolveColumn(trainDf, name))
    .filter((name): name is string => name !== null && name !== target && isNumericColumn(trainDf, name))
  const features = requested.length > 0
    ? requested
    : trainDf.columns.filter((c) => isNumericColumn(trainDf, c) && c !== target)
  if (features.length === 0) return {}

  const encoded = encodeLabels([...trainDf.rows, ...(testDf?.rows ?? [])].map((r) => r[target]))
  const encoder = new Map(encoded.classes.map((c, i) => [c, i]))
  let train = featureRows(trainDf, features, target, encoder)
  let test = testDf ? featureRows(testDf, features, target, encoder) : { X: makeMatrix([]), y: new Int32Array(0) }
  if (train.X.n === 0) return {}

  if (params.standardize !== false) {
    const asRows = (m: Matrix) => Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, i * m.d + m.d)))
    const trainRows = asRows(train.X)
    const { mean, scale } = standardizeWith(trainRows)
    const apply = (rows: number[][]) => makeMatrix(rows.map((r) => r.map((v, j) => (v - mean[j]) / scale[j])))
    train = { X: apply(trainRows), y: train.y }
    if (test.X.n > 0) test = { X: apply(asRows(test.X)), y: test.y }
  }

  const model = svmFit(train.X, train.y, encoded.classes.length, {
    C: Number(params.C ?? 1),
    kernel: KERNELS[Math.round(Number(params.kernel ?? 0))] ?? 'rbf',
    gamma: Math.round(Number(params.gamma ?? 0)) === 1 ? 'auto' : 'scale',
    degree: Math.round(Number(params.degree ?? 3)),
  })

  const trainPredictions = svmPredict(model, train.X)
  let trainCorrect = 0
  for (let i = 0; i < train.y.length; i++) if (trainPredictions[i] === train.y[i]) trainCorrect++
  const trainAccuracy = train.y.length > 0 ? trainCorrect / train.y.length : 0

  const testPredictions = test.X.n > 0 ? svmPredict(model, test.X) : new Int32Array(0)
  let testCorrect = 0
  for (let i = 0; i < test.y.length; i++) if (testPredictions[i] === test.y[i]) testCorrect++
  const testAccuracy = test.y.length > 0 ? testCorrect / test.y.length : 0

  const report = test.X.n > 0 ? classificationReport(test.y, testPredictions, encoded.classes) : {}
  const kernelName = KERNELS[Math.round(Number(params.kernel ?? 0))] ?? 'rbf'
  const modelName = `SVM (${kernelName}, C=${Number(params.C ?? 1)})`

  const [w, h] = previewSize(inputs.img_size, { width: 540, height: 420, ...params })
  const preview = canvas(cv, ctx, w, h)

  if (features.length === 2) {
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity
    const consider = (m: Matrix) => {
      for (let i = 0; i < m.n; i++) {
        xMin = Math.min(xMin, m.data[i * 2]); xMax = Math.max(xMax, m.data[i * 2])
        yMin = Math.min(yMin, m.data[i * 2 + 1]); yMax = Math.max(yMax, m.data[i * 2 + 1])
      }
    }
    consider(train.X)
    if (test.X.n > 0) consider(test.X)
    const marginX = (xMax - xMin) * 0.12 + 1e-6
    const marginY = (yMax - yMin) * 0.12 + 1e-6
    const axes = drawAxes(cv, preview, xMin - marginX, xMax + marginX, yMin - marginY, yMax + marginY, true,
      `${modelName}  test acc = ${(testAccuracy * 100).toFixed(1)}%`)

    // Decision regions on a coarse grid; each cell costs a full kernel sweep.
    const steps = Math.max(8, Math.min(Math.round(Number(params.boundary_res ?? 120)), 90))
    const point = new Float64Array(2)
    for (let gy = 0; gy < steps; gy++) {
      for (let gx = 0; gx < steps; gx++) {
        point[0] = axes.xMin + ((gx + 0.5) / steps) * (axes.xMax - axes.xMin)
        point[1] = axes.yMax - ((gy + 0.5) / steps) * (axes.yMax - axes.yMin)
        const colour = classColour(cv, svmPredictOne(model, point), encoded.classes.length)
        const faded = new cv.Scalar(
          PLOT_BG[0] + (colour[0] - PLOT_BG[0]) * 0.22,
          PLOT_BG[1] + (colour[1] - PLOT_BG[1]) * 0.22,
          PLOT_BG[2] + (colour[2] - PLOT_BG[2]) * 0.22,
          255
        )
        cv.rectangle(preview,
          new cv.Point(Math.round(axes.left + (gx * axes.width) / steps), Math.round(axes.top + (gy * axes.height) / steps)),
          new cv.Point(Math.round(axes.left + ((gx + 1) * axes.width) / steps), Math.round(axes.top + ((gy + 1) * axes.height) / steps)),
          faded, -1)
      }
    }
    cv.rectangle(preview, new cv.Point(axes.left, axes.top), new cv.Point(axes.left + axes.width, axes.top + axes.height),
      new cv.Scalar(PLOT_INK[0], PLOT_INK[1], PLOT_INK[2], 255), 1)

    // The support vectors are ringed, which is the whole point of the picture.
    const supports = new Set<number>()
    for (const pair of model.pairs) for (const index of pair.model.supportIndices) supports.add(index)
    for (let i = 0; i < train.X.n; i++) {
      const [px, py] = project(axes, train.X.data[i * 2], train.X.data[i * 2 + 1])
      cv.circle(preview, new cv.Point(px, py), 3, classColour(cv, train.y[i], encoded.classes.length), -1)
      if (supports.has(i)) cv.circle(preview, new cv.Point(px, py), 6, new cv.Scalar(200, 200, 200, 255), 1)
    }
    for (let i = 0; i < test.X.n; i++) {
      const [px, py] = project(axes, test.X.data[i * 2], test.X.data[i * 2 + 1])
      cv.circle(preview, new cv.Point(px, py), 5, classColour(cv, test.y[i], encoded.classes.length), -1)
      cv.circle(preview, new cv.Point(px, py), 5, new cv.Scalar(255, 255, 255, 255), 1)
    }
    encoded.classes.slice(0, 10).forEach((label, i) => {
      const y = axes.top + 12 + i * 13
      cv.circle(preview, new cv.Point(axes.left + axes.width - 96, y - 3), 3, classColour(cv, i, encoded.classes.length), -1)
      text(cv, preview, label.slice(0, 12), axes.left + axes.width - 88, y, 0.32)
    })
  } else {
    // More than two features: the confusion matrix, as the desktop falls back to.
    const cm = confusionMatrix(test.y, testPredictions, encoded.classes.length)
    text(cv, preview, `${modelName}  acc = ${(testAccuracy * 100).toFixed(1)}%`, 8, 18, 0.42)
    const n = Math.max(1, cm.length)
    const left = 70
    const top = 30
    const cell = Math.max(16, Math.min(Math.floor((w - left - 12) / n), Math.floor((h - top - 34) / n)))
    for (let i = 0; i < n; i++) {
      const rowTotal = cm[i].reduce((a, b) => a + b, 0)
      for (let j = 0; j < n; j++) {
        const value = rowTotal > 0 ? cm[i][j] / rowTotal : 0
        const shade = new cv.Scalar(255 - value * 100, 255 - value * 180, 255 - value * 235, 255)
        const x0 = left + j * cell
        const y0 = top + i * cell
        cv.rectangle(preview, new cv.Point(x0, y0), new cv.Point(x0 + cell, y0 + cell), shade, -1)
        cv.rectangle(preview, new cv.Point(x0, y0), new cv.Point(x0 + cell, y0 + cell), new cv.Scalar(70, 70, 70, 255), 1)
        text(cv, preview, value.toFixed(2), x0 + 2, y0 + cell / 2 + 4, 0.3,
          value > 0.5 ? new cv.Scalar(255, 255, 255, 255) : new cv.Scalar(51, 51, 51, 255))
      }
      text(cv, preview, encoded.classes[i].slice(0, 9), 4, top + i * cell + cell / 2 + 4, 0.32)
    }
  }

  // The report table, shared in layout with the KNN and forest nodes.
  const reportKeys = Object.keys(report).filter(
    (k) => k !== 'accuracy' && k !== 'macro avg' && k !== 'weighted avg' && typeof (report as any)[k] === 'object'
  )
  const lines = [...reportKeys, 'macro avg', 'weighted avg'].filter((k) => (report as any)[k])
  const reportWidth = Math.max(w, 420)
  const reportImg = canvas(cv, ctx, reportWidth, 46 + lines.length * 20)
  cv.rectangle(reportImg, new cv.Point(0, 0), new cv.Point(reportWidth, 24), new cv.Scalar(58, 42, 42, 255), -1)
  text(cv, reportImg, `${modelName}  ·  Accuracy ${(((report as any).accuracy ?? 0) * 100).toFixed(1)}%`, 8, 16, 0.4, new cv.Scalar(252, 180, 165, 255))
  const columns = [8, Math.round(reportWidth * 0.42), Math.round(reportWidth * 0.58), Math.round(reportWidth * 0.74), Math.round(reportWidth * 0.88)]
  ;['Class', 'Prec', 'Recall', 'F1', 'Supp'].forEach((label, i) => text(cv, reportImg, label, columns[i], 40, 0.35, new cv.Scalar(252, 180, 165, 255)))
  lines.forEach((key, i) => {
    const m = (report as any)[key]
    const y = 60 + i * 20
    const f1 = m['f1-score']
    const band = f1 >= 0.9 ? new cv.Scalar(183, 231, 110, 255) : f1 >= 0.7 ? new cv.Scalar(61, 211, 252, 255) : new cv.Scalar(113, 113, 248, 255)
    text(cv, reportImg, key.slice(0, 18), columns[0], y, 0.35)
    text(cv, reportImg, m.precision.toFixed(2), columns[1], y, 0.35)
    text(cv, reportImg, m.recall.toFixed(2), columns[2], y, 0.35)
    text(cv, reportImg, f1.toFixed(2), columns[3], y, 0.35, band)
    text(cv, reportImg, String(m.support), columns[4], y, 0.35)
  })

  return {
    main: preview,
    preview,
    accuracy: testAccuracy,
    train_acc: trainAccuracy,
    report: reportImg,
    report_data: report,
    n_support: supportVectorCount(model),
  }
}
