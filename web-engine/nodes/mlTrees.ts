/**
 * Decision Tree, Random Forest, Robust Line Fit and the Parameter Optimizer.
 *
 * The tree diagram, the importance bars and the scatter plots are drawn with
 * OpenCV; the fitting itself lives in `../tree` and `../ml`.
 */
import type { NodeImpl, RunContext } from '../types'
import { DataFrame, isDf, isNumericColumn, makeDf, previewSize, resolveColumn, splitList } from '../dataframe'
import {
  classificationReport,
  confusionMatrix,
  lineFitHuber,
  lineFitL2,
  lineFitTheilSen,
  makeMatrix,
  Matrix,
} from '../ml'
import {
  Criterion,
  DecisionTree,
  fitForest,
  fitTree,
  forestPredict,
  oobScore,
  treePredict,
} from '../tree'
import { classColour, drawAxes, PLOT_BG, PLOT_INK, project } from './mlData'

function tableIn(inputs: Record<string, unknown>, ...names: string[]): DataFrame | null {
  for (const name of names) {
    const value = inputs[name]
    if (isDf(value)) return value
  }
  return null
}

function canvas(cv: any, ctx: RunContext, w: number, h: number): any {
  return ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(PLOT_BG[0], PLOT_BG[1], PLOT_BG[2], 255)))
}

function text(cv: any, img: any, s: string, x: number, y: number, scale = 0.38, colour?: any): void {
  cv.putText(img, s, new cv.Point(x, y), cv.FONT_HERSHEY_SIMPLEX, scale,
    colour ?? new cv.Scalar(PLOT_INK[0], PLOT_INK[1], PLOT_INK[2], 255), 1, cv.LINE_AA)
}

function pickFeatures(df: DataFrame, raw: unknown, exclude: string): string[] {
  const numeric = df.columns.filter((c) => isNumericColumn(df, c) && c !== exclude)
  const requested = splitList(raw)
    .map((name) => resolveColumn(df, name))
    .filter((name): name is string => name !== null && numeric.includes(name))
  return requested.length > 0 ? requested : numeric
}

function encodeLabels(values: unknown[]): { codes: Int32Array; classes: string[] } {
  const classes = [...new Set(values.map((v) => String(v)))].sort()
  const index = new Map(classes.map((c, i) => [c, i]))
  const codes = new Int32Array(values.length)
  values.forEach((v, i) => { codes[i] = index.get(String(v)) ?? 0 })
  return { codes, classes }
}

/** Feature rows plus their label codes, dropping any row with a missing value. */
function prepare(df: DataFrame, features: string[], target: string, encoder: Map<string, number>): { X: Matrix; y: Int32Array } {
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

function accuracy(predicted: Int32Array, truth: Int32Array): number {
  if (truth.length === 0) return 0
  let correct = 0
  for (let i = 0; i < truth.length; i++) if (predicted[i] === truth[i]) correct++
  return correct / truth.length
}

/** Horizontal importance bars, smallest at the bottom as the desktop draws them. */
function importancePanel(cv: any, ctx: RunContext, features: string[], importances: Float64Array, width: number, topN: number): any {
  const order = [...features.keys()].sort((a, b) => importances[b] - importances[a]).slice(0, Math.max(1, topN))
  const height = Math.max(140, order.length * 22 + 50)
  const img = canvas(cv, ctx, Math.max(320, width), height)
  text(cv, img, 'Feature Importances', 8, 20, 0.42)
  const reach = Math.max(...Array.from(importances), 1e-9)
  const left = 90
  const barWidth = img.cols - left - 60
  order.forEach((featureIndex, position) => {
    const y = 40 + position * 22
    const length = Math.round((importances[featureIndex] / reach) * barWidth)
    cv.rectangle(img, new cv.Point(left, y), new cv.Point(left + Math.max(1, length), y + 14),
      classColour(cv, position, order.length), -1)
    text(cv, img, features[featureIndex].slice(0, 13), 4, y + 12, 0.33)
    text(cv, img, importances[featureIndex].toFixed(3), left + Math.max(1, length) + 4, y + 12, 0.32)
  })
  return img
}

function confusionPanel(cv: any, ctx: RunContext, cm: number[][], classes: string[], width: number, height: number, title: string): any {
  const img = canvas(cv, ctx, width, height)
  text(cv, img, title, 8, 18, 0.42)
  const n = Math.max(1, cm.length)
  const left = 70
  const top = 30
  const cell = Math.max(16, Math.min(Math.floor((width - left - 12) / n), Math.floor((height - top - 34) / n)))
  for (let i = 0; i < n; i++) {
    const rowTotal = cm[i].reduce((a, b) => a + b, 0)
    for (let j = 0; j < n; j++) {
      const value = rowTotal > 0 ? cm[i][j] / rowTotal : 0
      const shade = new cv.Scalar(255 - value * 100, 255 - value * 180, 255 - value * 235, 255)
      const x0 = left + j * cell
      const y0 = top + i * cell
      cv.rectangle(img, new cv.Point(x0, y0), new cv.Point(x0 + cell, y0 + cell), shade, -1)
      cv.rectangle(img, new cv.Point(x0, y0), new cv.Point(x0 + cell, y0 + cell), new cv.Scalar(70, 70, 70, 255), 1)
      text(cv, img, value.toFixed(2), x0 + 2, y0 + cell / 2 + 4, 0.3,
        value > 0.5 ? new cv.Scalar(255, 255, 255, 255) : new cv.Scalar(51, 51, 51, 255))
    }
    text(cv, img, classes[i].slice(0, 9), 4, top + i * cell + cell / 2 + 4, 0.32)
    text(cv, img, classes[i].slice(0, 5), left + i * cell + 2, top + n * cell + 14, 0.32)
  }
  return img
}

function reportPanel(cv: any, ctx: RunContext, report: Record<string, any>, width: number, title: string): any {
  const classes = Object.keys(report).filter(
    (k) => k !== 'accuracy' && k !== 'macro avg' && k !== 'weighted avg' && typeof report[k] === 'object'
  )
  const lines = [...classes, 'macro avg', 'weighted avg'].filter((k) => report[k])
  const img = canvas(cv, ctx, width, 46 + lines.length * 20)
  cv.rectangle(img, new cv.Point(0, 0), new cv.Point(width, 24), new cv.Scalar(58, 42, 42, 255), -1)
  text(cv, img, `${title}  ·  Accuracy ${((Number(report.accuracy) || 0) * 100).toFixed(1)}%`, 8, 16, 0.4, new cv.Scalar(252, 180, 165, 255))
  const columns = [8, Math.round(width * 0.42), Math.round(width * 0.58), Math.round(width * 0.74), Math.round(width * 0.88)]
  ;['Class', 'Prec', 'Recall', 'F1', 'Supp'].forEach((label, i) => text(cv, img, label, columns[i], 40, 0.35, new cv.Scalar(252, 180, 165, 255)))
  lines.forEach((key, i) => {
    const m = report[key]
    const y = 60 + i * 20
    const f1 = m['f1-score']
    const band = f1 >= 0.9 ? new cv.Scalar(183, 231, 110, 255) : f1 >= 0.7 ? new cv.Scalar(61, 211, 252, 255) : new cv.Scalar(113, 113, 248, 255)
    text(cv, img, key.slice(0, 18), columns[0], y, 0.35)
    text(cv, img, m.precision.toFixed(2), columns[1], y, 0.35)
    text(cv, img, m.recall.toFixed(2), columns[2], y, 0.35)
    text(cv, img, f1.toFixed(2), columns[3], y, 0.35, band)
    text(cv, img, String(m.support), columns[4], y, 0.35)
  })
  return img
}

/* ---------------------------------------------------------- decision tree */

/**
 * Draws the tree the way sklearn's `plot_tree` does: one box per node carrying
 * the split test, the impurity, the sample count and the class counts, with
 * children laid out under their parent. Leaves are spaced evenly across the
 * width and every internal node sits above the midpoint of its subtree.
 */
function drawTree(cv: any, ctx: RunContext, tree: DecisionTree, features: string[], classes: string[], criterion: string, title: string): any {
  const boxW = 132
  const boxH = 56
  const gapX = 16
  const gapY = 34
  const width = Math.max(420, tree.leaves * (boxW + gapX) + 40)
  const height = 40 + (tree.depth + 1) * (boxH + gapY)
  const img = canvas(cv, ctx, width, height)
  text(cv, img, title, 8, 20, 0.42)

  const centres = new Float64Array(tree.nodes.length)
  let nextLeaf = 0
  // Post-order: a leaf claims the next slot, an internal node centres on its children.
  const place = (index: number): number => {
    const node = tree.nodes[index]
    if (node.feature < 0) {
      centres[index] = 20 + nextLeaf * (boxW + gapX) + boxW / 2
      nextLeaf++
    } else {
      centres[index] = (place(node.left) + place(node.right)) / 2
    }
    return centres[index]
  }
  place(0)

  const draw = (index: number) => {
    const node = tree.nodes[index]
    const cx = centres[index]
    const y = 32 + node.depth * (boxH + gapY)
    const x0 = Math.round(cx - boxW / 2)
    const isLeaf = node.feature < 0
    // Leaves are tinted by their majority class, as `filled=True` does.
    const fill = isLeaf ? classColour(cv, node.prediction, Math.max(1, classes.length)) : new cv.Scalar(48, 48, 48, 255)
    const faded = new cv.Scalar(
      PLOT_BG[0] + (fill[0] - PLOT_BG[0]) * 0.55,
      PLOT_BG[1] + (fill[1] - PLOT_BG[1]) * 0.55,
      PLOT_BG[2] + (fill[2] - PLOT_BG[2]) * 0.55,
      255
    )
    cv.rectangle(img, new cv.Point(x0, y), new cv.Point(x0 + boxW, y + boxH), faded, -1)
    cv.rectangle(img, new cv.Point(x0, y), new cv.Point(x0 + boxW, y + boxH), new cv.Scalar(110, 110, 110, 255), 1)

    const head = isLeaf ? `leaf -> ${classes[node.prediction] ?? node.prediction}` : `${features[node.feature].slice(0, 9)} <= ${node.threshold.toPrecision(4)}`
    text(cv, img, head.slice(0, 22), x0 + 4, y + 14, 0.3)
    text(cv, img, `${criterion}=${node.impurity.toFixed(3)}`, x0 + 4, y + 27, 0.28)
    text(cv, img, `n=${node.samples}`, x0 + 4, y + 40, 0.28)
    text(cv, img, `[${Array.from(node.counts, (v) => Math.round(v)).join(',')}]`.slice(0, 22), x0 + 4, y + 52, 0.28)

    if (!isLeaf) {
      for (const child of [node.left, node.right]) {
        const childY = 32 + tree.nodes[child].depth * (boxH + gapY)
        cv.line(img, new cv.Point(Math.round(cx), y + boxH), new cv.Point(Math.round(centres[child]), childY),
          new cv.Scalar(120, 120, 120, 255), 1)
        draw(child)
      }
    }
  }
  draw(0)
  return img
}

const CRITERIA: Criterion[] = ['gini', 'entropy', 'entropy']

export const mlDecisionTree: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const trainDf = tableIn(inputs, 'train', 'table', 'main')
  const testDf = tableIn(inputs, 'test')
  if (!trainDf) return {}

  const target = resolveColumn(trainDf, params.target) ?? trainDf.columns[trainDf.columns.length - 1]
  const features = pickFeatures(trainDf, params.features, target)
  if (features.length === 0) return {}

  const encoded = encodeLabels([...trainDf.rows, ...(testDf?.rows ?? [])].map((r) => r[target]))
  const encoder = new Map(encoded.classes.map((c, i) => [c, i]))
  const train = prepare(trainDf, features, target, encoder)
  const test = testDf ? prepare(testDf, features, target, encoder) : { X: makeMatrix([]), y: new Int32Array(0) }
  if (train.X.n === 0) return {}

  // 'log_loss' is sklearn's alias for the entropy criterion.
  const criterionIndex = Math.round(Number(params.criterion) || 0)
  const criterion = CRITERIA[criterionIndex] ?? 'gini'
  const tree = fitTree(train.X, train.y, encoded.classes.length, [...Array(train.X.n).keys()], {
    criterion,
    maxDepth: Math.round(Number(params.max_depth) || 0),
    minSamplesSplit: Math.round(Number(params.min_samples_split) || 2),
  })

  const trainAccuracy = accuracy(treePredict(tree, train.X), train.y)
  const testAccuracy = test.X.n > 0 ? accuracy(treePredict(tree, test.X), test.y) : 0

  const [w] = previewSize(inputs.img_size, { width: 700, height: 480, ...params })
  const treePlot = drawTree(cv, ctx, tree, features, encoded.classes, criterionIndex === 0 ? 'gini' : 'entropy',
    `Decision Tree  depth=${tree.depth}  leaves=${tree.leaves}  acc=${(testAccuracy * 100).toFixed(1)}%`)

  return {
    main: treePlot,
    tree_plot: treePlot,
    importance: importancePanel(cv, ctx, features, tree.importances, w, features.length),
    accuracy: testAccuracy,
    train_acc: trainAccuracy,
    depth: tree.depth,
    importances: Array.from(tree.importances),
  }
}

/* ----------------------------------------------------------- random forest */

const MAX_FEATURE_MODES: ('sqrt' | 'log2' | null)[] = ['sqrt', 'log2', null]

export const mlRandomForest: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const trainDf = tableIn(inputs, 'train', 'table', 'main')
  const testDf = tableIn(inputs, 'test')
  if (!trainDf || !testDf) return {}

  const target = resolveColumn(trainDf, params.target) ?? trainDf.columns[trainDf.columns.length - 1]
  const features = pickFeatures(trainDf, params.features, target)
  if (features.length === 0) return {}

  const encoded = encodeLabels([...trainDf.rows, ...testDf.rows].map((r) => r[target]))
  const encoder = new Map(encoded.classes.map((c, i) => [c, i]))
  const train = prepare(trainDf, features, target, encoder)
  const test = prepare(testDf, features, target, encoder)
  if (train.X.n === 0) return {}

  const bootstrap = params.bootstrap !== false
  const modeIndex = Math.round(Number(params.max_features) || 0)
  const forest = fitForest(train.X, train.y, encoded.classes.length, {
    nEstimators: Math.round(Number(params.n_estimators) || 100),
    maxDepth: Math.round(Number(params.max_depth) || 0),
    minSamplesSplit: Math.round(Number(params.min_samples_split) || 2),
    criterion: CRITERIA[Math.round(Number(params.criterion) || 0)] ?? 'gini',
    // 'none (all)' is a legitimate null, so an out-of-range index has to be
    // caught by the bounds check — `?? 'sqrt'` would swallow the null itself.
    maxFeaturesMode: MAX_FEATURE_MODES[modeIndex] !== undefined ? MAX_FEATURE_MODES[modeIndex] : 'sqrt',
    bootstrap,
    seed: Math.round(Number(params.random_state) ?? 42),
  })

  const trainAccuracy = accuracy(forestPredict(forest, train.X), train.y)
  const testPredictions = test.X.n > 0 ? forestPredict(forest, test.X) : new Int32Array(0)
  const testAccuracy = accuracy(testPredictions, test.y)
  // OOB needs bootstrapping: without it every tree saw every row.
  const oob = params.oob && bootstrap ? oobScore(forest, train.X, train.y) : 0

  const report = test.X.n > 0 ? classificationReport(test.y, testPredictions, encoded.classes) : {}
  const [w, h] = previewSize(inputs.img_size, { width: 540, height: 420, ...params })
  const preview = confusionPanel(cv, ctx, confusionMatrix(test.y, testPredictions, encoded.classes.length),
    encoded.classes, w, h, `Random Forest  ${forest.trees.length} trees  acc=${(testAccuracy * 100).toFixed(1)}%`)

  // The optional third input classifies a whole frame, e.g. every pixel of an image.
  let predictions: DataFrame | null = null
  const predictDf = tableIn(inputs, 'predict_table')
  if (predictDf) {
    const rows: number[][] = []
    const keep: number[] = []
    predictDf.rows.forEach((record, i) => {
      const values = features.map((f) => Number(record[f]))
      if (values.every((v) => Number.isFinite(v))) { rows.push(values); keep.push(i) }
    })
    if (rows.length > 0) {
      const labels = forestPredict(forest, makeMatrix(rows))
      predictions = makeDf([...predictDf.columns, 'prediction'],
        keep.map((rowIndex, i) => ({ ...predictDf.rows[rowIndex], prediction: encoded.classes[labels[i]] })))
    }
  }

  return {
    main: preview,
    preview,
    importance: importancePanel(cv, ctx, features, forest.importances, w, Math.round(Number(params.top_n_feat) || 20)),
    accuracy: testAccuracy,
    train_acc: trainAccuracy,
    oob_score: oob,
    report: reportPanel(cv, ctx, report as Record<string, any>, Math.max(w, 420), `Random Forest (${forest.trees.length} trees)`),
    report_data: report,
    predictions,
    importances: Array.from(forest.importances),
  }
}

/* ------------------------------------------------------------- robust line */

const FIT_MODES = ['Least Squares (L2)', 'Huber', 'Theil-Sen (median)']

export const mlRobustLine: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const df = tableIn(inputs, 'table', 'data')
  const image = inputs.image ?? inputs.main
  const mode = Math.round(Number(params.mode) || 0)
  const empty = { main: null, slope: 0, intercept: 0, n_points: 0 }

  let x: number[] = []
  let y: number[] = []
  let xLabel = 'x'
  let yLabel = 'y'

  if (df) {
    const numeric = df.columns.filter((c) => isNumericColumn(df, c))
    const xCol = resolveColumn(df, params.x_col) ?? numeric[0]
    const yCol = resolveColumn(df, params.y_col) ?? numeric[1]
    if (!xCol || !yCol) return empty
    xLabel = xCol
    yLabel = yCol
    for (const record of df.rows) {
      const a = Number(record[xCol])
      const b = Number(record[yCol])
      if (Number.isFinite(a) && Number.isFinite(b)) { x.push(a); y.push(b) }
    }
  } else if (image && typeof (image as any).cols === 'number') {
    // Chart-image mode: dark marker blobs become points, with y measured from
    // the bottom so the fit reads like a normal plot rather than upside down.
    const source = image as any
    const gray = new cv.Mat()
    if (source.channels() > 1) cv.cvtColor(source, gray, cv.COLOR_BGR2GRAY)
    else source.copyTo(gray)
    const binary = new cv.Mat()
    cv.threshold(gray, binary, Math.round(Number(params.marker_thresh) || 128) - 1, 255, cv.THRESH_BINARY_INV)
    const labels = new cv.Mat()
    const stats = new cv.Mat()
    const centroids = new cv.Mat()
    const count = cv.connectedComponentsWithStats(binary, labels, stats, centroids, 8, cv.CV_32S)
    const minArea = Math.round(Number(params.min_blob_px) || 4)
    for (let i = 1; i < count; i++) {
      if (stats.intAt(i, 4) < minArea) continue
      x.push(centroids.doubleAt(i, 0))
      y.push(source.rows - centroids.doubleAt(i, 1))
    }
    gray.delete(); binary.delete(); labels.delete(); stats.delete(); centroids.delete()
    xLabel = 'x (px)'
    yLabel = 'y (px)'
    if (x.length < 2) return empty
  } else {
    return empty
  }

  if (x.length < 2) return empty

  const fit = mode === 1
    ? lineFitHuber(x, y, Number(params.huber_delta) || 1.35)
    : mode === 2
      ? lineFitTheilSen(x, y)
      : lineFitL2(x, y)

  const [w, h] = previewSize(inputs.img_size, { width: 600, height: 400, ...params })
  const img = canvas(cv, ctx, w, h)
  const xMin = Math.min(...x)
  const xMax = Math.max(...x)
  const lineLow = fit.slope * xMin + fit.intercept
  const lineHigh = fit.slope * xMax + fit.intercept
  const yMin = Math.min(...y, lineLow, lineHigh)
  const yMax = Math.max(...y, lineLow, lineHigh)
  const pad = (xMax - xMin) * 0.04 + 1e-9
  const yPad = (yMax - yMin) * 0.06 + 1e-9
  const axes = drawAxes(cv, img, xMin - pad, xMax + pad, yMin - yPad, yMax + yPad, true,
    `${FIT_MODES[mode]}: y=${fit.slope.toFixed(3)}x+${fit.intercept.toFixed(2)}`)

  const dot = new cv.Scalar(248, 189, 56, 255)   // BGR of #38bdf8
  for (let i = 0; i < x.length; i++) {
    const [px, py] = project(axes, x[i], y[i])
    cv.circle(img, new cv.Point(px, py), 3, dot, -1)
  }
  const [ax, ay] = project(axes, xMin, lineLow)
  const [bx, by] = project(axes, xMax, lineHigh)
  cv.line(img, new cv.Point(ax, ay), new cv.Point(bx, by), new cv.Scalar(94, 197, 34, 255), 2)
  text(cv, img, xLabel.slice(0, 14), axes.left + axes.width - 70, h - 22, 0.32)
  text(cv, img, yLabel.slice(0, 14), 4, axes.top - 2, 0.32)

  return {
    main: img,
    // The desktop rounds both to five decimals before publishing them.
    slope: Math.round(fit.slope * 1e5) / 1e5,
    intercept: Math.round(fit.intercept * 1e5) / 1e5,
    n_points: x.length,
  }
}

/* ------------------------------------------------------ parameter optimizer */

interface BestParamsState {
  history: Map<number, Record<string, number>>
  counter: number
  lastReset: number
}

/**
 * Tracks a metric per training step and reports the step with the best weighted
 * score. Weights come from per-metric `weight_<name>` params, which the desktop
 * UI creates as metrics arrive; a metric with no weight contributes nothing.
 */
export const mlBestParams: NodeImpl = (inputs, params, ctx) => {
  const key = `${ctx.nodeId}:best_params`
  let state = ctx.state.get(key) as BestParamsState | undefined
  if (!state) {
    state = { history: new Map(), counter: 0, lastReset: 0 }
    ctx.state.set(key, state)
  }

  // Rising edge on the reset trigger, the convention every one-shot action uses.
  const reset = Number(params.reset) || 0
  if (reset === 1 && state.lastReset !== 1) {
    state.history.clear()
    state.counter = 0
    state.lastReset = reset
    return { best_step: 0, best_values: {}, counter: 0, current_values: {}, best_step_values: {} }
  }
  state.lastReset = reset

  let portLabels: Record<string, string> = {}
  try {
    const parsed = JSON.parse(String(params.port_labels ?? '{}'))
    if (parsed && typeof parsed === 'object') portLabels = parsed as Record<string, string>
  } catch {
    // A malformed mapping just means the port ids are used as metric names.
  }

  const current: Record<string, number | string> = {}
  for (const [port, value] of Object.entries(inputs)) {
    if (port === 'counter' || value === null || value === undefined) continue
    let label = portLabels[port]
    if (!label) {
      for (const [id, mapped] of Object.entries(portLabels)) {
        if (id.endsWith(port) || port.endsWith(id)) { label = mapped; break }
      }
    }
    const name = label || port.split('__').pop() || port

    if (typeof value === 'object' && !Array.isArray(value)) {
      // A dict input contributes each of its own numeric entries as a metric.
      current[name] = '__dict__'
      for (const [sub, subValue] of Object.entries(value as Record<string, unknown>)) {
        const numeric = Number(subValue)
        if (Number.isFinite(numeric)) current[String(sub)] = numeric
      }
    } else {
      const numeric = Number(value)
      if (Number.isFinite(numeric)) current[name] = numeric
    }
  }

  const weights: Record<string, number> = {}
  for (const name of Object.keys(current)) {
    const active = params[`active_${name}`]
    const enabled = active === undefined ? true : typeof active === 'string' ? active.toLowerCase() === 'true' : Boolean(active)
    if (enabled) weights[name] = Number(params[`weight_${name}`]) || 0
  }

  if (inputs.counter !== null && inputs.counter !== undefined) {
    const counter = Number(inputs.counter)
    const step = Number.isFinite(counter) ? Math.trunc(counter) : 0
    if (Number.isFinite(counter)) state.counter = counter
    if (Object.keys(current).length > 0) {
      const entry = state.history.get(step) ?? {}
      Object.assign(entry, current)
      state.history.set(step, entry as Record<string, number>)
    }
  }

  let bestStep = 0
  let bestScore = -Infinity
  for (const [step, metrics] of state.history) {
    let score = 0
    for (const [name, value] of Object.entries(metrics)) {
      if (value === ('__dict__' as unknown)) continue
      score += (weights[name] ?? 0) * Number(value)
    }
    if (score > bestScore) { bestScore = score; bestStep = step }
  }
  const bestValues = state.history.get(bestStep) ?? {}

  return {
    best_step: bestStep,
    best_values: bestValues,
    counter: state.counter,
    current_values: current,
    best_step_values: bestValues,
  }
}
