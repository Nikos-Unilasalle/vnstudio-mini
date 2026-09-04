/**
 * Machine Learning nodes: K-Means, PCA, train/test split, KNN, linear
 * regression and the two scoring panels.
 *
 * The desktop versions call scikit-learn and draw with matplotlib. The
 * algorithms live in `../ml`, and the figures are drawn with OpenCV here, the
 * same substitution the rest of the port makes.
 */
import type { NodeImpl, RunContext } from '../types'
import {
  DataFrame,
  dfMeta,
  isDf,
  isNa,
  isNumericColumn,
  makeDf,
  previewSize,
  resolveColumn,
  splitList,
} from '../dataframe'
import {
  assignClusters,
  calinskiHarabasz,
  classificationReport,
  ClassMetrics,
  confusionMatrix,
  daviesBouldin,
  dunnIndex,
  knnFit,
  knnPredict,
  knnPredictOne,
  kmeans,
  leastSquares,
  makeMatrix,
  Matrix,
  Metric,
  pca,
  pcaInverse,
  r2Score,
  shuffledIndices,
  silhouetteScore,
  standardize,
} from '../ml'
import { Axes, classColour, drawAxes, PLOT_BG, PLOT_INK, project } from './mlData'

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

function ink(cv: any): any {
  return new cv.Scalar(PLOT_INK[0], PLOT_INK[1], PLOT_INK[2], 255)
}

function text(cv: any, img: any, s: string, x: number, y: number, scale = 0.4, colour?: any): void {
  cv.putText(img, s, new cv.Point(x, y), cv.FONT_HERSHEY_SIMPLEX, scale, colour ?? ink(cv), 1, cv.LINE_AA)
}

/** Numeric feature columns, from a comma list or every numeric column. */
function pickFeatures(df: DataFrame, raw: unknown, exclude?: string): string[] {
  const numeric = df.columns.filter((c) => isNumericColumn(df, c) && c !== exclude)
  const requested = splitList(raw)
    .map((name) => resolveColumn(df, name))
    .filter((name): name is string => name !== null && numeric.includes(name))
  return requested.length > 0 ? requested : numeric
}

/** Rows where every feature is a finite number, as a matrix plus the row indices kept. */
function featureMatrix(df: DataFrame, features: string[]): { X: Matrix; keep: number[] } {
  const rows: number[][] = []
  const keep: number[] = []
  df.rows.forEach((r, i) => {
    const values = features.map((f) => Number(r[f]))
    if (values.every((v) => Number.isFinite(v))) {
      rows.push(values)
      keep.push(i)
    }
  })
  return { X: makeMatrix(rows), keep }
}

/** sklearn's LabelEncoder: classes sorted by their string form, mapped to 0..n-1. */
function encodeLabels(values: unknown[]): { codes: Int32Array; classes: string[] } {
  const classes = [...new Set(values.map((v) => String(v)))].sort()
  const index = new Map(classes.map((c, i) => [c, i]))
  const codes = new Int32Array(values.length)
  values.forEach((v, i) => { codes[i] = index.get(String(v)) ?? 0 })
  return { codes, classes }
}

function extent(X: Matrix, axis: number): [number, number] {
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < X.n; i++) {
    const v = X.data[i * X.d + axis]
    min = Math.min(min, v)
    max = Math.max(max, v)
  }
  if (!Number.isFinite(min)) return [0, 1]
  const margin = (max - min) * 0.12 + 1e-6
  return [min - margin, max + margin]
}

/**
 * Fills the plot area with the decision region each pixel block falls in, the
 * job matplotlib's `contourf` does on the desktop. `classify` is called on a
 * coarse grid and the result is drawn as blocks, since calling it per pixel
 * would make a KNN boundary cost a million distance sweeps.
 */
function drawRegions(cv: any, img: any, axes: Axes, resolution: number, total: number, classify: (x: number, y: number) => number): void {
  const steps = Math.max(8, Math.min(resolution, 160))
  const cellW = axes.width / steps
  const cellH = axes.height / steps
  for (let gy = 0; gy < steps; gy++) {
    for (let gx = 0; gx < steps; gx++) {
      const x = axes.xMin + ((gx + 0.5) / steps) * (axes.xMax - axes.xMin)
      const y = axes.yMax - ((gy + 0.5) / steps) * (axes.yMax - axes.yMin)
      const label = classify(x, y)
      const colour = classColour(cv, label, total)
      // 22% opacity, matching the desktop's contourf alpha, mixed against the
      // plot background so the scatter on top stays legible.
      const faded = new cv.Scalar(
        PLOT_BG[0] + (colour[0] - PLOT_BG[0]) * 0.22,
        PLOT_BG[1] + (colour[1] - PLOT_BG[1]) * 0.22,
        PLOT_BG[2] + (colour[2] - PLOT_BG[2]) * 0.22,
        255
      )
      const x0 = Math.round(axes.left + gx * cellW)
      const y0 = Math.round(axes.top + gy * cellH)
      cv.rectangle(
        img,
        new cv.Point(x0, y0),
        new cv.Point(Math.round(axes.left + (gx + 1) * cellW), Math.round(axes.top + (gy + 1) * cellH)),
        faded,
        -1
      )
    }
  }
}

/** A small legend in the top-right of the plot area. */
function legend(cv: any, img: any, axes: Axes, entries: string[], total: number): void {
  const x = axes.left + axes.width - 96
  entries.slice(0, 10).forEach((label, i) => {
    const y = axes.top + 12 + i * 13
    cv.circle(img, new cv.Point(x, y - 3), 3, classColour(cv, i, total), -1)
    text(cv, img, label.slice(0, 12), x + 8, y, 0.32)
  })
}

/* ------------------------------------------------------------------ K-Means */

export const mlKmeans: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const df = tableIn(inputs, 'table', 'data', 'main')
  if (!df) return {}
  const meta = dfMeta(df, 0)

  const features = pickFeatures(df, params.features)
  if (features.length === 0) return { df_meta: meta }

  const k = Math.max(2, Math.round(Number(params.k) || 3))
  const { X: raw, keep } = featureMatrix(df, features)
  if (raw.n < k) return { df_meta: meta }

  const wantsStandardize = params.standardize !== false
  const scaler = wantsStandardize ? standardize(raw) : null
  const X = scaler ? scaler.scaled : raw

  const seed = Math.round(Number(params.random_state) ?? 42)
  const options = {
    init: (Math.round(Number(params.init) || 0) === 1 ? 'random' : 'k-means++') as 'random' | 'k-means++',
    maxIter: Math.round(Number(params.max_iter) || 300),
    nInit: Math.round(Number(params.n_init) || 10),
    seed,
  }
  const model = kmeans(X, k, options)
  const silhouette = k > 1 && X.n > k ? silhouetteScore(X, model.labels) : 0

  const outRows = keep.map((rowIndex, i) => ({ ...df.rows[rowIndex], cluster: model.labels[i] }))
  const table = makeDf([...df.columns, 'cluster'], outRows)

  // Two features plot directly; anything wider is shown through a PCA projection,
  // as the desktop node does.
  const usePca = features.length !== 2
  const projection = usePca ? pca(X, 2) : null
  const view: Matrix = projection
    ? { data: projection.scores, n: X.n, d: 2 }
    : X
  const centresView: number[][] = []
  for (let c = 0; c < k; c++) {
    const centre = Array.from(model.centers.subarray(c * X.d, c * X.d + X.d))
    if (projection) {
      const coords: number[] = []
      for (let comp = 0; comp < 2; comp++) {
        let sum = 0
        for (let j = 0; j < X.d; j++) sum += (centre[j] - projection.mean[j]) * projection.components[comp * X.d + j]
        coords.push(sum)
      }
      centresView.push(coords)
    } else {
      centresView.push(centre)
    }
  }

  const [w, h] = previewSize(inputs.img_size, { width: 540, height: 420, ...params })
  const img = canvas(cv, ctx, w, h)
  const [xMin, xMax] = extent(view, 0)
  const [yMin, yMax] = extent(view, 1)
  const title = `K-Means k=${k}  inertia=${model.inertia.toFixed(1)}  sil=${silhouette.toFixed(3)}${usePca ? '  (PCA)' : ''}`
  const axes = drawAxes(cv, img, xMin, xMax, yMin, yMax, true, title)

  if (params.show_regions !== false) {
    const resolution = Math.round(Number(params.boundary_res) || 120)
    drawRegions(cv, img, axes, resolution, k, (x, y) => {
      const point = projection ? Array.from(pcaInverse(projection, [x, y])) : [x, y]
      return assignClusters(makeMatrix([point]), model.centers, k)[0]
    })
    // Redraw the frame, which the region fill has just painted over.
    cv.rectangle(img, new cv.Point(axes.left, axes.top), new cv.Point(axes.left + axes.width, axes.top + axes.height), ink(cv), 1)
  }

  for (let i = 0; i < view.n; i++) {
    const [px, py] = project(axes, view.data[i * view.d], view.data[i * view.d + 1])
    cv.circle(img, new cv.Point(px, py), 3, classColour(cv, model.labels[i], k), -1)
  }
  // Centroids as white crosses, the desktop's 'X' marker.
  const white = new cv.Scalar(255, 255, 255, 255)
  for (const centre of centresView) {
    const [px, py] = project(axes, centre[0], centre[1])
    cv.line(img, new cv.Point(px - 5, py - 5), new cv.Point(px + 5, py + 5), white, 2)
    cv.line(img, new cv.Point(px - 5, py + 5), new cv.Point(px + 5, py - 5), white, 2)
  }
  legend(cv, img, axes, Array.from({ length: k }, (_, i) => `Cluster ${i}`), k)

  let elbow: any = null
  if (params.show_elbow) {
    const kMax = Math.min(Math.round(Number(params.k_max) || 10), X.n)
    const inertias: number[] = []
    for (let ki = 1; ki <= kMax; ki++) inertias.push(kmeans(X, ki, options).inertia)
    elbow = canvas(cv, ctx, w, Math.round(h * 0.7))
    const elbowAxes = drawAxes(cv, elbow, 1, kMax, 0, Math.max(...inertias, 1), true, 'Elbow curve')
    const line = new cv.Scalar(246, 156, 91, 255)   // BGR of the desktop's #5b9cf6
    for (let i = 0; i < inertias.length; i++) {
      const [px, py] = project(elbowAxes, i + 1, inertias[i])
      cv.circle(elbow, new cv.Point(px, py), 3, line, -1)
      if (i > 0) {
        const [qx, qy] = project(elbowAxes, i, inertias[i - 1])
        cv.line(elbow, new cv.Point(qx, qy), new cv.Point(px, py), line, 1)
      }
    }
    const [kx] = project(elbowAxes, k, 0)
    cv.line(elbow, new cv.Point(kx, elbowAxes.top), new cv.Point(kx, elbowAxes.top + elbowAxes.height), new cv.Scalar(11, 158, 245, 255), 1)
  }

  return {
    table,
    main: img,
    preview: img,
    elbow_plot: elbow,
    inertia: model.inertia,
    silhouette,
    df_meta: meta,
  }
}

/* ---------------------------------------------------------------------- PCA */

export const mlPca: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const df = tableIn(inputs, 'table', 'data', 'main')
  if (!df) return {}

  const features = pickFeatures(df, params.features)
  if (features.length < 2) return {}

  const { X: raw, keep } = featureMatrix(df, features)
  if (raw.n < 2) return {}
  const X = params.standardize !== false ? standardize(raw).scaled : raw

  const nComp = Math.max(2, Math.min(Math.round(Number(params.n_components) || 5), features.length, X.n))
  const model = pca(X, nComp)

  const columns = [...features, ...Array.from({ length: model.nComp }, (_, i) => `PC${i + 1}`)]
  const rows = keep.map((rowIndex, i) => {
    const out: Record<string, unknown> = {}
    for (const f of features) out[f] = df.rows[rowIndex][f]
    for (let c = 0; c < model.nComp; c++) out[`PC${c + 1}`] = model.scores[i * model.nComp + c]
    return out
  })
  const transformed = makeDf(columns, rows)

  const pc1 = model.explainedVarianceRatio[0] * 100
  const pc2 = model.nComp >= 2 ? model.explainedVarianceRatio[1] * 100 : 0

  const [w, h] = previewSize(inputs.img_size, { width: 540, height: 420, ...params })
  const scatter = canvas(cv, ctx, w, h)
  const view: Matrix = { data: model.scores, n: X.n, d: model.nComp }
  const [xMin, xMax] = extent(view, 0)
  const [yMin, yMax] = extent(view, 1)
  const axes = drawAxes(cv, scatter, xMin, xMax, yMin, yMax, true,
    `PCA ${model.nComp} comps  PC1 ${pc1.toFixed(1)}%  PC2 ${pc2.toFixed(1)}%`)

  const hueName = resolveColumn(df, params.hue_col)
  let hueCodes: Int32Array | null = null
  let hueClasses: string[] = []
  if (hueName) {
    const encoded = encodeLabels(keep.map((rowIndex) => df.rows[rowIndex][hueName]))
    hueCodes = encoded.codes
    hueClasses = encoded.classes.slice(0, 20)
  }
  const dotSize = Math.max(1, Math.round(Math.sqrt(Number(params.dot_size) || 30) / 1.6))
  for (let i = 0; i < view.n; i++) {
    const [px, py] = project(axes, model.scores[i * model.nComp], model.scores[i * model.nComp + 1])
    const colour = hueCodes ? classColour(cv, hueCodes[i], Math.max(1, hueClasses.length)) : classColour(cv, i, view.n)
    cv.circle(scatter, new cv.Point(px, py), dotSize, colour, -1)
  }

  if (params.show_loadings) {
    // Biplot arrows: each feature's weight on PC1/PC2, scaled to the cloud.
    let reach = 0
    for (let i = 0; i < view.n * model.nComp; i++) reach = Math.max(reach, Math.abs(model.scores[i]))
    const scale = reach * 0.8
    const arrow = new cv.Scalar(22, 115, 249, 255)   // BGR of #f97316
    const [ox, oy] = project(axes, 0, 0)
    features.slice(0, 8).forEach((name, j) => {
      // components is nComp x d row-major: PC1's weight for feature j, then PC2's.
      const [ax, ay] = project(axes, model.components[j] * scale, model.components[X.d + j] * scale)
      cv.line(scatter, new cv.Point(ox, oy), new cv.Point(ax, ay), arrow, 1)
      text(cv, scatter, name.slice(0, 10), ax + 3, ay, 0.32, arrow)
    })
  }
  if (hueClasses.length > 0) legend(cv, scatter, axes, hueClasses, hueClasses.length)

  const variance = canvas(cv, ctx, Math.max(360, model.nComp * 70), 320)
  const varianceAxes = drawAxes(cv, variance, 0.5, model.nComp + 0.5, 0, 105, true, 'Explained variance (%)')
  let cumulative = 0
  const cumulativePoints: [number, number][] = []
  for (let c = 0; c < model.nComp; c++) {
    const percent = model.explainedVarianceRatio[c] * 100
    const [cx, cy] = project(varianceAxes, c + 1, percent)
    const [, baseY] = project(varianceAxes, c + 1, 0)
    const half = Math.max(4, Math.round(varianceAxes.width / (model.nComp * 3)))
    cv.rectangle(variance, new cv.Point(cx - half, cy), new cv.Point(cx + half, baseY), classColour(cv, c, model.nComp), -1)
    cumulative += percent
    cumulativePoints.push(project(varianceAxes, c + 1, cumulative))
  }
  const cumulativeColour = new cv.Scalar(22, 115, 249, 255)
  for (let c = 0; c < cumulativePoints.length; c++) {
    const [px, py] = cumulativePoints[c]
    cv.circle(variance, new cv.Point(px, py), 3, cumulativeColour, -1)
    if (c > 0) cv.line(variance, new cv.Point(cumulativePoints[c - 1][0], cumulativePoints[c - 1][1]), new cv.Point(px, py), cumulativeColour, 1)
  }

  return {
    transformed,
    main: scatter,
    scatter,
    variance_plot: variance,
    pc1_variance: pc1,
    pc2_variance: pc2,
    explained_variance: Array.from(model.explainedVarianceRatio, (v) => v * 100),
    df_meta: dfMeta(transformed, 0),
  }
}

/* ------------------------------------------------------- train / test split */

export const mlTrainTestSplit: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  let df = tableIn(inputs, 'table', 'data', 'main')
  if (!df) return {}

  // Unlabelled rows (label == -1 by default) are dropped before splitting, so
  // an unlabelled sample never lands in the test set.
  const filterCol = resolveColumn(df, params.filter_col)
  if (filterCol) {
    const noData = Number(params.filter_nodata ?? -1)
    df = makeDf(df.columns, df.rows.filter((r) => Number(r[filterCol]) !== noData))
    if (df.rows.length === 0) return {}
  }

  const testFraction = Math.min(0.95, Math.max(0.01, (Number(params.test_size) || 20) / 100))
  const seed = Math.round(Number(params.random_state) ?? 42)
  const shuffle = params.shuffle !== false
  const stratifyCol = resolveColumn(df, params.stratify_col)

  const trainRows: Record<string, unknown>[] = []
  const testRows: Record<string, unknown>[] = []

  /** Splits one pool of row indices, taking the test share off the front. */
  const splitPool = (pool: number[], salt: number) => {
    const order = shuffle ? shuffledIndices(pool.length, seed + salt).map((i) => pool[i]) : [...pool]
    const testCount = Math.ceil(pool.length * testFraction)
    order.forEach((rowIndex, position) => {
      (position < testCount ? testRows : trainRows).push(df!.rows[rowIndex])
    })
  }

  if (stratifyCol) {
    // Stratified: each class is split at the same ratio, so the class balance
    // of the test set matches the whole frame.
    const pools = new Map<string, number[]>()
    df.rows.forEach((r, i) => {
      const key = String(r[stratifyCol])
      const pool = pools.get(key)
      if (pool) pool.push(i)
      else pools.set(key, [i])
    })
    ;[...pools.keys()].sort().forEach((key, salt) => splitPool(pools.get(key)!, salt))
  } else {
    splitPool([...df.rows.keys()], 0)
  }

  const train = makeDf(df.columns, trainRows)
  const test = makeDf(df.columns, testRows)

  const [w, h] = previewSize(inputs.img_size, { width: 300, height: 160, ...params })
  const img = canvas(cv, ctx, w, h)
  cv.rectangle(img, new cv.Point(0, 0), new cv.Point(w, 26), new cv.Scalar(45, 45, 45, 255), -1)
  text(cv, img, 'Train / Test Split', 8, 17, 0.42)
  cv.line(img, new cv.Point(0, 26), new cv.Point(w, 26), new cv.Scalar(80, 80, 80, 255), 1)
  const total = trainRows.length + testRows.length
  const trainShare = total > 0 ? trainRows.length / total : 0
  const barWidth = w - 16
  cv.rectangle(img, new cv.Point(8, 38), new cv.Point(8 + barWidth, 54), new cv.Scalar(60, 60, 60, 255), -1)
  cv.rectangle(img, new cv.Point(8, 38), new cv.Point(8 + Math.round(barWidth * trainShare), 54), new cv.Scalar(246, 130, 59, 255), -1)
  text(cv, img, `Train : ${trainRows.length} rows  (${Math.round(trainShare * 100)}%)`, 8, 72, 0.4, new cv.Scalar(255, 200, 140, 255))
  text(cv, img, `Test  : ${testRows.length} rows  (${Math.round((1 - trainShare) * 100)}%)`, 8, 90, 0.4, new cv.Scalar(80, 165, 255, 255))
  if (stratifyCol) text(cv, img, `Stratified by: ${stratifyCol}`, 8, 112, 0.38, new cv.Scalar(160, 160, 160, 255))

  return {
    train,
    test,
    train_count: trainRows.length,
    test_count: testRows.length,
    main: img,
    preview: img,
  }
}

/* --------------------------------------------------------- report rendering */

/** The precision/recall/F1 table both classifier nodes show. */
function reportPanel(cv: any, ctx: RunContext, report: Record<string, unknown>, width: number, title: string): any {
  const classes = Object.keys(report).filter(
    (k) => k !== 'accuracy' && k !== 'macro avg' && k !== 'weighted avg' && typeof report[k] === 'object'
  )
  const lines = [...classes, 'macro avg', 'weighted avg'].filter((k) => report[k])
  const height = 46 + lines.length * 20
  const img = canvas(cv, ctx, width, height)

  const accuracy = Number(report.accuracy ?? 0)
  cv.rectangle(img, new cv.Point(0, 0), new cv.Point(width, 24), new cv.Scalar(58, 42, 42, 255), -1)
  text(cv, img, `${title}  ·  Accuracy ${(accuracy * 100).toFixed(1)}%`, 8, 16, 0.4, new cv.Scalar(252, 180, 165, 255))

  const columns = [8, Math.round(width * 0.42), Math.round(width * 0.58), Math.round(width * 0.74), Math.round(width * 0.88)]
  const header = ['Class', 'Prec', 'Recall', 'F1', 'Supp']
  header.forEach((label, i) => text(cv, img, label, columns[i], 40, 0.35, new cv.Scalar(252, 180, 165, 255)))

  lines.forEach((key, i) => {
    const metrics = report[key] as ClassMetrics
    const y = 60 + i * 20
    const f1 = metrics['f1-score']
    // F1 colour bands are the desktop's: green ≥ 0.9, amber ≥ 0.7, else red.
    const f1Colour = f1 >= 0.9
      ? new cv.Scalar(183, 231, 110, 255)
      : f1 >= 0.7
        ? new cv.Scalar(61, 211, 252, 255)
        : new cv.Scalar(113, 113, 248, 255)
    text(cv, img, key.slice(0, 18), columns[0], y, 0.35)
    text(cv, img, metrics.precision.toFixed(2), columns[1], y, 0.35)
    text(cv, img, metrics.recall.toFixed(2), columns[2], y, 0.35)
    text(cv, img, f1.toFixed(2), columns[3], y, 0.35, f1Colour)
    text(cv, img, String(metrics.support), columns[4], y, 0.35)
  })
  return img
}

/** Row-normalised confusion matrix as a blue heatmap with the values written in. */
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
      // matplotlib's 'Blues': white at 0, deep blue at 1.
      const shade = new cv.Scalar(255 - value * 100, 255 - value * 180, 255 - value * 235, 255)
      const x0 = left + j * cell
      const y0 = top + i * cell
      cv.rectangle(img, new cv.Point(x0, y0), new cv.Point(x0 + cell, y0 + cell), shade, -1)
      cv.rectangle(img, new cv.Point(x0, y0), new cv.Point(x0 + cell, y0 + cell), new cv.Scalar(70, 70, 70, 255), 1)
      const label = value.toFixed(2)
      text(cv, img, label, x0 + Math.max(2, (cell - label.length * 7) / 2), y0 + cell / 2 + 4, 0.32,
        value > 0.5 ? new cv.Scalar(255, 255, 255, 255) : new cv.Scalar(51, 51, 51, 255))
    }
    text(cv, img, classes[i].slice(0, 9), 4, top + i * cell + cell / 2 + 4, 0.32)
    text(cv, img, classes[i].slice(0, 5), left + i * cell + 2, top + n * cell + 14, 0.32)
  }
  text(cv, img, 'True \\ Predicted', 4, height - 6, 0.32)
  return img
}

/* ------------------------------------------------------------ KNN classifier */

const METRICS: Metric[] = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

export const mlKnnClassifier: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const trainDf = tableIn(inputs, 'train', 'table', 'main')
  const testDf = tableIn(inputs, 'test')
  if (!trainDf || !testDf) return {}

  const target = resolveColumn(trainDf, params.target) ?? trainDf.columns[trainDf.columns.length - 1]
  const features = pickFeatures(trainDf, params.features, target)
  if (features.length === 0) return {}

  // Both frames share one encoder so class 2 means the same thing in each.
  const encoded = encodeLabels([...trainDf.rows, ...testDf.rows].map((r) => r[target]))
  const classes = encoded.classes
  const trainSet = featureMatrix(trainDf, features)
  const testSet = featureMatrix(testDf, features)
  if (trainSet.X.n === 0) return {}

  const yTrain = Int32Array.from(trainSet.keep, (i) => encoded.codes[i])
  const yTest = Int32Array.from(testSet.keep, (i) => encoded.codes[trainDf.rows.length + i])

  const k = Math.round(Number(params.k) || 5)
  const metric = METRICS[Math.round(Number(params.metric) || 0)] ?? 'euclidean'
  const weighted = Math.round(Number(params.weights) || 0) === 1
  const model = knnFit(trainSet.X, yTrain, k, metric, weighted, classes.length)

  const trainPred = knnPredict(model, trainSet.X)
  let trainCorrect = 0
  for (let i = 0; i < yTrain.length; i++) if (trainPred[i] === yTrain[i]) trainCorrect++
  const trainAccuracy = yTrain.length > 0 ? trainCorrect / yTrain.length : 0

  const testPred = testSet.X.n > 0 ? knnPredict(model, testSet.X) : new Int32Array(0)
  let testCorrect = 0
  for (let i = 0; i < yTest.length; i++) if (testPred[i] === yTest[i]) testCorrect++
  const testAccuracy = yTest.length > 0 ? testCorrect / yTest.length : 0

  const report = testSet.X.n > 0 ? classificationReport(yTest, testPred, classes) : {}

  const [w, h] = previewSize(inputs.img_size, { width: 540, height: 420, ...params })
  let preview: any
  if (features.length === 2) {
    preview = canvas(cv, ctx, w, h)
    const all = makeMatrix([
      ...Array.from({ length: trainSet.X.n }, (_, i) => Array.from(trainSet.X.data.subarray(i * 2, i * 2 + 2))),
      ...Array.from({ length: testSet.X.n }, (_, i) => Array.from(testSet.X.data.subarray(i * 2, i * 2 + 2))),
    ])
    const [xMin, xMax] = extent(all, 0)
    const [yMin, yMax] = extent(all, 1)
    const axes = drawAxes(cv, preview, xMin, xMax, yMin, yMax, true,
      `KNN k=${k}  test acc = ${(testAccuracy * 100).toFixed(1)}%`)
    drawRegions(cv, preview, axes, Math.round(Number(params.boundary_res) || 150), classes.length,
      (x, y) => knnPredictOne(model, [x, y]))
    cv.rectangle(preview, new cv.Point(axes.left, axes.top), new cv.Point(axes.left + axes.width, axes.top + axes.height), ink(cv), 1)

    for (let i = 0; i < trainSet.X.n; i++) {
      const [px, py] = project(axes, trainSet.X.data[i * 2], trainSet.X.data[i * 2 + 1])
      cv.circle(preview, new cv.Point(px, py), 3, classColour(cv, yTrain[i], classes.length), -1)
    }
    // Test points are drawn larger with a white ring, as the desktop's stars are.
    for (let i = 0; i < testSet.X.n; i++) {
      const [px, py] = project(axes, testSet.X.data[i * 2], testSet.X.data[i * 2 + 1])
      cv.circle(preview, new cv.Point(px, py), 5, classColour(cv, yTest[i], classes.length), -1)
      cv.circle(preview, new cv.Point(px, py), 5, new cv.Scalar(255, 255, 255, 255), 1)
    }
    legend(cv, preview, axes, classes, classes.length)
  } else {
    preview = confusionPanel(cv, ctx, confusionMatrix(yTest, testPred, classes.length), classes, w, h,
      `KNN k=${k}  acc = ${(testAccuracy * 100).toFixed(1)}%`)
  }

  return {
    main: preview,
    preview,
    accuracy: testAccuracy,
    train_acc: trainAccuracy,
    report: reportPanel(cv, ctx, report as Record<string, unknown>, Math.max(w, 420), `KNN k=${k}`),
    report_data: report,
  }
}

/* -------------------------------------------------------- linear regression */

export const mlLinearRegression: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const trainDf = tableIn(inputs, 'train', 'table', 'main')
  const testDf = tableIn(inputs, 'test')
  if (!trainDf) return {}

  const target = resolveColumn(trainDf, params.target) ?? trainDf.columns[trainDf.columns.length - 1]
  const features = pickFeatures(trainDf, params.features, target)
  if (features.length === 0) return {}

  const fitIntercept = params.fit_intercept !== false

  /** Feature rows plus targets, keeping only rows where both sides are finite. */
  const extract = (df: DataFrame | null): { rows: number[][]; y: number[] } => {
    const rows: number[][] = []
    const y: number[] = []
    if (!df) return { rows, y }
    for (const record of df.rows) {
      const values = features.map((f) => Number(record[f]))
      const label = Number(record[target])
      if (values.every((v) => Number.isFinite(v)) && Number.isFinite(label)) {
        rows.push(values)
        y.push(label)
      }
    }
    return { rows, y }
  }

  const train = extract(trainDf)
  const test = extract(testDf)
  if (train.rows.length === 0) return {}

  let mean: Float64Array | null = null
  let scale: Float64Array | null = null
  if (params.standardize) {
    const scaler = standardize(makeMatrix(train.rows))
    mean = scaler.mean
    scale = scaler.scale
    const apply = (rows: number[][]) => rows.map((r) => r.map((v, j) => (v - mean![j]) / scale![j]))
    train.rows = apply(train.rows)
    test.rows = apply(test.rows)
  }

  const design = (rows: number[][]) => (fitIntercept ? rows.map((r) => [...r, 1]) : rows)
  const coefficients = leastSquares(design(train.rows), train.y)
  const intercept = fitIntercept ? coefficients[coefficients.length - 1] : 0
  const weights = fitIntercept ? Array.from(coefficients.subarray(0, features.length)) : Array.from(coefficients)

  const predict = (rows: number[][]) =>
    rows.map((r) => r.reduce((sum, v, j) => sum + v * weights[j], intercept))

  const trainPred = predict(train.rows)
  const testPred = predict(test.rows)
  const r2Train = r2Score(train.y, trainPred)
  const r2Test = test.y.length > 0 ? r2Score(test.y, testPred) : NaN
  const residuals = test.y.map((v, i) => v - testPred[i])
  const rmse = residuals.length > 0 ? Math.sqrt(residuals.reduce((a, b) => a + b * b, 0) / residuals.length) : NaN

  const [w, h] = previewSize(inputs.img_size, { width: 600, height: 420, ...params })
  const preview = canvas(cv, ctx, w, h)
  const half = Math.floor(w / 2)

  if (test.y.length > 0) {
    // Left: predicted against actual, with the perfect-fit diagonal.
    const lo = Math.min(...test.y, ...testPred)
    const hi = Math.max(...test.y, ...testPred)
    const leftPanel = preview.roi(new cv.Rect(0, 0, half, h))
    const leftAxes = drawAxes(cv, leftPanel, lo, hi, lo, hi, true, `Predicted vs Actual  R2=${r2Test.toFixed(3)}`)
    const blue = new cv.Scalar(246, 130, 59, 255)
    const orange = new cv.Scalar(22, 115, 249, 255)
    const [dx0, dy0] = project(leftAxes, lo, lo)
    const [dx1, dy1] = project(leftAxes, hi, hi)
    cv.line(leftPanel, new cv.Point(dx0, dy0), new cv.Point(dx1, dy1), orange, 1)
    test.y.forEach((actual, i) => {
      const [px, py] = project(leftAxes, actual, testPred[i])
      cv.circle(leftPanel, new cv.Point(px, py), 3, blue, -1)
    })
    leftPanel.delete()

    // Right: residuals against the prediction.
    const rightPanel = preview.roi(new cv.Rect(half, 0, w - half, h))
    const spread = Math.max(...residuals.map(Math.abs), 1e-6)
    const rightAxes = drawAxes(cv, rightPanel, Math.min(...testPred), Math.max(...testPred), -spread, spread, true,
      `Residuals  RMSE=${rmse.toFixed(3)}`)
    const [, zeroY] = project(rightAxes, 0, 0)
    cv.line(rightPanel, new cv.Point(rightAxes.left, zeroY), new cv.Point(rightAxes.left + rightAxes.width, zeroY), new cv.Scalar(22, 115, 249, 255), 1)
    const purple = new cv.Scalar(247, 85, 168, 255)
    testPred.forEach((value, i) => {
      const [px, py] = project(rightAxes, value, residuals[i])
      cv.circle(rightPanel, new cv.Point(px, py), 3, purple, -1)
    })
    rightPanel.delete()
  } else {
    text(cv, preview, 'No test data', Math.round(w / 2) - 40, Math.round(h / 2), 0.5)
  }
  text(cv, preview, `R2 train=${r2Train.toFixed(3)}  test=${Number.isNaN(r2Test) ? 'n/a' : r2Test.toFixed(3)}`, 8, h - 6, 0.4)

  // Coefficient bar chart, one horizontal bar per feature.
  const coefHeight = Math.max(160, features.length * 26 + 60)
  const coefPlot = canvas(cv, ctx, Math.max(360, w), coefHeight)
  const reach = Math.max(...weights.map(Math.abs), 1e-9)
  const coefAxes = drawAxes(cv, coefPlot, -reach * 1.1, reach * 1.1, -0.5, features.length - 0.5, true,
    fitIntercept ? `Coefficients (intercept=${intercept.toFixed(3)})` : 'Coefficients')
  const [zeroX] = project(coefAxes, 0, 0)
  features.forEach((name, j) => {
    const [barX, barY] = project(coefAxes, weights[j], features.length - 1 - j)
    const colour = weights[j] >= 0 ? new cv.Scalar(246, 130, 59, 255) : new cv.Scalar(68, 68, 239, 255)
    cv.rectangle(coefPlot, new cv.Point(Math.min(zeroX, barX), barY - 6), new cv.Point(Math.max(zeroX, barX), barY + 6), colour, -1)
    text(cv, coefPlot, name.slice(0, 8), 4, barY + 4, 0.3)
  })

  return {
    main: preview,
    preview,
    coef_plot: coefPlot,
    r2_test: r2Test,
    r2_train: r2Train,
    rmse,
    coefficients: weights,
    intercept,
  }
}

/* ---------------------------------------------------------- cluster validity */

export const mlClusterValidity: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const df = tableIn(inputs, 'table', 'data', 'main')
  if (!df) return {}

  const clusterCol = resolveColumn(df, params.cluster_col) ?? (df.columns.includes('cluster') ? 'cluster' : null)
  if (!clusterCol) return {}
  const features = pickFeatures(df, params.features, clusterCol)
  if (features.length === 0) return {}

  const rows: number[][] = []
  const rawLabels: number[] = []
  for (const record of df.rows) {
    const values = features.map((f) => Number(record[f]))
    const label = record[clusterCol]
    if (values.every((v) => Number.isFinite(v)) && !isNa(label)) {
      rows.push(values)
      rawLabels.push(Math.round(Number(label)))
    }
  }
  const labels = Int32Array.from(rawLabels)
  const k = new Set(rawLabels).size
  if (k < 2 || rows.length <= k) return {}

  const matrix = makeMatrix(rows)
  const X = params.standardize !== false ? standardize(matrix).scaled : matrix

  const ch = calinskiHarabasz(X, labels)
  const db = daviesBouldin(X, labels)
  const dunn = dunnIndex(X, labels)
  const scores = { calinski_harabasz: ch, davies_bouldin: db, dunn, k }

  let preview: any = null
  if (params.show_plot !== false) {
    const [w, h] = previewSize(inputs.img_size, { width: 440, height: 300, ...params })
    preview = canvas(cv, ctx, w, h)
    text(cv, preview, `Cluster validity (k=${k})`, 8, 20, 0.42)
    const names = ['Calinski-H+', 'Davies-B-', 'Dunn+']
    const values = [ch, db, dunn]
    const colours = [new cv.Scalar(246, 156, 91, 255), new cv.Scalar(11, 158, 245, 255), new cv.Scalar(153, 211, 52, 255)]
    // Each bar is scaled to its own value, since the three indices live on
    // completely different ranges — the printed number carries the magnitude.
    const top = 40
    const bottom = h - 34
    const reach = Math.max(...values.map(Math.abs), 1e-9)
    names.forEach((name, i) => {
      const width = Math.floor(w / 3)
      const x0 = i * width + 18
      const barHeight = Math.round(((Math.abs(values[i]) / reach) * (bottom - top - 16)))
      cv.rectangle(preview, new cv.Point(x0, bottom - barHeight), new cv.Point(x0 + width - 36, bottom), colours[i], -1)
      text(cv, preview, name, x0, h - 16, 0.32)
      text(cv, preview, values[i].toPrecision(4), x0, bottom - barHeight - 5, 0.32)
    })
  }

  return {
    calinski_harabasz: ch,
    davies_bouldin: db,
    dunn,
    main: preview,
    preview,
    df_meta: scores,
  }
}

/* ------------------------------------------------- classification report node */

function parseLabelMap(raw: unknown): Map<string, string> {
  const map = new Map<string, string>()
  for (const item of String(raw ?? '').split(',')) {
    const at = item.indexOf('=')
    if (at > 0) map.set(item.slice(0, at).trim(), item.slice(at + 1).trim())
  }
  return map
}

export const mlClassificationReport: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const report = inputs.report_data as Record<string, unknown> | undefined
  if (!report || typeof report !== 'object') return {}

  const targetA = String(params.class_target_a ?? '95').trim()
  const targetB = String(params.class_target_b ?? '60').trim()
  const high = Number(params.f1_threshold_high ?? 0.85)
  const low = Number(params.f1_threshold_low ?? 0.75)
  const labelMap = parseLabelMap(params.class_labels)

  const skip = new Set(['accuracy', 'macro avg', 'weighted avg'])
  const classKeys = Object.keys(report)
    .filter((k) => !skip.has(k) && typeof report[k] === 'object' && report[k] !== null)
    // Numeric class codes sort numerically, as the desktop's key does.
    .sort((a, b) => (/^\d+$/.test(a) && /^\d+$/.test(b) ? Number(a) - Number(b) : a < b ? -1 : 1))

  const oa = Number((report as Record<string, unknown>).accuracy ?? 0)
  const metricsOf = (key: string): ClassMetrics | null => {
    const value = report[key]
    return value && typeof value === 'object' ? (value as ClassMetrics) : null
  }
  const f1A = metricsOf(targetA)?.['f1-score'] ?? 0
  const f1B = metricsOf(targetB)?.['f1-score'] ?? 0

  const conf = inputs.conf_matrix as { normalized?: number[][]; labels?: unknown[]; original_classes?: unknown[] } | undefined
  const hasMatrix = Boolean(conf && Array.isArray(conf.normalized) && Array.isArray(conf.labels))

  const width = 760
  const rows = [...classKeys, ...['macro avg', 'weighted avg'].filter((k) => report[k]), 'Overall Accuracy']
  const matrixSize = hasMatrix ? Math.min(360, 60 + conf!.normalized!.length * 42) : 0
  const height = 60 + matrixSize + rows.length * 22 + 40
  const img = canvas(cv, ctx, width, height)

  const nameOf = (key: string) => labelMap.get(key) ?? key
  text(cv, img, `Classification Report  ·  OA = ${(oa * 100).toFixed(1)}%  ·  F1(${nameOf(targetA)}) = ${f1A.toFixed(3)}  ·  F1(${nameOf(targetB)}) = ${f1B.toFixed(3)}`, 8, 20, 0.42)

  let y = 40
  if (hasMatrix) {
    const normalized = conf!.normalized!.map((r) => r.map(Number))
    const displayLabels = (conf!.original_classes?.length ? conf!.original_classes! : conf!.labels!)
      .map((c) => nameOf(String(typeof c === 'number' ? Math.round(c) : c)))
    const n = normalized.length
    const cell = Math.max(18, Math.min(40, Math.floor((matrixSize - 40) / Math.max(1, n))))
    const left = 90
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const value = normalized[i][j] ?? 0
        const shade = new cv.Scalar(255 - value * 100, 255 - value * 180, 255 - value * 235, 255)
        const x0 = left + j * cell
        const y0 = y + i * cell
        cv.rectangle(img, new cv.Point(x0, y0), new cv.Point(x0 + cell, y0 + cell), shade, -1)
        cv.rectangle(img, new cv.Point(x0, y0), new cv.Point(x0 + cell, y0 + cell), new cv.Scalar(70, 70, 70, 255), 1)
        text(cv, img, value.toFixed(2), x0 + 2, y0 + cell / 2 + 4, 0.3,
          value > 0.5 ? new cv.Scalar(255, 255, 255, 255) : new cv.Scalar(51, 51, 51, 255))
      }
      text(cv, img, String(displayLabels[i] ?? i).slice(0, 12), 4, y + i * cell + cell / 2 + 4, 0.32)
    }
    text(cv, img, 'Confusion matrix - row-normalized (diagonal = recall)', left, y + n * cell + 16, 0.34)
    y += matrixSize
  }

  const columns = [8, 300, 400, 500, 600]
  const header = ['Class', 'Precision', 'Recall', 'F1', 'Support']
  header.forEach((label, i) => text(cv, img, label, columns[i], y, 0.36, new cv.Scalar(252, 180, 165, 255)))
  y += 20

  const drawRow = (label: string, metrics: ClassMetrics | null, starred: boolean) => {
    if (metrics) {
      const f1 = metrics['f1-score']
      const band = f1 >= high
        ? new cv.Scalar(183, 231, 110, 255)
        : f1 >= low
          ? new cv.Scalar(61, 211, 252, 255)
          : new cv.Scalar(113, 113, 248, 255)
      text(cv, img, `${starred ? '* ' : '  '}${label}`.slice(0, 30), columns[0], y, 0.35)
      text(cv, img, metrics.precision.toFixed(3), columns[1], y, 0.35)
      text(cv, img, metrics.recall.toFixed(3), columns[2], y, 0.35)
      text(cv, img, f1.toFixed(3), columns[3], y, 0.35, band)
      text(cv, img, String(Math.round(metrics.support)), columns[4], y, 0.35)
    } else {
      text(cv, img, label, columns[0], y, 0.35)
    }
    y += 22
  }

  for (const key of classKeys) drawRow(nameOf(key), metricsOf(key), key === targetA || key === targetB)
  for (const key of ['macro avg', 'weighted avg']) if (report[key]) drawRow(key, metricsOf(key), false)
  text(cv, img, 'Overall Accuracy', columns[0], y, 0.35)
  text(cv, img, oa.toFixed(3), columns[1], y, 0.35)
  y += 22
  text(cv, img, `F1 color: >=${(high * 100).toFixed(0)}% green  ·  >=${(low * 100).toFixed(0)}% amber  ·  else red  ·  * = target class`, 8, y + 6, 0.3, new cv.Scalar(136, 136, 136, 255))

  return { main: img, report: img, f1_main: f1A, f1_b: f1B, oa }
}
