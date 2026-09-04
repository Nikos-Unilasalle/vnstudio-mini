/**
 * The numerical core behind the Machine Learning nodes.
 *
 * The desktop nodes call scikit-learn. There is no scikit-learn in a browser,
 * so the handful of algorithms those nodes actually use are written out here:
 * K-Means with k-means++ seeding, PCA through a Jacobi eigendecomposition,
 * least squares through Householder QR, k-nearest-neighbours, and the cluster
 * and classification metrics. Each follows scikit-learn's own definition —
 * including its ddof choices and its zero-division rules — so a graph built on
 * the desktop reports the same numbers here.
 */

/** A feature matrix, stored row-major: `data[i * d + j]` is sample i, feature j. */
export interface Matrix {
  data: Float64Array
  n: number
  d: number
}

export function makeMatrix(rows: number[][]): Matrix {
  const n = rows.length
  const d = n > 0 ? rows[0].length : 0
  const data = new Float64Array(n * d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) data[i * d + j] = rows[i][j]
  return { data, n, d }
}

export function row(X: Matrix, i: number): Float64Array {
  return X.data.subarray(i * X.d, i * X.d + X.d)
}

/**
 * StandardScaler: centre each column and divide by its population standard
 * deviation (ddof = 0, which is what sklearn uses). A constant column has zero
 * variance; sklearn leaves it alone rather than dividing by zero, and so does this.
 */
export function standardize(X: Matrix): { scaled: Matrix; mean: Float64Array; scale: Float64Array } {
  const { n, d } = X
  const mean = new Float64Array(d)
  const scale = new Float64Array(d)
  for (let j = 0; j < d; j++) {
    let sum = 0
    for (let i = 0; i < n; i++) sum += X.data[i * d + j]
    mean[j] = n > 0 ? sum / n : 0
    let variance = 0
    for (let i = 0; i < n; i++) {
      const delta = X.data[i * d + j] - mean[j]
      variance += delta * delta
    }
    variance = n > 0 ? variance / n : 0
    scale[j] = variance > 1e-12 ? Math.sqrt(variance) : 1
  }
  const data = new Float64Array(n * d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) data[i * d + j] = (X.data[i * d + j] - mean[j]) / scale[j]
  return { scaled: { data, n, d }, mean, scale }
}

export function applyScaler(X: Matrix, mean: Float64Array, scale: Float64Array): Matrix {
  const data = new Float64Array(X.n * X.d)
  for (let i = 0; i < X.n; i++) for (let j = 0; j < X.d; j++) data[i * X.d + j] = (X.data[i * X.d + j] - mean[j]) / scale[j]
  return { data, n: X.n, d: X.d }
}

/** Squared Euclidean distance between sample i of X and a free-standing point. */
function sqDistance(X: Matrix, i: number, point: Float64Array, offset = 0): number {
  let total = 0
  for (let j = 0; j < X.d; j++) {
    const delta = X.data[i * X.d + j] - point[offset + j]
    total += delta * delta
  }
  return total
}

/** Mulberry32 — the same generator `seededRandom` uses, kept local to avoid a cycle. */
function rng(seed: number): () => number {
  let state = (seed >>> 0) || 1
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/* ------------------------------------------------------------------ K-Means */

export interface KMeansResult {
  labels: Int32Array
  /** k × d, row-major. */
  centers: Float64Array
  inertia: number
  iterations: number
}

/**
 * k-means++ seeding: the first centre is uniform, each later one is drawn with
 * probability proportional to its squared distance from the nearest centre
 * already chosen. This is what makes a single Lloyd run reliable enough that
 * the elbow curve is smooth.
 */
function kmeansPlusPlus(X: Matrix, k: number, random: () => number): Float64Array {
  const { n, d } = X
  const centers = new Float64Array(k * d)
  let first = Math.min(n - 1, Math.floor(random() * n))
  for (let j = 0; j < d; j++) centers[j] = X.data[first * d + j]

  const closest = new Float64Array(n)
  for (let i = 0; i < n; i++) closest[i] = sqDistance(X, i, centers, 0)

  for (let c = 1; c < k; c++) {
    let total = 0
    for (let i = 0; i < n; i++) total += closest[i]
    let pick = n - 1
    if (total > 0) {
      let threshold = random() * total
      for (let i = 0; i < n; i++) {
        threshold -= closest[i]
        if (threshold <= 0) { pick = i; break }
      }
    } else {
      pick = Math.min(n - 1, Math.floor(random() * n))
    }
    for (let j = 0; j < d; j++) centers[c * d + j] = X.data[pick * d + j]
    for (let i = 0; i < n; i++) closest[i] = Math.min(closest[i], sqDistance(X, i, centers, c * d))
  }
  return centers
}

function lloyd(X: Matrix, k: number, centers: Float64Array, maxIter: number): KMeansResult {
  const { n, d } = X
  const labels = new Int32Array(n)
  const counts = new Int32Array(k)
  const sums = new Float64Array(k * d)
  let inertia = 0
  let iteration = 0

  for (; iteration < maxIter; iteration++) {
    let moved = false
    inertia = 0
    for (let i = 0; i < n; i++) {
      let best = 0
      let bestDistance = Infinity
      for (let c = 0; c < k; c++) {
        const distance = sqDistance(X, i, centers, c * d)
        if (distance < bestDistance) { bestDistance = distance; best = c }
      }
      inertia += bestDistance
      if (labels[i] !== best) { labels[i] = best; moved = true }
    }
    if (!moved && iteration > 0) break

    counts.fill(0)
    sums.fill(0)
    for (let i = 0; i < n; i++) {
      const c = labels[i]
      counts[c]++
      for (let j = 0; j < d; j++) sums[c * d + j] += X.data[i * d + j]
    }
    for (let c = 0; c < k; c++) {
      // An emptied cluster keeps its previous centre rather than collapsing to
      // the origin, which is what sklearn's relocation ends up doing in effect.
      if (counts[c] === 0) continue
      for (let j = 0; j < d; j++) centers[c * d + j] = sums[c * d + j] / counts[c]
    }
  }

  // One last assignment pass so labels and inertia describe the final centres.
  inertia = 0
  for (let i = 0; i < n; i++) {
    let best = 0
    let bestDistance = Infinity
    for (let c = 0; c < k; c++) {
      const distance = sqDistance(X, i, centers, c * d)
      if (distance < bestDistance) { bestDistance = distance; best = c }
    }
    labels[i] = best
    inertia += bestDistance
  }
  return { labels, centers, inertia, iterations: iteration }
}

export function kmeans(
  X: Matrix,
  k: number,
  options: { init?: 'k-means++' | 'random'; maxIter?: number; nInit?: number; seed?: number } = {}
): KMeansResult {
  const { n, d } = X
  const clusters = Math.max(1, Math.min(k, n))
  const maxIter = options.maxIter ?? 300
  const nInit = Math.max(1, options.nInit ?? 10)
  const random = rng(options.seed ?? 42)

  let best: KMeansResult | null = null
  for (let attempt = 0; attempt < nInit; attempt++) {
    let start: Float64Array
    if (options.init === 'random') {
      start = new Float64Array(clusters * d)
      const chosen = new Set<number>()
      for (let c = 0; c < clusters; c++) {
        let pick = Math.min(n - 1, Math.floor(random() * n))
        while (chosen.has(pick) && chosen.size < n) pick = (pick + 1) % n
        chosen.add(pick)
        for (let j = 0; j < d; j++) start[c * d + j] = X.data[pick * d + j]
      }
    } else {
      start = kmeansPlusPlus(X, clusters, random)
    }
    const result = lloyd(X, clusters, start, maxIter)
    if (!best || result.inertia < best.inertia) best = result
  }
  return best!
}

/** Assigns each row of `points` to the nearest of `centers` (k × d, row-major). */
export function assignClusters(points: Matrix, centers: Float64Array, k: number): Int32Array {
  const labels = new Int32Array(points.n)
  for (let i = 0; i < points.n; i++) {
    let best = 0
    let bestDistance = Infinity
    for (let c = 0; c < k; c++) {
      const distance = sqDistance(points, i, centers, c * points.d)
      if (distance < bestDistance) { bestDistance = distance; best = c }
    }
    labels[i] = best
  }
  return labels
}

/**
 * Mean silhouette coefficient, sklearn's `silhouette_score`: for each sample,
 * (b - a) / max(a, b) where a is the mean distance to its own cluster and b the
 * smallest mean distance to any other. A singleton cluster scores 0 by definition.
 */
export function silhouetteScore(X: Matrix, labels: Int32Array): number {
  const { n } = X
  if (n < 2) return 0
  const clusters = [...new Set([...labels])].sort((a, b) => a - b)
  if (clusters.length < 2) return 0
  const index = new Map(clusters.map((c, i) => [c, i]))
  const sizes = new Int32Array(clusters.length)
  for (let i = 0; i < n; i++) sizes[index.get(labels[i])!]++

  let total = 0
  const sums = new Float64Array(clusters.length)
  for (let i = 0; i < n; i++) {
    sums.fill(0)
    for (let j = 0; j < n; j++) {
      if (i === j) continue
      let distance = 0
      for (let f = 0; f < X.d; f++) {
        const delta = X.data[i * X.d + f] - X.data[j * X.d + f]
        distance += delta * delta
      }
      sums[index.get(labels[j])!] += Math.sqrt(distance)
    }
    const own = index.get(labels[i])!
    if (sizes[own] <= 1) continue   // silhouette of a singleton is 0
    const a = sums[own] / (sizes[own] - 1)
    let b = Infinity
    for (let c = 0; c < clusters.length; c++) {
      if (c === own || sizes[c] === 0) continue
      b = Math.min(b, sums[c] / sizes[c])
    }
    total += (b - a) / Math.max(a, b)
  }
  return total / n
}

/* ---------------------------------------------------------------------- PCA */

/**
 * Eigendecomposition of a symmetric matrix by the cyclic Jacobi method.
 * Returns eigenvalues in descending order with their eigenvectors as columns.
 */
export function jacobiEigen(input: Float64Array, d: number): { values: Float64Array; vectors: Float64Array } {
  const a = Float64Array.from(input)
  const v = new Float64Array(d * d)
  for (let i = 0; i < d; i++) v[i * d + i] = 1

  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0
    for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) off += a[p * d + q] * a[p * d + q]
    if (off < 1e-24) break

    for (let p = 0; p < d - 1; p++) {
      for (let q = p + 1; q < d; q++) {
        const apq = a[p * d + q]
        if (Math.abs(apq) < 1e-30) continue
        const theta = (a[q * d + q] - a[p * d + p]) / (2 * apq)
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        for (let i = 0; i < d; i++) {
          const aip = a[i * d + p]
          const aiq = a[i * d + q]
          a[i * d + p] = c * aip - s * aiq
          a[i * d + q] = s * aip + c * aiq
        }
        for (let i = 0; i < d; i++) {
          const api = a[p * d + i]
          const aqi = a[q * d + i]
          a[p * d + i] = c * api - s * aqi
          a[q * d + i] = s * api + c * aqi
        }
        for (let i = 0; i < d; i++) {
          const vip = v[i * d + p]
          const viq = v[i * d + q]
          v[i * d + p] = c * vip - s * viq
          v[i * d + q] = s * vip + c * viq
        }
      }
    }
  }

  const order = [...Array(d).keys()].sort((x, y) => a[y * d + y] - a[x * d + x])
  const values = new Float64Array(d)
  const vectors = new Float64Array(d * d)
  for (let k = 0; k < d; k++) {
    values[k] = a[order[k] * d + order[k]]
    // Sign is arbitrary in an eigendecomposition; fixing the largest-magnitude
    // entry positive makes the projection reproducible from run to run.
    let pivot = 0
    for (let i = 1; i < d; i++) if (Math.abs(v[i * d + order[k]]) > Math.abs(v[pivot * d + order[k]])) pivot = i
    const sign = v[pivot * d + order[k]] < 0 ? -1 : 1
    for (let i = 0; i < d; i++) vectors[i * d + k] = sign * v[i * d + order[k]]
  }
  return { values, vectors }
}

export interface PcaResult {
  /** nComp × d, row-major: each row is a principal axis in feature space. */
  components: Float64Array
  nComp: number
  mean: Float64Array
  explainedVariance: Float64Array
  explainedVarianceRatio: Float64Array
  /** n × nComp, row-major: the input projected onto the components. */
  scores: Float64Array
}

/** PCA on the covariance matrix (ddof = 1, matching sklearn's `explained_variance_`). */
export function pca(X: Matrix, nComponents: number): PcaResult {
  const { n, d } = X
  const nComp = Math.max(1, Math.min(nComponents, d, n))
  const mean = new Float64Array(d)
  for (let j = 0; j < d; j++) {
    let sum = 0
    for (let i = 0; i < n; i++) sum += X.data[i * d + j]
    mean[j] = n > 0 ? sum / n : 0
  }
  const centred = new Float64Array(n * d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) centred[i * d + j] = X.data[i * d + j] - mean[j]

  const cov = new Float64Array(d * d)
  const denominator = Math.max(1, n - 1)
  for (let j = 0; j < d; j++) {
    for (let k = j; k < d; k++) {
      let sum = 0
      for (let i = 0; i < n; i++) sum += centred[i * d + j] * centred[i * d + k]
      cov[j * d + k] = cov[k * d + j] = sum / denominator
    }
  }

  const { values, vectors } = jacobiEigen(cov, d)
  let totalVariance = 0
  for (let j = 0; j < d; j++) totalVariance += Math.max(0, values[j])

  const components = new Float64Array(nComp * d)
  const explainedVariance = new Float64Array(nComp)
  const explainedVarianceRatio = new Float64Array(nComp)
  for (let c = 0; c < nComp; c++) {
    explainedVariance[c] = values[c]
    explainedVarianceRatio[c] = totalVariance > 0 ? values[c] / totalVariance : 0
    for (let j = 0; j < d; j++) components[c * d + j] = vectors[j * d + c]
  }

  const scores = new Float64Array(n * nComp)
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nComp; c++) {
      let sum = 0
      for (let j = 0; j < d; j++) sum += centred[i * d + j] * components[c * d + j]
      scores[i * nComp + c] = sum
    }
  }
  return { components, nComp, mean, explainedVariance, explainedVarianceRatio, scores }
}

/** Maps a point in component space back to feature space — PCA's `inverse_transform`. */
export function pcaInverse(model: PcaResult, coordinates: number[]): Float64Array {
  const d = model.mean.length
  const out = Float64Array.from(model.mean)
  for (let c = 0; c < model.nComp && c < coordinates.length; c++) {
    for (let j = 0; j < d; j++) out[j] += coordinates[c] * model.components[c * d + j]
  }
  return out
}

/* -------------------------------------------------------- least squares */

/**
 * Least-squares solution of A·x = b by Householder QR.
 *
 * The normal equations would square the condition number, which shows up as
 * nonsense coefficients the moment two features are correlated — exactly the
 * case a regression node gets pointed at. QR avoids that.
 */
export function leastSquares(A: number[][], b: number[]): Float64Array {
  const m = A.length
  const n = m > 0 ? A[0].length : 0
  if (m === 0 || n === 0) return new Float64Array(n)

  const r = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) r[i * n + j] = A[i][j]
  const y = Float64Array.from(b)

  for (let k = 0; k < Math.min(n, m - 1); k++) {
    let norm = 0
    for (let i = k; i < m; i++) norm += r[i * n + k] * r[i * n + k]
    norm = Math.sqrt(norm)
    if (norm < 1e-300) continue
    const alpha = r[k * n + k] > 0 ? -norm : norm
    const v = new Float64Array(m)
    for (let i = k; i < m; i++) v[i] = r[i * n + k]
    v[k] -= alpha
    let vNorm = 0
    for (let i = k; i < m; i++) vNorm += v[i] * v[i]
    if (vNorm < 1e-300) continue

    for (let j = k; j < n; j++) {
      let dot = 0
      for (let i = k; i < m; i++) dot += v[i] * r[i * n + j]
      const factor = (2 * dot) / vNorm
      for (let i = k; i < m; i++) r[i * n + j] -= factor * v[i]
    }
    let dot = 0
    for (let i = k; i < m; i++) dot += v[i] * y[i]
    const factor = (2 * dot) / vNorm
    for (let i = k; i < m; i++) y[i] -= factor * v[i]
  }

  const x = new Float64Array(n)
  for (let i = Math.min(n, m) - 1; i >= 0; i--) {
    let sum = y[i]
    for (let j = i + 1; j < n; j++) sum -= r[i * n + j] * x[j]
    // A rank-deficient column contributes nothing; leaving its coefficient at
    // zero is the minimum-norm choice, which is what lstsq returns too.
    x[i] = Math.abs(r[i * n + i]) > 1e-10 ? sum / r[i * n + i] : 0
  }
  return x
}

/** Coefficient of determination, sklearn's `.score()` for a regressor. */
export function r2Score(yTrue: number[], yPred: number[]): number {
  const n = yTrue.length
  if (n === 0) return NaN
  const mean = yTrue.reduce((a, b) => a + b, 0) / n
  let residual = 0
  let total = 0
  for (let i = 0; i < n; i++) {
    residual += (yTrue[i] - yPred[i]) ** 2
    total += (yTrue[i] - mean) ** 2
  }
  return total > 0 ? 1 - residual / total : 0
}

/* ------------------------------------------------------------------- KNN */

export type Metric = 'euclidean' | 'manhattan' | 'chebyshev' | 'minkowski'

function distance(a: Float64Array | number[], b: Float64Array, d: number, metric: Metric): number {
  let total = 0
  switch (metric) {
    case 'manhattan':
      for (let j = 0; j < d; j++) total += Math.abs(a[j] - b[j])
      return total
    case 'chebyshev':
      for (let j = 0; j < d; j++) total = Math.max(total, Math.abs(a[j] - b[j]))
      return total
    default:
      // sklearn's `minkowski` defaults to p = 2, so it coincides with euclidean.
      for (let j = 0; j < d; j++) total += (a[j] - b[j]) ** 2
      return Math.sqrt(total)
  }
}

export interface KnnModel {
  X: Matrix
  y: Int32Array
  k: number
  metric: Metric
  weighted: boolean
  nClasses: number
}

export function knnFit(X: Matrix, y: Int32Array, k: number, metric: Metric, weighted: boolean, nClasses: number): KnnModel {
  return { X, y, k: Math.max(1, Math.min(k, X.n)), metric, weighted, nClasses }
}

export function knnPredictOne(model: KnnModel, point: Float64Array | number[]): number {
  const { X, y, k, metric, weighted, nClasses } = model
  const neighbours: { distance: number; index: number }[] = []
  for (let i = 0; i < X.n; i++) {
    neighbours.push({ distance: distance(point, row(X, i), X.d, metric), index: i })
  }
  // Ties on distance are broken by sample order, so a prediction never depends
  // on the sort's stability.
  neighbours.sort((a, b) => a.distance - b.distance || a.index - b.index)

  const votes = new Float64Array(nClasses)
  for (let i = 0; i < k; i++) {
    const neighbour = neighbours[i]
    if (weighted) {
      // sklearn's 'distance' weighting: an exact match takes the vote outright.
      if (neighbour.distance === 0) {
        votes.fill(0)
        votes[y[neighbour.index]] = 1
        break
      }
      votes[y[neighbour.index]] += 1 / neighbour.distance
    } else {
      votes[y[neighbour.index]] += 1
    }
  }
  let best = 0
  for (let c = 1; c < nClasses; c++) if (votes[c] > votes[best]) best = c
  return best
}

export function knnPredict(model: KnnModel, points: Matrix): Int32Array {
  const out = new Int32Array(points.n)
  for (let i = 0; i < points.n; i++) out[i] = knnPredictOne(model, row(points, i))
  return out
}

/* --------------------------------------------------------------- metrics */

export function confusionMatrix(yTrue: Int32Array | number[], yPred: Int32Array | number[], nClasses: number): number[][] {
  const cm: number[][] = Array.from({ length: nClasses }, () => new Array(nClasses).fill(0))
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i]
    const p = yPred[i]
    if (t >= 0 && t < nClasses && p >= 0 && p < nClasses) cm[t][p]++
  }
  return cm
}

export interface ClassMetrics {
  precision: number
  recall: number
  'f1-score': number
  support: number
}

export type ClassificationReport = Record<string, ClassMetrics | number>

/**
 * sklearn's `classification_report(output_dict=True, zero_division=0)`: per-class
 * precision/recall/F1/support, then macro and weighted averages, plus accuracy.
 */
export function classificationReport(
  yTrue: Int32Array | number[],
  yPred: Int32Array | number[],
  classNames: string[]
): ClassificationReport {
  const nClasses = classNames.length
  const cm = confusionMatrix(yTrue, yPred, nClasses)
  const report: ClassificationReport = {}

  let correct = 0
  let totalSupport = 0
  const perClass: ClassMetrics[] = []
  for (let c = 0; c < nClasses; c++) {
    const tp = cm[c][c]
    let predicted = 0
    let support = 0
    for (let i = 0; i < nClasses; i++) {
      predicted += cm[i][c]
      support += cm[c][i]
    }
    correct += tp
    totalSupport += support
    const precision = predicted > 0 ? tp / predicted : 0
    const recall = support > 0 ? tp / support : 0
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0
    const metrics: ClassMetrics = { precision, recall, 'f1-score': f1, support }
    perClass.push(metrics)
    report[classNames[c]] = metrics
  }

  report.accuracy = totalSupport > 0 ? correct / totalSupport : 0
  const macro = (pick: (m: ClassMetrics) => number) =>
    nClasses > 0 ? perClass.reduce((a, m) => a + pick(m), 0) / nClasses : 0
  const weighted = (pick: (m: ClassMetrics) => number) =>
    totalSupport > 0 ? perClass.reduce((a, m) => a + pick(m) * m.support, 0) / totalSupport : 0

  report['macro avg'] = {
    precision: macro((m) => m.precision),
    recall: macro((m) => m.recall),
    'f1-score': macro((m) => m['f1-score']),
    support: totalSupport,
  }
  report['weighted avg'] = {
    precision: weighted((m) => m.precision),
    recall: weighted((m) => m.recall),
    'f1-score': weighted((m) => m['f1-score']),
    support: totalSupport,
  }
  return report
}

function centroidsOf(X: Matrix, labels: Int32Array, groups: number[]): Float64Array {
  const centroids = new Float64Array(groups.length * X.d)
  const counts = new Int32Array(groups.length)
  const index = new Map(groups.map((g, i) => [g, i]))
  for (let i = 0; i < X.n; i++) {
    const g = index.get(labels[i])
    if (g === undefined) continue
    counts[g]++
    for (let j = 0; j < X.d; j++) centroids[g * X.d + j] += X.data[i * X.d + j]
  }
  for (let g = 0; g < groups.length; g++) {
    if (counts[g] === 0) continue
    for (let j = 0; j < X.d; j++) centroids[g * X.d + j] /= counts[g]
  }
  return centroids
}

/** Calinski-Harabasz: between-cluster over within-cluster dispersion. Higher is better. */
export function calinskiHarabasz(X: Matrix, labels: Int32Array): number {
  const groups = [...new Set([...labels])].sort((a, b) => a - b)
  const k = groups.length
  const n = X.n
  if (k < 2 || n <= k) return 0
  const centroids = centroidsOf(X, labels, groups)
  const index = new Map(groups.map((g, i) => [g, i]))

  const overall = new Float64Array(X.d)
  for (let i = 0; i < n; i++) for (let j = 0; j < X.d; j++) overall[j] += X.data[i * X.d + j]
  for (let j = 0; j < X.d; j++) overall[j] /= n

  const counts = new Int32Array(k)
  for (let i = 0; i < n; i++) counts[index.get(labels[i])!]++

  let between = 0
  for (let g = 0; g < k; g++) {
    let sum = 0
    for (let j = 0; j < X.d; j++) sum += (centroids[g * X.d + j] - overall[j]) ** 2
    between += counts[g] * sum
  }
  let within = 0
  for (let i = 0; i < n; i++) {
    const g = index.get(labels[i])!
    for (let j = 0; j < X.d; j++) within += (X.data[i * X.d + j] - centroids[g * X.d + j]) ** 2
  }
  if (within === 0) return 0
  return (between / within) * ((n - k) / (k - 1))
}

/** Davies-Bouldin: mean worst-case cluster similarity. Lower is better. */
export function daviesBouldin(X: Matrix, labels: Int32Array): number {
  const groups = [...new Set([...labels])].sort((a, b) => a - b)
  const k = groups.length
  if (k < 2) return 0
  const centroids = centroidsOf(X, labels, groups)
  const index = new Map(groups.map((g, i) => [g, i]))

  const spread = new Float64Array(k)
  const counts = new Int32Array(k)
  for (let i = 0; i < X.n; i++) {
    const g = index.get(labels[i])!
    let sum = 0
    for (let j = 0; j < X.d; j++) sum += (X.data[i * X.d + j] - centroids[g * X.d + j]) ** 2
    spread[g] += Math.sqrt(sum)
    counts[g]++
  }
  for (let g = 0; g < k; g++) if (counts[g] > 0) spread[g] /= counts[g]

  let total = 0
  for (let a = 0; a < k; a++) {
    let worst = 0
    for (let b = 0; b < k; b++) {
      if (a === b) continue
      let separation = 0
      for (let j = 0; j < X.d; j++) separation += (centroids[a * X.d + j] - centroids[b * X.d + j]) ** 2
      separation = Math.sqrt(separation)
      if (separation > 0) worst = Math.max(worst, (spread[a] + spread[b]) / separation)
    }
    total += worst
  }
  return total / k
}

/**
 * Dunn index, in the centroid-based form the desktop node uses: the smallest
 * distance between two centroids over the widest cluster diameter (twice the
 * furthest point-to-centroid distance). Higher is better.
 */
export function dunnIndex(X: Matrix, labels: Int32Array): number {
  const groups = [...new Set([...labels])].sort((a, b) => a - b)
  const k = groups.length
  if (k < 2) return 0
  const centroids = centroidsOf(X, labels, groups)
  const index = new Map(groups.map((g, i) => [g, i]))

  let maxIntra = 0
  for (let i = 0; i < X.n; i++) {
    const g = index.get(labels[i])!
    let sum = 0
    for (let j = 0; j < X.d; j++) sum += (X.data[i * X.d + j] - centroids[g * X.d + j]) ** 2
    maxIntra = Math.max(maxIntra, Math.sqrt(sum) * 2)
  }
  if (maxIntra <= 0) return 0

  let minInter = Infinity
  for (let a = 0; a < k; a++) {
    for (let b = a + 1; b < k; b++) {
      let sum = 0
      for (let j = 0; j < X.d; j++) sum += (centroids[a * X.d + j] - centroids[b * X.d + j]) ** 2
      minInter = Math.min(minInter, Math.sqrt(sum))
    }
  }
  return Number.isFinite(minInter) ? minInter / maxIntra : 0
}

/**
 * A seeded shuffle of 0..n-1. sklearn's split rides on numpy's Mersenne
 * Twister, which is not reproducible here; what matters for the node is that
 * the same seed always yields the same partition, and this delivers that.
 */
export function shuffledIndices(n: number, seed: number): number[] {
  const order = [...Array(n).keys()]
  const random = rng(seed)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    const t = order[i]
    order[i] = order[j]
    order[j] = t
  }
  return order
}

/* ------------------------------------------------- robust straight-line fits */

/** Ordinary least squares on one predictor: the polyfit degree-1 baseline. */
export function lineFitL2(x: number[], y: number[]): { slope: number; intercept: number } {
  const n = x.length
  if (n < 2) return { slope: 0, intercept: 0 }
  const meanX = x.reduce((a, b) => a + b, 0) / n
  const meanY = y.reduce((a, b) => a + b, 0) / n
  let covariance = 0
  let variance = 0
  for (let i = 0; i < n; i++) {
    covariance += (x[i] - meanX) * (y[i] - meanY)
    variance += (x[i] - meanX) ** 2
  }
  const slope = variance > 0 ? covariance / variance : 0
  return { slope, intercept: meanY - slope * meanX }
}

/**
 * Huber regression, minimising sklearn's own objective
 *
 *   Σ ( σ + H_ε((x_i·w + c − y_i)/σ)·σ ) + α‖w‖²,   H_ε(z) = z² for |z| < ε,
 *                                                            2ε|z| − ε² beyond
 *
 * jointly over the slope, the intercept and the scale σ. sklearn hands this to
 * L-BFGS-B; the objective is convex, so gradient descent with a backtracking
 * line search reaches the same optimum. Fitting σ rather than fixing it at the
 * MAD is what makes the result match the desktop instead of merely resembling it.
 */
export function lineFitHuber(x: number[], y: number[], epsilon: number, alpha = 1e-4): { slope: number; intercept: number } {
  const n = x.length
  if (n < 2) return { slope: 0, intercept: 0 }
  const eps = Math.max(1.0001, epsilon)

  const start = lineFitL2(x, y)
  const residuals = x.map((xi, i) => Math.abs(y[i] - (start.slope * xi + start.intercept)))
  const sorted = [...residuals].sort((a, b) => a - b)
  // A scale of zero would divide by nothing, so a perfect fit starts at 1.
  let scale = Math.max(1e-6, sorted[Math.floor(n / 2)] || 1)
  let slope = start.slope
  let intercept = start.intercept

  const objectiveAndGradient = (w: number, c: number, s: number): { value: number; gw: number; gc: number; gs: number } => {
    let value = alpha * w * w
    let gw = 2 * alpha * w
    let gc = 0
    let gs = 0
    for (let i = 0; i < n; i++) {
      const residual = w * x[i] + c - y[i]
      const z = residual / s
      if (Math.abs(z) < eps) {
        value += s + (z * z) * s
        // d/dw of (r²/σ) is 2r·x/σ; d/dσ of (σ + r²/σ) is 1 − r²/σ².
        gw += (2 * residual * x[i]) / s
        gc += (2 * residual) / s
        gs += 1 - z * z
      } else {
        value += s + (2 * eps * Math.abs(z) - eps * eps) * s
        const sign = residual >= 0 ? 1 : -1
        gw += 2 * eps * sign * x[i]
        gc += 2 * eps * sign
        gs += 1 - eps * eps
      }
    }
    return { value, gw, gc, gs }
  }

  let step = 1e-3
  let current = objectiveAndGradient(slope, intercept, scale)
  for (let iteration = 0; iteration < 4000; iteration++) {
    const gradientNorm = Math.hypot(current.gw, current.gc, current.gs)
    if (gradientNorm < 1e-11) break

    let taken = false
    for (let attempt = 0; attempt < 40; attempt++) {
      const w = slope - step * current.gw
      const c = intercept - step * current.gc
      // σ must stay positive; the barrier is a floor rather than a penalty.
      const s = Math.max(1e-9, scale - step * current.gs)
      const trial = objectiveAndGradient(w, c, s)
      if (trial.value < current.value) {
        slope = w
        intercept = c
        scale = s
        current = trial
        step *= 1.6
        taken = true
        break
      }
      step *= 0.4
    }
    if (!taken) break
  }
  return { slope, intercept }
}

/**
 * Theil-Sen, in sklearn's formulation: fit a line through every pair of points,
 * then take the spatial (L1) median of those (intercept, slope) estimates by
 * Weiszfeld iteration. That is the same as `TheilSenRegressor` for a single
 * predictor, where n_subsamples is 2 and the whole population of pairs fits
 * under max_subpopulation.
 */
export function lineFitTheilSen(x: number[], y: number[]): { slope: number; intercept: number } {
  const n = x.length
  if (n < 2) return { slope: 0, intercept: 0 }
  const estimates: [number, number][] = []
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = x[j] - x[i]
      if (Math.abs(dx) < 1e-12) continue
      const slope = (y[j] - y[i]) / dx
      estimates.push([y[i] - slope * x[i], slope])
    }
  }
  if (estimates.length === 0) return lineFitL2(x, y)

  // Weiszfeld: start at the componentwise median, then iterate the
  // inverse-distance-weighted mean until it stops moving.
  const axis = (k: number) => {
    const values = estimates.map((e) => e[k]).sort((a, b) => a - b)
    const mid = values.length >> 1
    return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2
  }
  let point: [number, number] = [axis(0), axis(1)]
  for (let iteration = 0; iteration < 300; iteration++) {
    let weightSum = 0
    let sumA = 0
    let sumB = 0
    let atVertex = false
    for (const [a, b] of estimates) {
      const distance = Math.hypot(a - point[0], b - point[1])
      if (distance < 1e-12) { atVertex = true; continue }
      const weight = 1 / distance
      weightSum += weight
      sumA += a * weight
      sumB += b * weight
    }
    if (weightSum === 0 || atVertex) break
    const next: [number, number] = [sumA / weightSum, sumB / weightSum]
    const moved = Math.hypot(next[0] - point[0], next[1] - point[1])
    point = next
    if (moved < 1e-12) break
  }
  return { intercept: point[0], slope: point[1] }
}
