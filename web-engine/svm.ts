/**
 * A C-SVC support vector machine, solved with libsvm's SMO.
 *
 * scikit-learn's `SVC` is a thin wrapper over libsvm, so this follows libsvm's
 * algorithm rather than a textbook SMO: the same second-order working-set
 * selection, the same 1e-3 stopping tolerance, the same one-against-one
 * decomposition for more than two classes. That is what makes the decision
 * boundary here the one the desktop node draws.
 */
import type { Matrix } from './ml'

export type Kernel = 'rbf' | 'linear' | 'poly' | 'sigmoid'

export interface SvmOptions {
  C?: number
  kernel?: Kernel
  /** 'scale' (sklearn's default) or 'auto'. */
  gamma?: 'scale' | 'auto' | number
  degree?: number
  coef0?: number
  tolerance?: number
  maxIterations?: number
}

/** sklearn's gamma: 'scale' is 1/(d·Var(X)), 'auto' is 1/d. */
export function resolveGamma(X: Matrix, gamma: SvmOptions['gamma']): number {
  if (typeof gamma === 'number') return gamma
  if (gamma === 'auto') return X.d > 0 ? 1 / X.d : 1
  let mean = 0
  for (let i = 0; i < X.data.length; i++) mean += X.data[i]
  mean /= Math.max(1, X.data.length)
  let variance = 0
  for (let i = 0; i < X.data.length; i++) variance += (X.data[i] - mean) ** 2
  variance /= Math.max(1, X.data.length)
  // sklearn guards a degenerate (zero-variance) matrix by falling back to 1.
  return variance > 0 ? 1 / (X.d * variance) : 1
}

function dot(X: Matrix, i: number, j: number): number {
  let sum = 0
  const a = i * X.d
  const b = j * X.d
  for (let k = 0; k < X.d; k++) sum += X.data[a + k] * X.data[b + k]
  return sum
}

function squaredDistance(X: Matrix, i: number, j: number): number {
  let sum = 0
  const a = i * X.d
  const b = j * X.d
  for (let k = 0; k < X.d; k++) sum += (X.data[a + k] - X.data[b + k]) ** 2
  return sum
}

export interface BinaryModel {
  /** Indices into the training matrix of the support vectors. */
  supportIndices: number[]
  /** y_i * alpha_i for each support vector, libsvm's `sv_coef`. */
  coefficients: number[]
  rho: number
}

interface KernelContext {
  X: Matrix
  kernel: Kernel
  gamma: number
  degree: number
  coef0: number
}

function kernelValue(k: KernelContext, i: number, j: number): number {
  switch (k.kernel) {
    case 'linear':
      return dot(k.X, i, j)
    case 'poly':
      return (k.gamma * dot(k.X, i, j) + k.coef0) ** k.degree
    case 'sigmoid':
      return Math.tanh(k.gamma * dot(k.X, i, j) + k.coef0)
    default:
      return Math.exp(-k.gamma * squaredDistance(k.X, i, j))
  }
}

/**
 * Solves one binary sub-problem over `indices` (labels +1 / -1 in `y`).
 *
 * The dual is  min ½aᵀQa − eᵀa  subject to  0 ≤ a ≤ C  and  yᵀa = 0, with
 * Q_ij = y_i·y_j·K(x_i, x_j). Two multipliers move per step, chosen by the
 * pair that promises the largest decrease in the objective.
 */
function solveBinary(k: KernelContext, indices: number[], y: Int8Array, C: number, tolerance: number, maxIterations: number): BinaryModel {
  const n = indices.length
  const alpha = new Float64Array(n)
  // G_t = Σ_j Q_tj·a_j − 1; every alpha starts at zero, so G starts at −1.
  const G = new Float64Array(n).fill(-1)
  const TAU = 1e-12

  // Kernel rows are recomputed rather than cached wholesale: a teaching-sized
  // problem stays fast, and a large one would not fit a dense n×n cache anyway.
  const rowCache = new Map<number, Float64Array>()
  const kernelRow = (t: number): Float64Array => {
    const cached = rowCache.get(t)
    if (cached) return cached
    const row = new Float64Array(n)
    for (let s = 0; s < n; s++) row[s] = kernelValue(k, indices[t], indices[s])
    // Bound the cache so a big problem cannot exhaust memory.
    if (rowCache.size < 2000) rowCache.set(t, row)
    return row
  }
  const diagonal = new Float64Array(n)
  for (let t = 0; t < n; t++) diagonal[t] = kernelValue(k, indices[t], indices[t])

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    // --- working set selection (libsvm's WSS 3) ---
    let i = -1
    let gMax = -Infinity
    let gMin = Infinity
    for (let t = 0; t < n; t++) {
      const up = y[t] > 0 ? alpha[t] < C : alpha[t] > 0
      if (!up) continue
      const value = -y[t] * G[t]
      if (value > gMax) { gMax = value; i = t }
    }
    if (i < 0) break

    const rowI = kernelRow(i)
    let j = -1
    let bestGain = 0
    for (let t = 0; t < n; t++) {
      const low = y[t] > 0 ? alpha[t] > 0 : alpha[t] < C
      if (!low) continue
      const value = -y[t] * G[t]
      if (value < gMin) gMin = value
      const b = gMax - value
      if (b <= 0) continue
      // libsvm writes this y-scaled, but the y factors always cancel: for both
      // equal and opposite labels it reduces to K_ii + K_tt - 2K_it.
      let a = diagonal[i] + diagonal[t] - 2 * rowI[t]
      if (a <= 0) a = TAU
      const gain = (b * b) / a
      if (gain > bestGain) { bestGain = gain; j = t }
    }
    if (j < 0 || gMax - gMin < tolerance) break

    // --- analytic update of the pair (i, j) ---
    const rowJ = kernelRow(j)
    let a = diagonal[i] + diagonal[j] - 2 * rowI[j]
    if (a <= 0) a = TAU
    const oldI = alpha[i]
    const oldJ = alpha[j]

    if (y[i] !== y[j]) {
      const delta = (-G[i] - G[j]) / a
      const diff = alpha[i] - alpha[j]
      alpha[i] += delta
      alpha[j] += delta
      if (diff > 0) {
        if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = diff }
      } else if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = -diff }
      if (diff > 0) {
        if (alpha[i] > C) { alpha[i] = C; alpha[j] = C - diff }
      } else if (alpha[j] > C) { alpha[j] = C; alpha[i] = C + diff }
    } else {
      const delta = (G[i] - G[j]) / a
      const sum = alpha[i] + alpha[j]
      alpha[i] -= delta
      alpha[j] += delta
      if (sum > C) {
        if (alpha[i] > C) { alpha[i] = C; alpha[j] = sum - C }
      } else if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = sum }
      if (sum > C) {
        if (alpha[j] > C) { alpha[j] = C; alpha[i] = sum - C }
      } else if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = sum }
    }

    const deltaI = alpha[i] - oldI
    const deltaJ = alpha[j] - oldJ
    for (let t = 0; t < n; t++) {
      G[t] += y[t] * y[i] * rowI[t] * deltaI + y[t] * y[j] * rowJ[t] * deltaJ
    }
  }

  // --- rho, averaged over the free support vectors when there are any ---
  // libsvm works with yG = y_t·G_t here, not the -y_t·G_t used for selection.
  let sum = 0
  let free = 0
  let upperBound = Infinity
  let lowerBound = -Infinity
  for (let t = 0; t < n; t++) {
    const yG = y[t] * G[t]
    if (alpha[t] >= C) {
      if (y[t] < 0) upperBound = Math.min(upperBound, yG)
      else lowerBound = Math.max(lowerBound, yG)
    } else if (alpha[t] <= 0) {
      if (y[t] > 0) upperBound = Math.min(upperBound, yG)
      else lowerBound = Math.max(lowerBound, yG)
    } else {
      free++
      sum += yG
    }
  }
  const rho = free > 0 ? sum / free : (upperBound + lowerBound) / 2

  const supportIndices: number[] = []
  const coefficients: number[] = []
  for (let t = 0; t < n; t++) {
    if (alpha[t] <= 0) continue
    supportIndices.push(indices[t])
    coefficients.push(y[t] * alpha[t])
  }
  return { supportIndices, coefficients, rho }
}

export interface SvmModel {
  X: Matrix
  kernelContext: KernelContext
  classes: number[]
  /** One binary model per class pair, in libsvm's (0,1), (0,2), … (1,2), … order. */
  pairs: { a: number; b: number; model: BinaryModel }[]
}

export function svmFit(X: Matrix, labels: Int32Array, nClasses: number, options: SvmOptions = {}): SvmModel {
  const C = options.C ?? 1
  const kernelContext: KernelContext = {
    X,
    kernel: options.kernel ?? 'rbf',
    gamma: resolveGamma(X, options.gamma ?? 'scale'),
    degree: options.degree ?? 3,
    coef0: options.coef0 ?? 0,
  }
  const tolerance = options.tolerance ?? 1e-3
  const maxIterations = options.maxIterations ?? 1000000

  const byClass: number[][] = Array.from({ length: nClasses }, () => [])
  for (let i = 0; i < X.n; i++) if (labels[i] >= 0 && labels[i] < nClasses) byClass[labels[i]].push(i)

  const pairs: SvmModel['pairs'] = []
  for (let a = 0; a < nClasses; a++) {
    for (let b = a + 1; b < nClasses; b++) {
      const indices = [...byClass[a], ...byClass[b]]
      if (indices.length === 0) continue
      const y = new Int8Array(indices.length)
      // libsvm labels the first class of the pair +1, the second -1.
      y.fill(1, 0, byClass[a].length)
      y.fill(-1, byClass[a].length)
      pairs.push({ a, b, model: solveBinary(kernelContext, indices, y, C, tolerance, maxIterations) })
    }
  }
  return { X, kernelContext, classes: [...Array(nClasses).keys()], pairs }
}

/** The signed distance for one class pair; positive votes for `a`. */
function decision(model: SvmModel, pair: SvmModel['pairs'][number], point: number[] | Float64Array): number {
  const k = model.kernelContext
  let sum = 0
  for (let s = 0; s < pair.model.supportIndices.length; s++) {
    const index = pair.model.supportIndices[s]
    let value: number
    switch (k.kernel) {
      case 'linear': {
        let d = 0
        for (let f = 0; f < k.X.d; f++) d += k.X.data[index * k.X.d + f] * point[f]
        value = d
        break
      }
      case 'poly': {
        let d = 0
        for (let f = 0; f < k.X.d; f++) d += k.X.data[index * k.X.d + f] * point[f]
        value = (k.gamma * d + k.coef0) ** k.degree
        break
      }
      case 'sigmoid': {
        let d = 0
        for (let f = 0; f < k.X.d; f++) d += k.X.data[index * k.X.d + f] * point[f]
        value = Math.tanh(k.gamma * d + k.coef0)
        break
      }
      default: {
        let d = 0
        for (let f = 0; f < k.X.d; f++) d += (k.X.data[index * k.X.d + f] - point[f]) ** 2
        value = Math.exp(-k.gamma * d)
      }
    }
    sum += pair.model.coefficients[s] * value
  }
  return sum - pair.model.rho
}

/**
 * One-against-one voting. Ties go to the class with the larger summed decision
 * value, then to the lower index — libsvm's own order.
 */
export function svmPredictOne(model: SvmModel, point: number[] | Float64Array): number {
  const votes = new Int32Array(model.classes.length)
  const totals = new Float64Array(model.classes.length)
  for (const pair of model.pairs) {
    const value = decision(model, pair, point)
    if (value > 0) { votes[pair.a]++; totals[pair.a] += value }
    else { votes[pair.b]++; totals[pair.b] -= value }
  }
  let best = 0
  for (let c = 1; c < votes.length; c++) {
    if (votes[c] > votes[best] || (votes[c] === votes[best] && totals[c] > totals[best])) best = c
  }
  return best
}

export function svmPredict(model: SvmModel, points: Matrix): Int32Array {
  const out = new Int32Array(points.n)
  const point = new Float64Array(points.d)
  for (let i = 0; i < points.n; i++) {
    for (let f = 0; f < points.d; f++) point[f] = points.data[i * points.d + f]
    out[i] = svmPredictOne(model, point)
  }
  return out
}

/** Total number of support vectors, which the node reports. */
export function supportVectorCount(model: SvmModel): number {
  const unique = new Set<number>()
  for (const pair of model.pairs) for (const index of pair.model.supportIndices) unique.add(index)
  return unique.size
}
