/**
 * CART decision trees and the random forest built on them.
 *
 * Follows scikit-learn's own splitter so a tree grown here matches one grown on
 * the desktop: candidate thresholds are the midpoints between consecutive
 * distinct feature values, the split chosen is the one with the largest
 * weighted impurity decrease, and feature importances are the Gini-importance
 * sums normalised to one.
 */
import type { Matrix } from './ml'
import { row } from './ml'

export type Criterion = 'gini' | 'entropy'

export interface TreeNode {
  /** -1 on a leaf. */
  feature: number
  threshold: number
  left: number
  right: number
  /** Class counts reaching this node. */
  counts: Float64Array
  impurity: number
  samples: number
  /** The majority class, precomputed for prediction. */
  prediction: number
  depth: number
}

export interface DecisionTree {
  nodes: TreeNode[]
  nClasses: number
  nFeatures: number
  depth: number
  leaves: number
  importances: Float64Array
}

/**
 * sklearn treats two feature values as equal when they differ by less than
 * this, which keeps float noise from producing a split with an empty side.
 */
const FEATURE_THRESHOLD = 1e-7

function impurityOf(counts: Float64Array, total: number, criterion: Criterion): number {
  if (total <= 0) return 0
  if (criterion === 'gini') {
    let sum = 0
    for (let c = 0; c < counts.length; c++) {
      const p = counts[c] / total
      sum += p * p
    }
    return 1 - sum
  }
  // 'entropy' and 'log_loss' are the same Shannon entropy in sklearn, log base 2.
  let sum = 0
  for (let c = 0; c < counts.length; c++) {
    const p = counts[c] / total
    if (p > 0) sum -= p * Math.log2(p)
  }
  return sum
}

export interface TreeOptions {
  criterion?: Criterion
  maxDepth?: number
  minSamplesSplit?: number
  minSamplesLeaf?: number
  /** Features considered per split; 0 or absent means all of them. */
  maxFeatures?: number
  random?: () => number
}

export function fitTree(X: Matrix, y: Int32Array, nClasses: number, indices: number[], options: TreeOptions = {}): DecisionTree {
  const criterion = options.criterion ?? 'gini'
  const maxDepth = options.maxDepth && options.maxDepth > 0 ? options.maxDepth : Infinity
  const minSamplesSplit = Math.max(2, options.minSamplesSplit ?? 2)
  const minSamplesLeaf = Math.max(1, options.minSamplesLeaf ?? 1)
  const maxFeatures = options.maxFeatures && options.maxFeatures > 0 ? Math.min(options.maxFeatures, X.d) : X.d
  const random = options.random
  const nodes: TreeNode[] = []
  const importances = new Float64Array(X.d)
  let deepest = 0
  let leaves = 0

  const countsOf = (rows: number[]): Float64Array => {
    const counts = new Float64Array(nClasses)
    for (const i of rows) counts[y[i]]++
    return counts
  }

  const build = (rows: number[], depth: number): number => {
    const counts = countsOf(rows)
    const impurity = impurityOf(counts, rows.length, criterion)
    let prediction = 0
    for (let c = 1; c < nClasses; c++) if (counts[c] > counts[prediction]) prediction = c
    const index = nodes.length
    nodes.push({ feature: -1, threshold: 0, left: -1, right: -1, counts, impurity, samples: rows.length, prediction, depth })
    deepest = Math.max(deepest, depth)

    if (depth >= maxDepth || rows.length < minSamplesSplit || impurity <= 0) {
      leaves++
      return index
    }

    // Which features to try. sklearn draws a subset without replacement when
    // max_features is set; with all of them the order does not matter.
    let candidates = [...Array(X.d).keys()]
    if (maxFeatures < X.d && random) {
      for (let i = candidates.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1))
        const t = candidates[i]
        candidates[i] = candidates[j]
        candidates[j] = t
      }
      candidates = candidates.slice(0, maxFeatures)
    }

    let bestFeature = -1
    let bestThreshold = 0
    let bestDecrease = 0
    let bestLeft: number[] = []
    let bestRight: number[] = []

    for (const feature of candidates) {
      const ordered = [...rows].sort((a, b) => X.data[a * X.d + feature] - X.data[b * X.d + feature])
      const leftCounts = new Float64Array(nClasses)
      const rightCounts = Float64Array.from(counts)
      for (let position = 0; position < ordered.length - 1; position++) {
        const sample = ordered[position]
        leftCounts[y[sample]]++
        rightCounts[y[sample]]--
        const here = X.data[sample * X.d + feature]
        const next = X.data[ordered[position + 1] * X.d + feature]
        if (next - here <= FEATURE_THRESHOLD) continue
        const leftN = position + 1
        const rightN = ordered.length - leftN
        if (leftN < minSamplesLeaf || rightN < minSamplesLeaf) continue

        const leftImpurity = impurityOf(leftCounts, leftN, criterion)
        const rightImpurity = impurityOf(rightCounts, rightN, criterion)
        const decrease = impurity - (leftN / ordered.length) * leftImpurity - (rightN / ordered.length) * rightImpurity
        if (decrease > bestDecrease + 1e-12) {
          bestDecrease = decrease
          bestFeature = feature
          bestThreshold = (here + next) / 2
          bestLeft = ordered.slice(0, leftN)
          bestRight = ordered.slice(leftN)
        }
      }
    }

    if (bestFeature < 0) {
      leaves++
      return index
    }

    // Gini importance: the impurity decrease weighted by how many samples pass
    // through the node, summed over the tree and normalised at the end.
    importances[bestFeature] += rows.length * bestDecrease
    nodes[index].feature = bestFeature
    nodes[index].threshold = bestThreshold
    nodes[index].left = build(bestLeft, depth + 1)
    nodes[index].right = build(bestRight, depth + 1)
    return index
  }

  build(indices, 0)

  let total = 0
  for (let j = 0; j < X.d; j++) total += importances[j]
  if (total > 0) for (let j = 0; j < X.d; j++) importances[j] /= total

  return { nodes, nClasses, nFeatures: X.d, depth: deepest, leaves, importances }
}

export function treePredictOne(tree: DecisionTree, point: Float64Array | number[]): number {
  let index = 0
  while (tree.nodes[index].feature >= 0) {
    const node = tree.nodes[index]
    index = point[node.feature] <= node.threshold ? node.left : node.right
  }
  return tree.nodes[index].prediction
}

/** Class probabilities from the leaf's own class counts, as sklearn's predict_proba does. */
export function treeProbaOne(tree: DecisionTree, point: Float64Array | number[]): Float64Array {
  let index = 0
  while (tree.nodes[index].feature >= 0) {
    const node = tree.nodes[index]
    index = point[node.feature] <= node.threshold ? node.left : node.right
  }
  const leaf = tree.nodes[index]
  const out = new Float64Array(tree.nClasses)
  for (let c = 0; c < tree.nClasses; c++) out[c] = leaf.samples > 0 ? leaf.counts[c] / leaf.samples : 0
  return out
}

export function treePredict(tree: DecisionTree, points: Matrix): Int32Array {
  const out = new Int32Array(points.n)
  for (let i = 0; i < points.n; i++) out[i] = treePredictOne(tree, row(points, i))
  return out
}

/* ------------------------------------------------------------ random forest */

export interface Forest {
  trees: DecisionTree[]
  /** Rows each tree was trained on, needed for the out-of-bag score. */
  bags: number[][]
  nClasses: number
  importances: Float64Array
}

export interface ForestOptions extends TreeOptions {
  nEstimators?: number
  bootstrap?: boolean
  seed?: number
  /** 'sqrt' | 'log2' | null, the three choices the desktop node offers. */
  maxFeaturesMode?: 'sqrt' | 'log2' | null
}

function mulberry(seed: number): () => number {
  let state = (seed >>> 0) || 1
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function fitForest(X: Matrix, y: Int32Array, nClasses: number, options: ForestOptions = {}): Forest {
  const nEstimators = Math.max(1, options.nEstimators ?? 100)
  const bootstrap = options.bootstrap !== false
  const random = mulberry(options.seed ?? 42)
  const mode = options.maxFeaturesMode === undefined ? 'sqrt' : options.maxFeaturesMode
  const maxFeatures = mode === 'sqrt'
    ? Math.max(1, Math.floor(Math.sqrt(X.d)))
    : mode === 'log2'
      ? Math.max(1, Math.floor(Math.log2(X.d)))
      : X.d

  const trees: DecisionTree[] = []
  const bags: number[][] = []
  const importances = new Float64Array(X.d)
  for (let t = 0; t < nEstimators; t++) {
    const bag: number[] = []
    if (bootstrap) {
      for (let i = 0; i < X.n; i++) bag.push(Math.floor(random() * X.n))
    } else {
      for (let i = 0; i < X.n; i++) bag.push(i)
    }
    const tree = fitTree(X, y, nClasses, bag, { ...options, maxFeatures, random })
    trees.push(tree)
    bags.push(bag)
    for (let j = 0; j < X.d; j++) importances[j] += tree.importances[j]
  }
  for (let j = 0; j < X.d; j++) importances[j] /= nEstimators

  return { trees, bags, nClasses, importances }
}

/** Soft voting over the trees' probabilities, which is what sklearn's forest does. */
export function forestPredictOne(forest: Forest, point: Float64Array | number[]): number {
  const votes = new Float64Array(forest.nClasses)
  for (const tree of forest.trees) {
    const proba = treeProbaOne(tree, point)
    for (let c = 0; c < forest.nClasses; c++) votes[c] += proba[c]
  }
  let best = 0
  for (let c = 1; c < forest.nClasses; c++) if (votes[c] > votes[best]) best = c
  return best
}

export function forestPredict(forest: Forest, points: Matrix): Int32Array {
  const out = new Int32Array(points.n)
  for (let i = 0; i < points.n; i++) out[i] = forestPredictOne(forest, row(points, i))
  return out
}

/**
 * Out-of-bag accuracy: each sample is classified by the trees whose bootstrap
 * sample excluded it. Samples that every tree happened to see are skipped, as
 * sklearn does (it warns in that case).
 */
export function oobScore(forest: Forest, X: Matrix, y: Int32Array): number {
  const votes: Float64Array[] = Array.from({ length: X.n }, () => new Float64Array(forest.nClasses))
  const seen = new Uint8Array(X.n)
  forest.trees.forEach((tree, t) => {
    const inBag = new Uint8Array(X.n)
    for (const i of forest.bags[t]) inBag[i] = 1
    for (let i = 0; i < X.n; i++) {
      if (inBag[i]) continue
      seen[i] = 1
      const proba = treeProbaOne(tree, row(X, i))
      for (let c = 0; c < forest.nClasses; c++) votes[i][c] += proba[c]
    }
  })
  let correct = 0
  let counted = 0
  for (let i = 0; i < X.n; i++) {
    if (!seen[i]) continue
    counted++
    let best = 0
    for (let c = 1; c < forest.nClasses; c++) if (votes[i][c] > votes[i][best]) best = c
    if (best === y[i]) correct++
  }
  return counted > 0 ? correct / counted : 0
}
