/**
 * scikit-learn's `train_test_split`, reproduced draw for draw.
 *
 * The desktop node calls sklearn, so the same seed gives the same rows there as
 * in a student's own `train_test_split(..., random_state=42)`. The web port used
 * to shuffle with its own generator, and the node and the Python node then
 * disagreed on which flowers were in the test set — a lesson comparing the two
 * showed different scores for the same parameters. Matching sklearn needs its
 * generator too: numpy's legacy `RandomState` (MT19937), whose `permutation`
 * and `choice(replace=False)` are what `ShuffleSplit` and
 * `StratifiedShuffleSplit` consume.
 *
 * Verified against numpy 2.5 / scikit-learn 1.9 (the output is index-for-index
 * identical, including the order of the rows).
 */

/** numpy.random.RandomState: MT19937 with numpy's legacy integer seeding. */
export class NumpyRandomState {
  private readonly mt = new Uint32Array(624)
  private index = 624

  constructor(seed: number) {
    // init_genrand, as numpy's legacy seeding does for a plain integer.
    const mt = this.mt
    mt[0] = seed >>> 0
    for (let i = 1; i < 624; i++) {
      const previous = mt[i - 1] ^ (mt[i - 1] >>> 30)
      // 1812433253 * previous + i, modulo 2^32, without losing precision.
      mt[i] = (((((previous & 0xffff0000) >>> 16) * 1812433253) << 16) + (previous & 0x0000ffff) * 1812433253 + i) >>> 0
    }
  }

  private twist(): void {
    const mt = this.mt
    for (let i = 0; i < 624; i++) {
      const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff)
      mt[i] = mt[(i + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0)
    }
    this.index = 0
  }

  uint32(): number {
    if (this.index >= 624) this.twist()
    let y = this.mt[this.index++]
    y ^= y >>> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= y >>> 18
    return y >>> 0
  }

  /** numpy's `random_interval`: uniform in [0, max] by masked rejection. */
  interval(max: number): number {
    if (max === 0) return 0
    let mask = max
    mask |= mask >>> 1
    mask |= mask >>> 2
    mask |= mask >>> 4
    mask |= mask >>> 8
    mask |= mask >>> 16
    mask >>>= 0
    let value: number
    do value = (this.uint32() & mask) >>> 0
    while (value > max)
    return value
  }

  /** `RandomState.shuffle` on a 1-D array, in place. */
  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i--) {
      const j = this.interval(i)
      const t = items[i]
      items[i] = items[j]
      items[j] = t
    }
    return items
  }

  /** `RandomState.permutation(n)`. */
  permutation(n: number): number[] {
    return this.shuffle([...Array(n).keys()])
  }
}

export interface SplitIndices {
  train: number[]
  test: number[]
}

/** sklearn's `_validate_shuffle_split` for a float test_size and no train_size. */
export function splitSizes(nSamples: number, testFraction: number): { nTrain: number; nTest: number } {
  const nTest = Math.ceil(testFraction * nSamples)
  return { nTrain: nSamples - nTest, nTest }
}

/** sklearn's `_approximate_mode`: how many draws each class gets, ties broken by `rng`. */
function approximateMode(classCounts: number[], nDraws: number, rng: NumpyRandomState): number[] {
  const total = classCounts.reduce((a, b) => a + b, 0)
  const continuous = classCounts.map((c) => (c / total) * nDraws)
  const floored = continuous.map(Math.floor)
  let needToAdd = Math.trunc(nDraws - floored.reduce((a, b) => a + b, 0))
  if (needToAdd > 0) {
    const remainder = continuous.map((c, i) => c - floored[i])
    const values = [...new Set(remainder)].sort((a, b) => b - a)
    for (const value of values) {
      const inds = remainder.flatMap((r, i) => (r === value ? [i] : []))
      const addNow = Math.min(inds.length, needToAdd)
      // choice(inds, size, replace=False) is permutation(len(inds))[:size].
      const chosen = rng.permutation(inds.length).slice(0, addNow).map((p) => inds[p])
      for (const i of chosen) floored[i] += 1
      needToAdd -= addNow
      if (needToAdd === 0) break
    }
  }
  return floored
}

/** Classes in numpy's `np.unique` order: numeric when every label is a number, else by code point. */
function sortedClasses(labels: unknown[]): string[] {
  const allNumeric = labels.every((v) => typeof v === 'number' && Number.isFinite(v))
  const unique = new Map<string, unknown>()
  for (const v of labels) unique.set(String(v), v)
  const keys = [...unique.keys()]
  return allNumeric
    ? keys.sort((a, b) => (unique.get(a) as number) - (unique.get(b) as number))
    : keys.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
}

/**
 * `train_test_split(range(n), test_size=testFraction, random_state=seed,
 * shuffle=shuffle, stratify=labels)` — the train and test row indices, in the
 * order sklearn returns them.
 */
export function sklearnTrainTestSplit(
  nSamples: number,
  testFraction: number,
  seed: number,
  shuffle = true,
  labels?: unknown[],
): SplitIndices {
  const { nTrain, nTest } = splitSizes(nSamples, testFraction)
  if (!shuffle) {
    return { train: [...Array(nTrain).keys()], test: Array.from({ length: nTest }, (_, i) => nTrain + i) }
  }
  const rng = new NumpyRandomState(seed)
  if (!labels) {
    const permutation = rng.permutation(nSamples)
    return { train: permutation.slice(nTest, nTest + nTrain), test: permutation.slice(0, nTest) }
  }

  // StratifiedShuffleSplit._iter_indices
  const classes = sortedClasses(labels)
  const classOf = new Map(classes.map((c, i) => [c, i]))
  const classIndices: number[][] = classes.map(() => [])
  labels.forEach((v, i) => classIndices[classOf.get(String(v))!].push(i))   // stable, like argsort(kind="stable")
  const classCounts = classIndices.map((ids) => ids.length)
  if (nTrain < classes.length || nTest < classes.length) {
    throw new Error(`Stratification impossible : il faut au moins ${classes.length} lignes dans le train et dans le test (une par classe).`)
  }
  if (classCounts.some((c) => c < 2)) {
    throw new Error('Stratification impossible : une classe n’a qu’une seule ligne.')
  }
  const nI = approximateMode(classCounts, nTrain, rng)
  const remaining = classCounts.map((c, i) => c - nI[i])
  const tI = approximateMode(remaining, nTest, rng)
  const train: number[] = []
  const test: number[] = []
  classes.forEach((_, i) => {
    const perm = rng.permutation(classCounts[i]).map((p) => classIndices[i][p])
    train.push(...perm.slice(0, nI[i]))
    test.push(...perm.slice(nI[i], nI[i] + tI[i]))
  })
  // rng.permutation(list) shuffles a copy of the list.
  return { train: rng.shuffle(train), test: rng.shuffle(test) }
}
