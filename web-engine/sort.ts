/**
 * SORT — Simple Online and Realtime Tracking (Bewley et al., 2016).
 *
 * The desktop node leans on filterpy for the Kalman filter and scipy for the
 * Hungarian assignment. Neither exists in the browser, and pulling a linear
 * algebra library in for two fixed-size problems is not worth it, so both are
 * written out here: a 7-state constant-velocity filter and the classic O(n³)
 * Hungarian algorithm with potentials.
 *
 * The filter matches filterpy's conventions exactly, including the Joseph-form
 * covariance update, so a track follows the same path it would on the desktop.
 */

type Matrix = number[][]

/* ------------------------------------------------------------ linear algebra */

function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(0))
}

function identity(n: number): Matrix {
  const m = zeros(n, n)
  for (let i = 0; i < n; i++) m[i][i] = 1
  return m
}

function multiply(a: Matrix, b: Matrix): Matrix {
  const rows = a.length
  const inner = b.length
  const cols = b[0].length
  const out = zeros(rows, cols)
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < inner; k++) {
      const aik = a[i][k]
      if (aik === 0) continue
      for (let j = 0; j < cols; j++) out[i][j] += aik * b[k][j]
    }
  }
  return out
}

function transpose(a: Matrix): Matrix {
  const out = zeros(a[0].length, a.length)
  for (let i = 0; i < a.length; i++) for (let j = 0; j < a[0].length; j++) out[j][i] = a[i][j]
  return out
}

function add(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((v, j) => v + b[i][j]))
}

function subtract(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((v, j) => v - b[i][j]))
}

/** Gauss-Jordan inverse; the only matrix inverted here is the 4×4 innovation covariance. */
function invert(a: Matrix): Matrix {
  const n = a.length
  const work = a.map((row, i) => [...row, ...identity(n)[i]])
  for (let col = 0; col < n; col++) {
    let pivot = col
    for (let r = col + 1; r < n; r++) if (Math.abs(work[r][col]) > Math.abs(work[pivot][col])) pivot = r
    if (Math.abs(work[pivot][col]) < 1e-12) return identity(n)
    ;[work[col], work[pivot]] = [work[pivot], work[col]]
    const scale = work[col][col]
    for (let j = 0; j < 2 * n; j++) work[col][j] /= scale
    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const factor = work[r][col]
      if (factor === 0) continue
      for (let j = 0; j < 2 * n; j++) work[r][j] -= factor * work[col][j]
    }
  }
  return work.map((row) => row.slice(n))
}

/* ---------------------------------------------------------------- Hungarian */

/**
 * Minimum-cost assignment, the equivalent of scipy's `linear_sum_assignment`.
 * O(n³) shortest-augmenting-path with potentials; rectangular inputs are
 * transposed so the loop always runs over the shorter side.
 */
export function linearSumAssignment(cost: Matrix): [number, number][] {
  const rows = cost.length
  if (rows === 0) return []
  const cols = cost[0].length
  if (cols === 0) return []

  if (rows > cols) {
    return linearSumAssignment(transpose(cost)).map(([r, c]) => [c, r] as [number, number])
  }

  const INF = Infinity
  const u = new Array<number>(rows + 1).fill(0)
  const v = new Array<number>(cols + 1).fill(0)
  const match = new Array<number>(cols + 1).fill(0)
  const way = new Array<number>(cols + 1).fill(0)

  for (let i = 1; i <= rows; i++) {
    match[0] = i
    let j0 = 0
    const minv = new Array<number>(cols + 1).fill(INF)
    const used = new Array<boolean>(cols + 1).fill(false)
    do {
      used[j0] = true
      const i0 = match[j0]
      let delta = INF
      let j1 = 0
      for (let j = 1; j <= cols; j++) {
        if (used[j]) continue
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
        if (cur < minv[j]) {
          minv[j] = cur
          way[j] = j0
        }
        if (minv[j] < delta) {
          delta = minv[j]
          j1 = j
        }
      }
      for (let j = 0; j <= cols; j++) {
        if (used[j]) {
          u[match[j]] += delta
          v[j] -= delta
        } else {
          minv[j] -= delta
        }
      }
      j0 = j1
    } while (match[j0] !== 0)
    do {
      const j1 = way[j0]
      match[j0] = match[j1]
      j0 = j1
    } while (j0)
  }

  const out: [number, number][] = []
  for (let j = 1; j <= cols; j++) if (match[j] > 0) out.push([match[j] - 1, j - 1])
  out.sort((a, b) => a[0] - b[0])
  return out
}

/* --------------------------------------------------------------------- IOU */

export type Box = [number, number, number, number]

/** Intersection over union for two [x1,y1,x2,y2] boxes. */
export function iou(a: Box, b: Box): number {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const inter = w * h
  const areaA = (a[2] - a[0]) * (a[3] - a[1])
  const areaB = (b[2] - b[0]) * (b[3] - b[1])
  return inter / (areaA + areaB - inter + 1e-9)
}

/** [x1,y1,x2,y2] → [cx, cy, scale, aspect], the filter's measurement space. */
function boxToMeasurement(box: Box): Matrix {
  const w = box[2] - box[0]
  const h = box[3] - box[1]
  return [[box[0] + w / 2], [box[1] + h / 2], [w * h], [h > 0 ? w / h : 1]]
}

/** [cx, cy, scale, aspect, …] → [x1,y1,x2,y2]. */
function measurementToBox(state: Matrix): Box {
  const cx = state[0][0]
  const cy = state[1][0]
  const s = state[2][0]
  const r = state[3][0]
  const w = Math.sqrt(Math.abs(s * r))
  const h = w > 0 ? s / w : 0
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
}

/* ------------------------------------------------------- Kalman box tracker */

class KalmanBoxTracker {
  readonly id: number
  label: string
  score: number
  timeSinceUpdate = 0
  hits = 0
  hitStreak = 0
  age = 0

  private x: Matrix
  private P: Matrix
  private readonly F: Matrix
  private readonly H: Matrix
  private readonly R: Matrix
  private readonly Q: Matrix

  constructor(id: number, box: Box, label = '', score = 0) {
    this.id = id
    this.label = label
    this.score = score

    // Constant-velocity model over [cx, cy, s, r, vx, vy, vs].
    this.F = [
      [1, 0, 0, 0, 1, 0, 0],
      [0, 1, 0, 0, 0, 1, 0],
      [0, 0, 1, 0, 0, 0, 1],
      [0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 1],
    ]
    this.H = [
      [1, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0],
    ]

    // filterpy starts every matrix at the identity; SORT then scales these
    // blocks. Scale and aspect are noisier measurements than position, and the
    // initial velocity is unknown, hence the large velocity covariance.
    this.R = identity(4)
    this.R[2][2] *= 10
    this.R[3][3] *= 10

    this.P = identity(7)
    for (let i = 4; i < 7; i++) this.P[i][i] *= 1000
    for (let i = 0; i < 7; i++) this.P[i][i] *= 10

    this.Q = identity(7)
    this.Q[6][6] *= 0.01
    for (let i = 4; i < 7; i++) this.Q[i][i] *= 0.01

    this.x = zeros(7, 1)
    const z = boxToMeasurement(box)
    for (let i = 0; i < 4; i++) this.x[i][0] = z[i][0]
  }

  update(box: Box, score = 0): void {
    this.timeSinceUpdate = 0
    this.hits += 1
    this.hitStreak += 1
    this.score = score

    const z = boxToMeasurement(box)
    const Ht = transpose(this.H)
    const PHt = multiply(this.P, Ht)
    const S = add(multiply(this.H, PHt), this.R)
    const K = multiply(PHt, invert(S))
    const y = subtract(z, multiply(this.H, this.x))
    this.x = add(this.x, multiply(K, y))

    // Joseph form, as filterpy uses: numerically stabler than (I − KH)P.
    const IKH = subtract(identity(7), multiply(K, this.H))
    this.P = add(multiply(multiply(IKH, this.P), transpose(IKH)), multiply(multiply(K, this.R), transpose(K)))
  }

  predict(): Box {
    // A negative predicted scale is unphysical; zero the scale velocity instead.
    if (this.x[6][0] + this.x[2][0] <= 0) this.x[6][0] = 0
    this.x = multiply(this.F, this.x)
    this.P = add(multiply(multiply(this.F, this.P), transpose(this.F)), this.Q)
    this.age += 1
    if (this.timeSinceUpdate > 0) this.hitStreak = 0
    this.timeSinceUpdate += 1
    return measurementToBox(this.x)
  }

  state(): Box {
    return measurementToBox(this.x)
  }
}

/* ------------------------------------------------------------------- SORT */

export interface RawTrack {
  track_id: number
  label: string
  score: number
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface Association {
  matches: [number, number][]
  unmatchedDetections: number[]
  unmatchedTrackers: number[]
}

/** IOU-based association between detections and predicted tracker boxes. */
export function associate(detections: Box[], trackers: Box[], iouThreshold: number): Association {
  if (trackers.length === 0) {
    return { matches: [], unmatchedDetections: detections.map((_, i) => i), unmatchedTrackers: [] }
  }
  if (detections.length === 0) {
    return { matches: [], unmatchedDetections: [], unmatchedTrackers: trackers.map((_, i) => i) }
  }

  const matrix = detections.map((d) => trackers.map((t) => iou(d, t)))

  // When every detection has at most one candidate above threshold and vice
  // versa, the assignment is forced and the Hungarian pass can be skipped.
  const above = matrix.map((row) => row.map((v) => (v > iouThreshold ? 1 : 0)))
  const rowMax = Math.max(...above.map((row) => row.reduce((a, b) => a + b, 0)))
  const colMax = Math.max(...above[0].map((_, j) => above.reduce((a, row) => a + row[j], 0)))

  let candidates: [number, number][]
  if (rowMax === 1 && colMax === 1) {
    candidates = []
    for (let i = 0; i < above.length; i++) {
      for (let j = 0; j < above[i].length; j++) if (above[i][j]) candidates.push([i, j])
    }
  } else {
    candidates = linearSumAssignment(matrix.map((row) => row.map((v) => -v)))
  }

  const matchedDetections = new Set(candidates.map(([d]) => d))
  const matchedTrackers = new Set(candidates.map(([, t]) => t))
  const unmatchedDetections = detections.map((_, i) => i).filter((i) => !matchedDetections.has(i))
  const unmatchedTrackers = trackers.map((_, i) => i).filter((i) => !matchedTrackers.has(i))

  // A pairing the assignment produced can still be too weak to accept.
  const matches: [number, number][] = []
  for (const [d, t] of candidates) {
    if (matrix[d][t] < iouThreshold) {
      unmatchedDetections.push(d)
      unmatchedTrackers.push(t)
    } else {
      matches.push([d, t])
    }
  }

  return { matches, unmatchedDetections, unmatchedTrackers }
}

export class SortTracker {
  private trackers: KalmanBoxTracker[] = []
  private frameCount = 0
  private nextId = 0

  constructor(
    private readonly maxAge = 5,
    private readonly minHits = 2,
    private readonly iouThreshold = 0.3
  ) {}

  update(detections: Box[], labels: string[], scores: number[]): RawTrack[] {
    this.frameCount += 1

    // Advance every tracker, dropping any whose state has gone non-finite.
    const predicted: Box[] = []
    const alive: KalmanBoxTracker[] = []
    for (const tracker of this.trackers) {
      const box = tracker.predict()
      if (box.some((v) => !Number.isFinite(v))) continue
      alive.push(tracker)
      predicted.push(box)
    }
    this.trackers = alive

    const { matches, unmatchedDetections } = associate(detections, predicted, this.iouThreshold)
    for (const [d, t] of matches) this.trackers[t].update(detections[d], scores[d] ?? 0)
    for (const d of unmatchedDetections) {
      this.trackers.push(new KalmanBoxTracker(this.nextId++, detections[d], labels[d] ?? '', scores[d] ?? 0))
    }

    const results: RawTrack[] = []
    for (let i = this.trackers.length - 1; i >= 0; i--) {
      const tracker = this.trackers[i]
      const box = tracker.state()
      // A track is reported once it has been confirmed by enough hits — except
      // in the opening frames, where nothing could have been confirmed yet.
      if (tracker.timeSinceUpdate <= 1 && (tracker.hitStreak >= this.minHits || this.frameCount <= this.minHits)) {
        results.push({
          track_id: tracker.id,
          label: tracker.label,
          score: tracker.score,
          x1: box[0],
          y1: box[1],
          x2: box[2],
          y2: box[3],
        })
      }
      if (tracker.timeSinceUpdate > this.maxAge) this.trackers.splice(i, 1)
    }
    return results
  }
}
