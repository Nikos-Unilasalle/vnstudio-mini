/**
 * Loss Explorer, Training Monitor and Monte Carlo Propagation.
 *
 * The first two are pedagogical plots; the desktop draws the Loss Explorer with
 * OpenCV already, so its canvas geometry is reproduced exactly, and the
 * Training Monitor's matplotlib figure is redrawn with the same helpers the
 * other ported charts use.
 */
import type { NodeImpl, RunContext } from '../types'
import { drawPolyline } from '../cvUtils'
import { previewSize, pyRound } from '../dataframe'
import { applyColormap, jetColor } from '../colormaps'
import { classColour, drawAxes, PLOT_BG, PLOT_INK, project } from './mlData'

/* ------------------------------------------------------------ loss explorer */

const LOSSES = ['Cross-Entropy', 'Dice', 'Focal', 'Smooth L1 (Huber)', 'GIoU', 'InfoNCE']

// The desktop's canvas, to the pixel: a white 560x360 with these margins.
const CURVE_W = 560
const CURVE_H = 360
const PAD_L = 60
const PAD_R = 20
const PAD_T = 30
const PAD_B = 40

interface Limits { xMin: number; xMax: number; yMin: number; yMax: number }

function toPixels(x: number, y: number, limits: Limits): { x: number; y: number } {
  return {
    x: PAD_L + ((x - limits.xMin) / (limits.xMax - limits.xMin + 1e-9)) * (CURVE_W - PAD_L - PAD_R),
    y: CURVE_H - PAD_B - ((y - limits.yMin) / (limits.yMax - limits.yMin + 1e-9)) * (CURVE_H - PAD_T - PAD_B),
  }
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

/** `np.linspace`, inclusive of both ends. */
function linspace(from: number, to: number, count: number): number[] {
  if (count <= 1) return [from]
  const out = new Array(count)
  for (let i = 0; i < count; i++) out[i] = from + ((to - from) * i) / (count - 1)
  return out
}

interface LossResult {
  /** Each curve: samples, colour (BGR) and legend text. */
  curves: { xs: number[]; ys: number[]; colour: [number, number, number]; label: string }[]
  limits: Limits
  xLabel: string
  markX: number
  markY: number
  loss: number
  grad: number
  info: Record<string, unknown>
}

function crossEntropy(point: number): LossResult {
  const z = linspace(-6, 6, 400)
  const yhat = z.map(sigmoid)
  return {
    curves: [
      { xs: z, ys: yhat.map((v) => -Math.log(Math.min(1, Math.max(1e-6, v)))), colour: [0, 90, 200], label: 'loss  -log(y_hat)' },
      { xs: z, ys: yhat.map((v) => v - 1), colour: [200, 90, 0], label: 'grad  y_hat - y' },
    ],
    limits: { xMin: -6, xMax: 6, yMin: -1.2, yMax: 6 },
    xLabel: 'logit z  (target = 1)',
    markX: Math.log(point / (1 - point)),
    markY: -Math.log(point),
    loss: -Math.log(point),
    grad: point - 1,
    info: { y_hat: pyRound(point, 3) },
  }
}

function dice(point: number): LossResult {
  const p = linspace(0.001, 1, 400)
  return {
    curves: [
      { xs: p, ys: p.map((v) => 1 - (2 * v) / (v + 1)), colour: [0, 90, 200], label: 'loss  1 - 2p/(p+1)' },
      { xs: p, ys: p.map((v) => -2 / (v + 1) ** 2), colour: [200, 90, 0], label: 'grad  dL/dp' },
    ],
    limits: { xMin: 0, xMax: 1, yMin: -2.2, yMax: 1.1 },
    xLabel: 'predicted prob p  (object pixel, g=1)',
    markX: point,
    markY: 1 - (2 * point) / (point + 1),
    loss: 1 - (2 * point) / (point + 1),
    grad: -2 / (point + 1) ** 2,
    info: { p: pyRound(point, 3) },
  }
}

function focal(point: number, gamma: number, alpha: number): LossResult {
  const p = linspace(0.001, 1, 400)
  const loss = alpha * (1 - point) ** gamma * -Math.log(point)
  return {
    curves: [
      { xs: p, ys: p.map((v) => -Math.log(v)), colour: [170, 170, 170], label: 'cross-entropy' },
      { xs: p, ys: p.map((v) => alpha * (1 - v) ** gamma * -Math.log(v)), colour: [0, 90, 200], label: `focal (g=${gamma}, a=${alpha})` },
    ],
    limits: { xMin: 0, xMax: 1, yMin: 0, yMax: 3 },
    xLabel: 'pt = prob of true class',
    markX: point,
    markY: loss,
    loss,
    // dL/dpt of alpha·(1-pt)^gamma·(-log pt).
    grad: alpha * ((1 - point) ** gamma * (-1 / point) + gamma * (1 - point) ** (gamma - 1) * Math.log(point)),
    info: {
      pt: pyRound(point, 3),
      ce: pyRound(-Math.log(point), 4),
      attenuation: pyRound((1 - point) ** gamma, 4),
    },
  }
}

function huber(point: number, beta: number): LossResult {
  const x = linspace(-3 * beta, 3 * beta, 400)
  const smooth = x.map((v) => (Math.abs(v) < beta ? (0.5 * v * v) / beta : Math.abs(v) - 0.5 * beta))
  const residual = (point * 2 - 1) * 3 * beta
  const inside = Math.abs(residual) < beta
  return {
    curves: [
      { xs: x, ys: x.map((v) => 0.5 * v * v), colour: [170, 170, 170], label: 'L2 = 0.5 x^2' },
      { xs: x, ys: smooth, colour: [0, 90, 200], label: `smooth L1 (beta=${beta})` },
    ],
    limits: { xMin: -3 * beta, xMax: 3 * beta, yMin: 0, yMax: Math.max(...smooth, 1e-3) * 1.1 },
    xLabel: 'residual x',
    markX: residual,
    markY: inside ? (0.5 * residual * residual) / beta : Math.abs(residual) - 0.5 * beta,
    loss: inside ? (0.5 * residual * residual) / beta : Math.abs(residual) - 0.5 * beta,
    grad: inside ? residual / beta : Math.sign(residual),
    info: { residual: pyRound(residual, 3), beta },
  }
}

function giou(point: number): LossResult {
  const side = 2
  const t = linspace(0, 4, 400)
  const iouAt = (offset: number) => {
    const intersection = Math.max(0, Math.min(side, offset + side) - Math.max(0, offset))
    const union = 2 * side - intersection
    return { iou: intersection / union, union }
  }
  const lossIou = t.map((offset) => 1 - iouAt(offset).iou)
  const lossGiou = t.map((offset) => {
    const { iou, union } = iouAt(offset)
    const cover = offset + side
    return 1 - (iou - (cover - union) / cover)
  })
  const offset = point * 4
  const { iou, union } = iouAt(offset)
  const cover = offset + side
  const generalised = iou - (cover - union) / cover
  return {
    curves: [
      { xs: t, ys: lossIou, colour: [170, 170, 170], label: 'L_IoU (flat once disjoint)' },
      { xs: t, ys: lossGiou, colour: [0, 90, 200], label: 'L_GIoU (keeps sloping)' },
    ],
    limits: { xMin: 0, xMax: 4, yMin: 0, yMax: 2.1 },
    xLabel: 'centre offset between boxes',
    markX: offset,
    markY: 1 - generalised,
    loss: 1 - generalised,
    grad: 1 - iou,
    info: {
      offset: pyRound(offset, 3),
      iou: pyRound(iou, 4),
      giou: pyRound(generalised, 4),
    },
  }
}

function infoNce(tau: number): LossResult {
  const sims = [0.9, 0.3, 0.2]
  const positive = (t: number) => {
    const exps = sims.map((s) => Math.exp(s / t))
    return exps[0] / exps.reduce((a, b) => a + b, 0)
  }
  const taus = linspace(0.05, 2, 400)
  const probabilities = taus.map(positive)
  const p = positive(tau)
  return {
    curves: [
      { xs: taus, ys: probabilities, colour: [200, 90, 0], label: 'p(positive)' },
      { xs: taus, ys: probabilities.map((v) => -Math.log(v)), colour: [0, 90, 200], label: 'loss -log p(pos)' },
    ],
    limits: { xMin: 0.05, xMax: 2, yMin: 0, yMax: 1.3 },
    xLabel: 'temperature tau',
    markX: tau,
    markY: p,
    loss: -Math.log(p),
    grad: p - 1,
    info: { tau: pyRound(tau, 3), p_positive: pyRound(p, 4) },
  }
}

export const mlLossExplorer: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const index = Math.round(Number(params.loss ?? 0))
  const name = LOSSES[index] ?? LOSSES[0]
  const point = Number(params.point ?? 0.5)
  const gamma = Number(params.gamma ?? 2)
  const alpha = Number(params.alpha ?? 0.25)
  const beta = Number(params.beta ?? 1)
  const tau = Number(params.tau ?? 0.5)

  const result = name === 'Cross-Entropy' ? crossEntropy(point)
    : name === 'Dice' ? dice(point)
      : name === 'Focal' ? focal(point, gamma, alpha)
        : name === 'Smooth L1 (Huber)' ? huber(point, beta)
          : name === 'GIoU' ? giou(point)
            : infoNce(tau)

  // This node draws on white, unlike the dark analytics charts.
  const img = ctx.track(new cv.Mat(CURVE_H, CURVE_W, cv.CV_8UC3, new cv.Scalar(255, 255, 255, 255)))
  const font = cv.FONT_HERSHEY_SIMPLEX
  cv.rectangle(img, new cv.Point(PAD_L, PAD_T), new cv.Point(CURVE_W - PAD_R, CURVE_H - PAD_B),
    new cv.Scalar(180, 180, 180, 255), 1)
  cv.putText(img, result.xLabel, new cv.Point(CURVE_W / 2 - 30, CURVE_H - 12), font, 0.5,
    new cv.Scalar(90, 90, 90, 255), 1, cv.LINE_AA)
  if (result.limits.yMin < 0 && result.limits.yMax > 0) {
    const left = toPixels(result.limits.xMin, 0, result.limits)
    const right = toPixels(result.limits.xMax, 0, result.limits)
    cv.line(img, new cv.Point(Math.round(left.x), Math.round(left.y)), new cv.Point(Math.round(right.x), Math.round(right.y)),
      new cv.Scalar(210, 210, 210, 255), 1)
  }

  result.curves.forEach((curve, i) => {
    const colour = new cv.Scalar(curve.colour[0], curve.colour[1], curve.colour[2], 255)
    drawPolyline(cv, img, curve.xs.map((x, k) => toPixels(x, curve.ys[k], result.limits)), false, colour, 2)
    cv.putText(img, curve.label, new cv.Point(PAD_L + 8, PAD_T + 18 + i * 20), font, 0.5, colour, 1, cv.LINE_AA)
  })

  const mark = toPixels(result.markX, result.markY, result.limits)
  cv.circle(img, new cv.Point(Math.round(mark.x), Math.round(mark.y)), 5, new cv.Scalar(0, 0, 0, 255), -1, cv.LINE_AA)

  const round4 = (v: number) => pyRound(v, 4)
  cv.putText(img, `${name}   L=${round4(result.loss)}   grad=${round4(result.grad)}`,
    new cv.Point(PAD_L, 20), font, 0.55, new cv.Scalar(0, 0, 0, 255), 1, cv.LINE_AA)

  return {
    main: img,
    loss: pyRound(result.loss, 5),
    grad: pyRound(result.grad, 5),
    data: { loss_name: name, ...result.info },
  }
}

/* --------------------------------------------------------- training monitor */

/** `np.convolve(arr, ones(w)/w, 'valid')` — the moving average the desktop uses. */
function movingAverage(values: number[], window: number): number[] {
  if (window <= 1 || values.length < window) return values
  const out: number[] = []
  let sum = 0
  for (let i = 0; i < values.length; i++) {
    sum += values[i]
    if (i >= window) sum -= values[i - window]
    if (i >= window - 1) out.push(sum / window)
  }
  return out
}

export const mlTrainingMonitor: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const history = inputs.loss_history as Record<string, unknown> | undefined
  const [w, h] = previewSize(inputs.img_size, { width: 700, height: 350, ...params })

  const numbers = (value: unknown): number[] =>
    Array.isArray(value) ? value.map(Number).filter((v) => Number.isFinite(v)) : []

  if (!history || typeof history !== 'object' || Array.isArray(history)) {
    const blank = ctx.track(new cv.Mat(200, 420, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))
    cv.putText(blank, 'Waiting for loss_history...', new cv.Point(20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5,
      new cv.Scalar(150, 150, 150, 255), 1, cv.LINE_AA)
    return { main: blank, preview: blank, best_epoch: 0, final_train_loss: 0, final_val_loss: 0 }
  }

  const trainLoss = numbers(history.train_loss)
  const valLoss = numbers(history.val_loss)
  const smoothing = Math.max(1, Math.round(Number(params.smooth ?? 1)))
  const logScale = Boolean(params.log_scale)

  const finalTrain = trainLoss.length > 0 ? trainLoss[trainLoss.length - 1] : 0
  const finalVal = valLoss.length > 0 ? valLoss[valLoss.length - 1] : 0
  // The best epoch follows validation when there is any, else training.
  const argMin = (values: number[]) => values.reduce((best, v, i) => (v < values[best] ? i : best), 0)
  const bestEpoch = valLoss.length > 0 ? argMin(valLoss) : trainLoss.length > 0 ? argMin(trainLoss) : 0

  const trainSmooth = movingAverage(trainLoss, smoothing)
  const valSmooth = movingAverage(valLoss, smoothing)
  const epochs = Math.max(trainLoss.length, valLoss.length)
  const all = [...trainSmooth, ...valSmooth]
  const transform = (v: number) => (logScale ? Math.log10(Math.max(v, 1e-12)) : v)
  const lows = all.length > 0 ? Math.min(...all.map(transform)) : 0
  const highs = all.length > 0 ? Math.max(...all.map(transform)) : 1

  const img = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(PLOT_BG[0], PLOT_BG[1], PLOT_BG[2], 255)))
  const title = valLoss.length > 0
    ? `Training Monitor | Epoch ${bestEpoch} | Train: ${finalTrain.toFixed(4)} | Val: ${finalVal.toFixed(4)}`
    : trainLoss.length > 0
      ? `Training Monitor | Epoch ${bestEpoch} | Train: ${finalTrain.toFixed(4)}`
      : 'Training Monitor | No data'
  const axes = drawAxes(cv, img, 0, Math.max(1, epochs - 1), lows, highs === lows ? lows + 1 : highs, true, title)

  // Smoothing shortens the series, so it is spread back across the full axis,
  // which is what the desktop's linspace does.
  const draw = (values: number[], length: number, colour: any) => {
    if (values.length === 0) return
    const points = values.map((v, i) => {
      const x = values.length > 1 ? ((length - 1) * i) / (values.length - 1) : 0
      const [px, py] = project(axes, x, transform(v))
      return { x: px, y: py }
    })
    drawPolyline(cv, img, points, false, colour, 2)
  }
  draw(trainSmooth, trainLoss.length, new cv.Scalar(255, 158, 74, 255))   // BGR of #4a9eff
  draw(valSmooth, valLoss.length, new cv.Scalar(74, 159, 255, 255))       // BGR of #ff9f4a

  if (params.show_best !== false && epochs > 0) {
    const [x] = project(axes, bestEpoch, lows)
    cv.line(img, new cv.Point(x, axes.top), new cv.Point(x, axes.top + axes.height), new cv.Scalar(136, 204, 68, 255), 1)
  }

  const font = cv.FONT_HERSHEY_SIMPLEX
  const legend = [
    ['Train loss', new cv.Scalar(255, 158, 74, 255)],
    ['Val loss', new cv.Scalar(74, 159, 255, 255)],
    [`Best epoch ${bestEpoch}`, new cv.Scalar(136, 204, 68, 255)],
  ] as [string, any][]
  legend.forEach(([label, colour], i) => {
    const y = axes.top + 14 + i * 14
    cv.line(img, new cv.Point(axes.left + axes.width - 110, y - 4), new cv.Point(axes.left + axes.width - 98, y - 4), colour, 2)
    cv.putText(img, label, new cv.Point(axes.left + axes.width - 94, y), font, 0.32,
      new cv.Scalar(PLOT_INK[0], PLOT_INK[1], PLOT_INK[2], 255), 1, cv.LINE_AA)
  })
  cv.putText(img, logScale ? 'log10(Loss)' : 'Loss', new cv.Point(4, axes.top - 4), font, 0.34,
    new cv.Scalar(PLOT_INK[0], PLOT_INK[1], PLOT_INK[2], 255), 1, cv.LINE_AA)
  cv.putText(img, 'Epoch', new cv.Point(axes.left + axes.width / 2 - 16, h - 4), font, 0.34,
    new cv.Scalar(PLOT_INK[0], PLOT_INK[1], PLOT_INK[2], 255), 1, cv.LINE_AA)

  return {
    main: img,
    preview: img,
    best_epoch: bestEpoch,
    final_train_loss: finalTrain,
    final_val_loss: finalVal,
  }
}

/* -------------------------------------------------- Monte Carlo propagation */

/** Mulberry32, so a fixed seed reproduces a run exactly. */
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

export const utilMonteCarloPropagation: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const seedInput = (inputs.seed ?? inputs.mask ?? inputs.main) as any
  if (!seedInput || typeof seedInput.cols !== 'number') {
    return { probability: null, preview: null, stats: null }
  }

  const width = seedInput.cols
  const height = seedInput.rows
  const gray = seedInput.channels() > 1 ? new cv.Mat() : seedInput
  if (seedInput.channels() > 1) cv.cvtColor(seedInput, gray, cv.COLOR_BGR2GRAY)
  const eight = gray.depth() === cv.CV_8U ? gray : new cv.Mat()
  if (gray.depth() !== cv.CV_8U) gray.convertTo(eight, cv.CV_8U, 255)
  const seedBytes = eight.data as Uint8Array

  const seedMask = new Uint8Array(width * height)
  let seedCount = 0
  for (let p = 0; p < seedMask.length; p++) {
    seedMask[p] = seedBytes[p] > 0 ? 1 : 0
    if (seedMask[p]) seedCount++
  }
  if (eight !== gray) eight.delete()
  if (gray !== seedInput) gray.delete()

  const simulations = Math.max(10, Math.round(Number(params.n_simulations ?? 100)))
  const steps = Math.max(1, Math.round(Number(params.n_steps ?? 10)))
  const resistance = Math.min(1, Math.max(0, Number(params.resistance ?? 0.5)))
  // The desktop's enum is stored as a string on this node, not an index.
  const fourConnected = String(params.neighborhood ?? '8-connected') === '4-connected' || Number(params.neighborhood) === 1
  const seedParam = Math.round(Number(params.seed_value ?? params.random_seed ?? params.seed ?? -1))

  // The attractiveness map scales the local infection probability.
  const attractiveness = new Float32Array(width * height).fill(1)
  const attractIn = inputs.attractiveness as any
  if (attractIn && typeof attractIn.cols === 'number') {
    const resized = new cv.Mat()
    if (attractIn.cols !== width || attractIn.rows !== height) {
      cv.resize(attractIn, resized, new cv.Size(width, height), 0, 0, cv.INTER_LINEAR)
    } else {
      attractIn.copyTo(resized)
    }
    const single = resized.channels() > 1 ? new cv.Mat() : resized
    if (resized.channels() > 1) cv.cvtColor(resized, single, cv.COLOR_BGR2GRAY)
    const bytes8 = single.depth() === cv.CV_8U ? single : new cv.Mat()
    if (single.depth() !== cv.CV_8U) single.convertTo(bytes8, cv.CV_8U, 255)
    const bytes = bytes8.data as Uint8Array
    for (let p = 0; p < attractiveness.length; p++) attractiveness[p] = bytes[p] / 255
    if (bytes8 !== single) bytes8.delete()
    if (single !== resized) single.delete()
    resized.delete()
  }

  const offsets: [number, number][] = fourConnected
    ? [[-1, 0], [1, 0], [0, -1], [0, 1]]
    : [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

  const random = seedParam < 0 ? Math.random : rng(seedParam)
  const accumulated = new Float64Array(width * height)
  const state = new Uint8Array(width * height)

  for (let sim = 0; sim < simulations; sim++) {
    state.set(seedMask)
    for (let step = 0; step < steps; step++) {
      // A pixel is on the border when it is off but touches something on; the
      // dilate-and-subtract the desktop does comes to the same thing.
      const infections: number[] = []
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const at = y * width + x
          if (state[at]) continue
          let touching = false
          for (const [dy, dx] of offsets) {
            const ny = y + dy
            const nx = x + dx
            if (ny < 0 || nx < 0 || ny >= height || nx >= width) continue
            if (state[ny * width + nx]) { touching = true; break }
          }
          if (!touching) continue
          if (random() < attractiveness[at] * (1 - resistance)) infections.push(at)
        }
      }
      if (infections.length === 0) break
      for (const at of infections) state[at] = 1
    }
    for (let p = 0; p < state.length; p++) accumulated[p] += state[p]
  }

  const probability = ctx.track(new cv.Mat(height, width, cv.CV_8U))
  const probabilityBytes = probability.data as Uint8Array
  let high = 0
  let medium = 0
  let low = 0
  for (let p = 0; p < accumulated.length; p++) {
    const fraction = accumulated[p] / simulations
    probabilityBytes[p] = Math.trunc(fraction * 255)
    const percent = fraction * 100
    if (percent > 50) high++
    else if (percent >= 15) medium++
    else if (percent >= 2) low++
  }

  const preview = ctx.track(applyColormap(cv, probability, jetColor))
  const previewBytes = preview.data as Uint8Array
  for (let p = 0; p < probabilityBytes.length; p++) {
    // Zero-probability pixels go black; the original seeds are marked cyan.
    if (probabilityBytes[p] === 0) {
      previewBytes[p * 3] = 0
      previewBytes[p * 3 + 1] = 0
      previewBytes[p * 3 + 2] = 0
    }
    if (seedMask[p]) {
      previewBytes[p * 3] = 255
      previewBytes[p * 3 + 1] = 255
      previewBytes[p * 3 + 2] = 0
    }
  }

  const total = width * height
  const round2 = (v: number) => pyRound(v, 2)
  return {
    probability,
    main: preview,
    preview,
    stats: {
      seed_surface_pct: round2((seedCount / total) * 100),
      risk_high_pct: round2((high / total) * 100),
      risk_medium_pct: round2((medium / total) * 100),
      risk_low_pct: round2((low / total) * 100),
      simulations_run: simulations,
    },
  }
}
