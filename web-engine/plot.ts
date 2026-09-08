/**
 * A small chart builder, standing in for the matplotlib the desktop scripts use.
 *
 * The Python nodes draw their figures with `plt` and hand back an RGB array.
 * There is no matplotlib here, so scripts get this instead: axes with ticks and
 * a grid, lines, markers, horizontal and vertical rules, scatter, bars and a
 * legend — enough for the convergence, sweep, calibration and correlation
 * figures, drawn with OpenCV onto a Mat.
 */

export interface FigureOptions {
  width?: number
  height?: number
  title?: string
  xlabel?: string
  ylabel?: string
  xlim?: [number, number]
  ylim?: [number, number]
  logY?: boolean
  /** Light background, as matplotlib's default. */
  dark?: boolean
}

export interface SeriesOptions {
  /** BGR triple. */
  color?: [number, number, number]
  label?: string
  width?: number
  marker?: boolean
  dashed?: boolean
}

const PALETTE: [number, number, number][] = [
  [180, 119, 31], [14, 127, 255], [44, 160, 44], [40, 39, 214],
  [189, 103, 148], [75, 86, 140], [194, 119, 227], [127, 127, 127],
]

interface Entry { label: string; color: [number, number, number] }

/**
 * One figure. Data is collected first and painted on `render`, so the axis
 * limits can be worked out from everything that was added.
 */
export class Figure {
  private readonly cv: any
  private readonly options: FigureOptions
  private readonly draws: ((paint: Painter) => void)[] = []
  private readonly legend: Entry[] = []
  private xMin = Infinity
  private xMax = -Infinity
  private yMin = Infinity
  private yMax = -Infinity
  private nextColour = 0

  constructor(cv: any, options: FigureOptions = {}) {
    this.cv = cv
    this.options = options
  }

  private take(options: SeriesOptions): [number, number, number] {
    return options.color ?? PALETTE[this.nextColour++ % PALETTE.length]
  }

  private extend(xs: ArrayLike<number>, ys: ArrayLike<number>): void {
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i]
      const y = ys[i]
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue
      if (x < this.xMin) this.xMin = x
      if (x > this.xMax) this.xMax = x
      if (y < this.yMin) this.yMin = y
      if (y > this.yMax) this.yMax = y
    }
  }

  line(xs: ArrayLike<number>, ys: ArrayLike<number>, options: SeriesOptions = {}): Figure {
    const colour = this.take(options)
    this.extend(xs, ys)
    if (options.label) this.legend.push({ label: options.label, color: colour })
    this.draws.push((paint) => paint.polyline(xs, ys, colour, options.width ?? 2, options.dashed ?? false, options.marker ?? false))
    return this
  }

  scatter(xs: ArrayLike<number>, ys: ArrayLike<number>, options: SeriesOptions & { size?: number } = {}): Figure {
    const colour = this.take(options)
    this.extend(xs, ys)
    if (options.label) this.legend.push({ label: options.label, color: colour })
    this.draws.push((paint) => paint.points(xs, ys, colour, options.size ?? 2))
    return this
  }

  bars(xs: ArrayLike<number>, ys: ArrayLike<number>, options: SeriesOptions & { barWidth?: number } = {}): Figure {
    const colour = this.take(options)
    this.extend(xs, ys)
    // A bar chart is read against zero, so the axis has to include it.
    if (this.yMin > 0) this.yMin = 0
    if (this.yMax < 0) this.yMax = 0
    if (options.label) this.legend.push({ label: options.label, color: colour })
    this.draws.push((paint) => paint.bars(xs, ys, colour, options.barWidth ?? 0.8))
    return this
  }

  hline(y: number, options: SeriesOptions = {}): Figure {
    const colour = this.take(options)
    if (Number.isFinite(y)) { if (y < this.yMin) this.yMin = y; if (y > this.yMax) this.yMax = y }
    if (options.label) this.legend.push({ label: options.label, color: colour })
    this.draws.push((paint) => paint.hline(y, colour, options.width ?? 1, options.dashed ?? true))
    return this
  }

  vline(x: number, options: SeriesOptions = {}): Figure {
    const colour = this.take(options)
    if (Number.isFinite(x)) { if (x < this.xMin) this.xMin = x; if (x > this.xMax) this.xMax = x }
    if (options.label) this.legend.push({ label: options.label, color: colour })
    this.draws.push((paint) => paint.vline(x, colour, options.width ?? 1, options.dashed ?? true))
    return this
  }

  render(): any {
    const cv = this.cv
    const width = this.options.width ?? 700
    const height = this.options.height ?? 450
    const dark = this.options.dark ?? false
    const background = dark ? 26 : 255
    const image = new cv.Mat(height, width, cv.CV_8UC3, new cv.Scalar(background, background, background, 255))
    const painter = new Painter(cv, image, this.options, this.bounds(), dark)
    painter.axes()
    for (const draw of this.draws) draw(painter)
    painter.frame()
    if (this.legend.length > 0) painter.legend(this.legend)
    return image
  }

  /** Axis limits: whatever was asked for, else the data padded a little. */
  private bounds(): { xMin: number; xMax: number; yMin: number; yMax: number } {
    let [xMin, xMax] = this.options.xlim ?? [this.xMin, this.xMax]
    let [yMin, yMax] = this.options.ylim ?? [this.yMin, this.yMax]
    if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) { xMin = 0; xMax = 1 }
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) { yMin = 0; yMax = 1 }
    if (!this.options.ylim) {
      const pad = (yMax - yMin) * 0.08 || 0.5
      yMin -= pad
      yMax += pad
    }
    if (xMax === xMin) xMax = xMin + 1
    if (yMax === yMin) yMax = yMin + 1
    return { xMin, xMax, yMin, yMax }
  }
}

/** Does the actual drawing, in pixels. */
class Painter {
  private readonly left: number
  private readonly top: number
  private readonly plotWidth: number
  private readonly plotHeight: number
  private readonly ink: any
  private readonly grid: any
  private readonly font: number

  constructor(
    private readonly cv: any,
    private readonly image: any,
    private readonly options: FigureOptions,
    private readonly limits: { xMin: number; xMax: number; yMin: number; yMax: number },
    dark: boolean
  ) {
    this.left = 68
    this.top = options.title ? 34 : 16
    this.plotWidth = image.cols - this.left - 18
    this.plotHeight = image.rows - this.top - 46
    const shade = dark ? 210 : 40
    this.ink = new cv.Scalar(shade, shade, shade, 255)
    const gridShade = dark ? 60 : 214
    this.grid = new cv.Scalar(gridShade, gridShade, gridShade, 255)
    this.font = cv.FONT_HERSHEY_SIMPLEX
  }

  /** Log scale is applied to the y axis only, as the desktop figures use it. */
  private ty(y: number): number {
    const { yMin, yMax } = this.limits
    if (this.options.logY) {
      const low = Math.log10(Math.max(yMin, 1e-12))
      const high = Math.log10(Math.max(yMax, 1e-11))
      const value = Math.log10(Math.max(y, 1e-12))
      return this.top + this.plotHeight - ((value - low) / (high - low || 1)) * this.plotHeight
    }
    return this.top + this.plotHeight - ((y - yMin) / (yMax - yMin || 1)) * this.plotHeight
  }

  private tx(x: number): number {
    const { xMin, xMax } = this.limits
    return this.left + ((x - xMin) / (xMax - xMin || 1)) * this.plotWidth
  }

  private text(s: string, x: number, y: number, scale = 0.36, colour = this.ink): void {
    this.cv.putText(this.image, s, new this.cv.Point(Math.round(x), Math.round(y)), this.font, scale, colour, 1, this.cv.LINE_AA)
  }

  private label(value: number): string {
    if (value === 0) return '0'
    const magnitude = Math.abs(value)
    if (magnitude >= 1e4 || magnitude < 1e-3) return value.toExponential(1)
    return String(Number(value.toPrecision(4)))
  }

  axes(): void {
    const { xMin, xMax, yMin, yMax } = this.limits
    for (let i = 0; i <= 4; i++) {
      const y = this.top + (this.plotHeight * i) / 4
      this.cv.line(this.image, new this.cv.Point(this.left, Math.round(y)), new this.cv.Point(this.left + this.plotWidth, Math.round(y)), this.grid, 1)
      const value = this.options.logY
        ? 10 ** (Math.log10(Math.max(yMax, 1e-11)) - ((Math.log10(Math.max(yMax, 1e-11)) - Math.log10(Math.max(yMin, 1e-12))) * i) / 4)
        : yMax - ((yMax - yMin) * i) / 4
      this.text(this.label(value), 4, y + 4)

      const x = this.left + (this.plotWidth * i) / 4
      this.cv.line(this.image, new this.cv.Point(Math.round(x), this.top), new this.cv.Point(Math.round(x), this.top + this.plotHeight), this.grid, 1)
      this.text(this.label(xMin + ((xMax - xMin) * i) / 4), x - 16, this.image.rows - 26)
    }
    if (this.options.title) this.text(this.options.title, this.left, 22, 0.46)
    if (this.options.xlabel) this.text(this.options.xlabel, this.left + this.plotWidth / 2 - 40, this.image.rows - 8, 0.38)
    if (this.options.ylabel) this.text(this.options.ylabel, 4, this.top - 5, 0.38)
  }

  frame(): void {
    this.cv.rectangle(this.image, new this.cv.Point(this.left, this.top),
      new this.cv.Point(this.left + this.plotWidth, this.top + this.plotHeight), this.ink, 1)
  }

  polyline(xs: ArrayLike<number>, ys: ArrayLike<number>, colour: [number, number, number], width: number, dashed: boolean, marker: boolean): void {
    const scalar = new this.cv.Scalar(colour[0], colour[1], colour[2], 255)
    let previous: { x: number; y: number } | null = null
    for (let i = 0; i < xs.length; i++) {
      if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) { previous = null; continue }
      const point = { x: this.tx(xs[i]), y: this.ty(ys[i]) }
      if (previous && (!dashed || i % 2 === 0)) {
        this.cv.line(this.image, new this.cv.Point(Math.round(previous.x), Math.round(previous.y)),
          new this.cv.Point(Math.round(point.x), Math.round(point.y)), scalar, width, this.cv.LINE_AA)
      }
      if (marker) this.cv.circle(this.image, new this.cv.Point(Math.round(point.x), Math.round(point.y)), 3, scalar, -1, this.cv.LINE_AA)
      previous = point
    }
  }

  points(xs: ArrayLike<number>, ys: ArrayLike<number>, colour: [number, number, number], size: number): void {
    const scalar = new this.cv.Scalar(colour[0], colour[1], colour[2], 255)
    for (let i = 0; i < xs.length; i++) {
      if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) continue
      this.cv.circle(this.image, new this.cv.Point(Math.round(this.tx(xs[i])), Math.round(this.ty(ys[i]))), size, scalar, -1, this.cv.LINE_AA)
    }
  }

  bars(xs: ArrayLike<number>, ys: ArrayLike<number>, colour: [number, number, number], barWidth: number): void {
    const scalar = new this.cv.Scalar(colour[0], colour[1], colour[2], 255)
    const step = xs.length > 1 ? Math.abs(this.tx(xs[1]) - this.tx(xs[0])) : this.plotWidth / 4
    const half = Math.max(1, (step * barWidth) / 2)
    const base = this.ty(0)
    for (let i = 0; i < xs.length; i++) {
      if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) continue
      const x = this.tx(xs[i])
      const y = this.ty(ys[i])
      this.cv.rectangle(this.image, new this.cv.Point(Math.round(x - half), Math.round(Math.min(base, y))),
        new this.cv.Point(Math.round(x + half), Math.round(Math.max(base, y))), scalar, -1)
    }
  }

  hline(y: number, colour: [number, number, number], width: number, dashed: boolean): void {
    const scalar = new this.cv.Scalar(colour[0], colour[1], colour[2], 255)
    const py = Math.round(this.ty(y))
    if (!dashed) {
      this.cv.line(this.image, new this.cv.Point(this.left, py), new this.cv.Point(this.left + this.plotWidth, py), scalar, width, this.cv.LINE_AA)
      return
    }
    for (let x = this.left; x < this.left + this.plotWidth; x += 10) {
      this.cv.line(this.image, new this.cv.Point(x, py), new this.cv.Point(Math.min(x + 5, this.left + this.plotWidth), py), scalar, width, this.cv.LINE_AA)
    }
  }

  vline(x: number, colour: [number, number, number], width: number, dashed: boolean): void {
    const scalar = new this.cv.Scalar(colour[0], colour[1], colour[2], 255)
    const px = Math.round(this.tx(x))
    if (!dashed) {
      this.cv.line(this.image, new this.cv.Point(px, this.top), new this.cv.Point(px, this.top + this.plotHeight), scalar, width, this.cv.LINE_AA)
      return
    }
    for (let y = this.top; y < this.top + this.plotHeight; y += 10) {
      this.cv.line(this.image, new this.cv.Point(px, y), new this.cv.Point(px, Math.min(y + 5, this.top + this.plotHeight)), scalar, width, this.cv.LINE_AA)
    }
  }

  legend(entries: Entry[]): void {
    const x = this.left + this.plotWidth - 168
    entries.slice(0, 8).forEach((entry, i) => {
      const y = this.top + 16 + i * 15
      const scalar = new this.cv.Scalar(entry.color[0], entry.color[1], entry.color[2], 255)
      this.cv.line(this.image, new this.cv.Point(x, y - 4), new this.cv.Point(x + 16, y - 4), scalar, 2, this.cv.LINE_AA)
      this.text(entry.label.slice(0, 26), x + 21, y, 0.33)
    })
  }
}

/** What the script sandbox receives as `plot`. */
export function makePlot(cv: any) {
  return {
    figure: (options: FigureOptions = {}) => new Figure(cv, options),
  }
}
