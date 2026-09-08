/**
 * The caption an analysis node draws over its own result image.
 *
 * These images are shown inside a node card, which scales them to its width —
 * so text baked in at a fixed size is magnified on a small raster and shrunk to
 * nothing on a large one. Everything here is therefore sized from the image
 * itself, and painted in the application's own palette so a node preview reads
 * as part of the interface rather than as a screenshot dropped into it.
 */

/** The interface palette, in the BGR order OpenCV expects. */
export const UI = {
  panel: [48, 37, 30] as [number, number, number],        // #1e2530 bg-primary
  panelLight: [63, 51, 44] as [number, number, number],   // #2c333f bg-secondary
  accent: [240, 124, 0] as [number, number, number],      // #007cf0
  text: [222, 222, 222] as [number, number, number],
  dim: [160, 160, 160] as [number, number, number],
}

export interface LegendEntry {
  label: string
  color: [number, number, number]
}

export interface CaptionOptions {
  /** The headline value, drawn larger and in the accent colour. */
  headline?: string
  /** Secondary figures, one per line under the headline. */
  lines?: string[]
  /** Colour key drawn along the bottom. */
  legend?: LegendEntry[]
  /** 0-1; how opaque the panels are over the image. */
  opacity?: number
}

/** Below this the raster has too few pixels to carry legible type. */
const MIN_DISPLAY_WIDTH = 360

/**
 * Type size for this image.
 *
 * A node card is a fixed width — around 220 px — so the image is scaled to fit
 * whatever its own size is: a 96 px raster is magnified, a 900 px scene shrunk
 * four times over. For the caption to come out the same apparent size either
 * way it has to be sized as a fraction of the image, not in absolute pixels.
 *
 * The constant is derived rather than guessed: HERSHEY caps are about 22·scale
 * pixels tall, and the target is roughly 11 px on screen once the card has
 * scaled the image, which gives scale = width / 440.
 */
function typeScale(width: number): number {
  return Math.max(width / 440, 0.34)
}

/**
 * Grows a small overlay until type will fit on it.
 *
 * These images go to a viewer, never into further measurement, so enlarging one
 * costs nothing; an integer factor with nearest-neighbour keeps mask edges hard
 * instead of smearing them. Anything already wide enough is returned untouched.
 */
export function forDisplay(cv: any, image: any, track: (m: any) => any): any {
  if (image.cols >= MIN_DISPLAY_WIDTH) return image
  const factor = Math.ceil(MIN_DISPLAY_WIDTH / image.cols)
  const grown = track(new cv.Mat())
  cv.resize(image, grown, new cv.Size(image.cols * factor, image.rows * factor), 0, 0, cv.INTER_NEAREST)
  return grown
}

/** Fills a rectangle by blending, so the image still shows through the panel. */
function blendRect(cv: any, image: any, x0: number, y0: number, x1: number, y1: number, colour: [number, number, number], alpha: number): void {
  const left = Math.max(0, Math.round(x0))
  const top = Math.max(0, Math.round(y0))
  const right = Math.min(image.cols, Math.round(x1))
  const bottom = Math.min(image.rows, Math.round(y1))
  if (right <= left || bottom <= top) return
  const data = image.data as Uint8Array
  const channels = image.channels()
  for (let y = top; y < bottom; y++) {
    for (let x = left; x < right; x++) {
      const at = (y * image.cols + x) * channels
      for (let c = 0; c < 3 && c < channels; c++) {
        data[at + c] = Math.round(data[at + c] * (1 - alpha) + colour[c] * alpha)
      }
    }
  }
}

/**
 * Draws the caption in place.
 *
 * The headline goes top-left on a translucent bar, the secondary lines under it,
 * and the colour key on a matching bar along the bottom.
 */
export function drawCaption(cv: any, image: any, options: CaptionOptions): void {
  const width = image.cols
  const height = image.rows
  const scale = typeScale(width)
  const opacity = options.opacity ?? 0.72
  const font = cv.FONT_HERSHEY_SIMPLEX
  const thickness = scale >= 0.75 ? 2 : 1

  const pad = Math.max(4, Math.round(width * 0.014))
  const headlineHeight = options.headline ? Math.round(26 * scale) : 0
  const lineHeight = Math.round(19 * scale)
  const lines = options.lines ?? []
  const topHeight = pad + headlineHeight + lines.length * lineHeight + (lines.length ? pad / 2 : 0)

  if (options.headline || lines.length) {
    blendRect(cv, image, 0, 0, width, topHeight, UI.panel, opacity)
    let y = pad + Math.round(headlineHeight * 0.78)
    if (options.headline) {
      cv.putText(image, options.headline, new cv.Point(pad, y), font, scale * 0.72,
        new cv.Scalar(UI.accent[0], UI.accent[1], UI.accent[2], 255), thickness, cv.LINE_AA)
      y += Math.round(lineHeight * 0.5)
    }
    for (const line of lines) {
      y += lineHeight
      cv.putText(image, line, new cv.Point(pad, y - Math.round(5 * scale)), font, scale * 0.5,
        new cv.Scalar(UI.text[0], UI.text[1], UI.text[2], 255), 1, cv.LINE_AA)
    }
  }

  const legend = options.legend ?? []
  if (legend.length === 0) return

  const chip = Math.max(5, Math.round(9 * scale))
  const barHeight = Math.round(chip + 12 * scale)
  const barTop = height - barHeight
  blendRect(cv, image, 0, barTop, width, height, UI.panel, opacity)

  let x = pad
  const textY = barTop + Math.round(barHeight * 0.72)
  for (const entry of legend) {
    cv.rectangle(image, new cv.Point(x, textY - chip), new cv.Point(x + chip, textY),
      new cv.Scalar(entry.color[0], entry.color[1], entry.color[2], 255), -1)
    cv.putText(image, entry.label, new cv.Point(x + chip + Math.round(4 * scale), textY), font, scale * 0.44,
      new cv.Scalar(UI.text[0], UI.text[1], UI.text[2], 255), 1, cv.LINE_AA)
    // Advance past the chip and its label. cv.getTextSize has no binding in this
    // build, so the width is estimated: HERSHEY_SIMPLEX advances about 20 px per
    // character at scale 1, and the extra ten leaves a gap between entries.
    x += chip + Math.round((10 + entry.label.length * 20 * 0.44) * scale)
    if (x > width - pad) break
  }
}

/** The colours the mask-comparison overlays share, so they stay consistent. */
export const MASK_COLOURS = {
  match: [90, 200, 90] as [number, number, number],       // true positive
  falsePositive: [90, 90, 235] as [number, number, number],
  falseNegative: [210, 120, 60] as [number, number, number],
  prediction: [80, 200, 235] as [number, number, number],
  truth: [235, 160, 70] as [number, number, number],
}
