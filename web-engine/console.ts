/**
 * Renders what a script printed as a terminal-like image.
 *
 * The Python node has no terminal: `print()` used to write to the worker's
 * console, where nobody looks. Its text is collected instead (python.ts) and
 * drawn here, monospaced, so `print(df.head())` keeps its column alignment the
 * way a notebook shows it. An image, rather than a string, because nothing on
 * the canvas renders multi-line text: the Inspector folds newlines away.
 */
import { makeCanvas, matFromCanvas } from './canvasCompat'

const MAX_LINES = 60
const MAX_COLUMNS = 120
const FONT_PX = 13
const LINE_PX = 18
const PAD = 10
const BG = '#1b1f27'
const INK = '#d7dde8'
const DIM = '#7d8796'

/** BGR Mat of the console, registered with `track` for cleanup. */
export function renderConsole(cv: any, track: (mat: any) => any, text: string): any {
  let lines = text.replace(/\r/g, '').replace(/\t/g, '    ').split('\n')
  if (lines.length && lines[lines.length - 1] === '') lines.pop()
  const notes: string[] = []
  if (lines.length === 0) notes.push('(print n’a rien affiché pendant ce passage)')
  if (lines.length > MAX_LINES) {
    notes.push(`… ${lines.length - MAX_LINES} lignes plus haut non affichées`)
    lines = lines.slice(-MAX_LINES)
  }
  lines = lines.map((line) => (line.length > MAX_COLUMNS ? line.slice(0, MAX_COLUMNS - 1) + '…' : line))

  const font = `${FONT_PX}px "DejaVu Sans Mono", Menlo, Consolas, monospace`
  const probe = makeCanvas(8, 8).getContext('2d')!
  probe.font = font
  const charWidth = probe.measureText('M').width || FONT_PX * 0.6
  const longest = Math.max(24, ...lines.map((l) => l.length), ...notes.map((l) => l.length))
  const width = Math.ceil(PAD * 2 + longest * charWidth)
  const height = PAD * 2 + LINE_PX * (lines.length + notes.length)

  const canvas = makeCanvas(width, height)
  const g = canvas.getContext('2d')!
  g.fillStyle = BG
  g.fillRect(0, 0, width, height)
  g.font = font
  g.textBaseline = 'top'
  let y = PAD
  for (const note of notes) {
    g.fillStyle = DIM
    g.fillText(note, PAD, y)
    y += LINE_PX
  }
  g.fillStyle = INK
  for (const line of lines) {
    g.fillText(line, PAD, y)
    y += LINE_PX
  }

  const rgba = track(matFromCanvas(cv, canvas))
  const bgr = track(new cv.Mat())
  cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR)
  return bgr
}
