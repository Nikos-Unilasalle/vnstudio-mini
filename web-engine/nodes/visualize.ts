import type { NodeImpl } from '../types'
import type { MeasuredRegion } from './measure'
import { makeCanvas, canvasToBase64 } from '../canvasCompat'

interface PlotterState {
  series: Map<string, { tick: number; value: number }[]>
  lastReset: unknown
  /** Auto-incrementing fallback when no `ticks` input is wired up. */
  tick: number
}

const SERIES_COLOURS = ['#4f8cff', '#57c785', '#f5a623', '#ff5c5c', '#b083f0', '#4dd0e1']

/**
 * `PlotterProNode` (src/components/nodes/scientific.tsx) builds its own
 * frame history client-side from live per-series numbers and a `_tick`
 * counter — like GrainHistogramNodeUI, it never reads the `main`/preview
 * image this also renders. That image stays (declared as the `main` output
 * port, for anyone who patches it into a generic Display node instead of
 * relying on the inline chart), but the values below are what the node's own
 * widget actually needs.
 */
export const plotterPro: NodeImpl = async (inputs, params, ctx) => {
  let state: PlotterState = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { series: new Map(), lastReset: null, tick: 0 }
    ctx.state.set(ctx.nodeId, state)
  }

  if (params.reset && params.reset !== state.lastReset) {
    state.series.clear()
    state.tick = 0
  }
  state.lastReset = params.reset

  const tick = typeof inputs.ticks === 'number' ? inputs.ticks : state.tick++
  const bufferSize = Math.max(2, Number(params.buffer_size) || 200)

  // Every connected port other than `ticks` is a curve, named after the port —
  // that is how the desktop node turns dynamic inputs into a multi-series plot.
  const seriesKeys: string[] = []
  for (const [port, value] of Object.entries(inputs)) {
    if (port === 'ticks' || typeof value !== 'number') continue
    seriesKeys.push(port)
    let points = state.series.get(port)
    if (!points) {
      points = []
      state.series.set(port, points)
    }
    points.push({ tick, value })
    if (points.length > bufferSize) points.splice(0, points.length - bufferSize)
    ctx.emit(port, value)
  }
  ctx.emit('series_keys', seriesKeys)
  ctx.emit('_tick', tick)

  const width = Math.max(100, Number(params.width) || 640)
  const height = Math.max(100, Number(params.height) || 360)
  ctx.emit('main_preview', await renderPlot(state.series, width, height, !!params.normalize, params.show_grid !== false))

  const summary: Record<string, number> = {}
  for (const [name, points] of state.series) {
    if (points.length > 0) summary[name] = points[points.length - 1].value
  }

  return { main: null, dict: summary, table: seriesToTable(state.series) }
}

function seriesToTable(series: Map<string, { tick: number; value: number }[]>): Record<string, unknown>[] {
  const byTick = new Map<number, Record<string, unknown>>()
  for (const [name, points] of series) {
    for (const point of points) {
      let row = byTick.get(point.tick)
      if (!row) {
        row = { tick: point.tick }
        byTick.set(point.tick, row)
      }
      row[name] = point.value
    }
  }
  return [...byTick.values()].sort((a, b) => (a.tick as number) - (b.tick as number))
}

async function renderPlot(
  series: Map<string, { tick: number; value: number }[]>,
  width: number,
  height: number,
  normalize: boolean,
  showGrid: boolean
): Promise<string> {
  const canvas = makeCanvas(width, height)
  const ctx = canvas.getContext('2d')!

  ctx.fillStyle = '#1e2530'
  ctx.fillRect(0, 0, width, height)

  const marginLeft = 48
  const marginBottom = 26
  const marginTop = 14
  const plotWidth = width - marginLeft - 12
  const plotHeight = height - marginTop - marginBottom

  const curves = [...series.entries()].filter(([, points]) => points.length > 1)
  if (curves.length === 0) {
    ctx.fillStyle = '#7a8593'
    ctx.font = '12px sans-serif'
    ctx.fillText('en attente de données…', marginLeft, marginTop + plotHeight / 2)
    return canvasToBase64(canvas, 0.8)
  }

  const allTicks = curves.flatMap(([, points]) => points.map((p) => p.tick))
  const minTick = Math.min(...allTicks)
  const maxTick = Math.max(...allTicks)
  const tickSpan = maxTick - minTick || 1

  // Normalising rescales every curve to its own 0..1 band, so series with very
  // different magnitudes stay comparable in shape.
  const globalValues = curves.flatMap(([, points]) => points.map((p) => p.value))
  const globalMin = Math.min(...globalValues)
  const globalMax = Math.max(...globalValues)

  if (showGrid) {
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'
    ctx.lineWidth = 1
    for (let i = 1; i < 4; i++) {
      const y = marginTop + (plotHeight * i) / 4
      ctx.beginPath()
      ctx.moveTo(marginLeft, y)
      ctx.lineTo(marginLeft + plotWidth, y)
      ctx.stroke()
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.35)'
    ctx.beginPath()
    ctx.moveTo(marginLeft, marginTop)
    ctx.lineTo(marginLeft, marginTop + plotHeight)
    ctx.lineTo(marginLeft + plotWidth, marginTop + plotHeight)
    ctx.stroke()
  }

  curves.forEach(([name, points], index) => {
    const values = points.map((p) => p.value)
    const low = normalize ? Math.min(...values) : globalMin
    const high = normalize ? Math.max(...values) : globalMax
    const span = high - low || 1

    ctx.strokeStyle = SERIES_COLOURS[index % SERIES_COLOURS.length]
    ctx.lineWidth = 2
    ctx.beginPath()
    points.forEach((point, i) => {
      const x = marginLeft + ((point.tick - minTick) / tickSpan) * plotWidth
      const y = marginTop + plotHeight - ((point.value - low) / span) * plotHeight
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    ctx.fillStyle = SERIES_COLOURS[index % SERIES_COLOURS.length]
    ctx.font = '11px sans-serif'
    ctx.fillText(name, marginLeft + 6 + index * 90, marginTop + 11)
  })

  ctx.fillStyle = '#aab4c0'
  ctx.font = '11px sans-serif'
  if (!normalize) {
    ctx.fillText(globalMax.toFixed(2), 4, marginTop + 10)
    ctx.fillText(globalMin.toFixed(2), 4, marginTop + plotHeight)
  }
  ctx.fillText(String(Math.round(minTick)), marginLeft, height - 8)
  ctx.fillText(String(Math.round(maxTick)), marginLeft + plotWidth - 28, height - 8)

  return canvasToBase64(canvas, 0.8)
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  const position = (p / 100) * (sorted.length - 1)
  const low = Math.floor(position)
  const high = Math.ceil(position)
  if (low === high) return sorted[low]
  return sorted[low] * (high - position) + sorted[high] * (position - low)
}

/**
 * `GrainHistogramNodeUI` (src/components/nodes/scientific.tsx) renders its own
 * Recharts bar+line chart from raw per-bin numbers — it never reads an image
 * output. This used to render a canvas PNG instead (matching how plotterPro
 * still does, for nodes without a native chart widget), which is why the node
 * produced *an* image but the UI — which was never looking for one — showed
 * nothing.
 */
export const geoGrainHistogram: NodeImpl = (inputs, params, ctx) => {
  const regions = inputs.regions as MeasuredRegion[] | undefined
  if (!regions || regions.length === 0) {
    ctx.emit('error', 'Awaiting data…')
    return {}
  }

  const metric = Number(params.metric) || 0
  const calibrated = regions[0].equivalent_diameter_um !== undefined
  const values = regions
    .map((r) => {
      if (metric === 3) return calibrated ? (r.area_um2 ?? r.area) : r.area
      return calibrated ? (r.equivalent_diameter_um ?? r.equivalent_diameter) : r.equivalent_diameter
    })
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b)

  if (values.length === 0) {
    ctx.emit('error', 'Awaiting data…')
    return {}
  }

  const unit = calibrated ? (metric === 3 ? 'µm²' : 'µm') : metric === 3 ? 'px²' : 'px'
  const label = metric === 3 ? 'Area' : 'Diameter'

  const min = values[0]
  const max = values[values.length - 1]
  const binCount = Math.max(4, Number(params.bins) || 30)
  const binWidth = (max - min) / binCount || 1
  const counts = new Array(binCount).fill(0)
  for (const value of values) {
    const index = Math.min(binCount - 1, Math.max(0, Math.floor((value - min) / binWidth)))
    counts[index]++
  }
  const bins = counts.map((_, i) => min + (i + 0.5) * binWidth)
  let cumulativeCount = 0
  const cumulative = counts.map((count) => {
    cumulativeCount += count
    return (cumulativeCount / values.length) * 100
  })

  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length

  ctx.emit('bins', bins)
  ctx.emit('counts', counts)
  ctx.emit('cumulative', cumulative)
  ctx.emit('d10', percentile(values, 10))
  ctx.emit('d50', percentile(values, 50))
  ctx.emit('d90', percentile(values, 90))
  ctx.emit('count', values.length)
  ctx.emit('mean', Number(mean.toFixed(2)))
  ctx.emit('std', Number(Math.sqrt(variance).toFixed(2)))
  ctx.emit('unit', unit)
  ctx.emit('label', label)
  return {}
}
