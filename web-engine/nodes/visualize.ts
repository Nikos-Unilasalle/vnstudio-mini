import type { NodeImpl } from '../types'
import type { MeasuredRegion } from './measure'

interface PlotterState {
  series: Map<string, { tick: number; value: number }[]>
  lastReset: unknown
}

const SERIES_COLOURS = ['#4f8cff', '#57c785', '#f5a623', '#ff5c5c', '#b083f0', '#4dd0e1']

export const plotterPro: NodeImpl = (inputs, params, ctx) => {
  let state: PlotterState = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { series: new Map(), lastReset: null }
    ctx.state.set(ctx.nodeId, state)
  }

  if (params.reset && params.reset !== state.lastReset) state.series.clear()
  state.lastReset = params.reset

  const tick = typeof inputs.ticks === 'number' ? inputs.ticks : undefined
  const bufferSize = Math.max(2, Number(params.buffer_size) || 200)

  // Every connected port other than `ticks` is a curve, named after the port —
  // that is how the desktop node turns dynamic inputs into a multi-series plot.
  for (const [port, value] of Object.entries(inputs)) {
    if (port === 'ticks' || typeof value !== 'number') continue
    let points = state.series.get(port)
    if (!points) {
      points = []
      state.series.set(port, points)
    }
    points.push({ tick: tick ?? points.length, value })
    if (points.length > bufferSize) points.splice(0, points.length - bufferSize)
  }

  const width = Math.max(100, Number(params.width) || 640)
  const height = Math.max(100, Number(params.height) || 360)
  const dataUrl = renderPlot(state.series, width, height, !!params.normalize, params.show_grid !== false)
  ctx.emit('main_preview', dataUrl.split(',')[1])

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

function renderPlot(
  series: Map<string, { tick: number; value: number }[]>,
  width: number,
  height: number,
  normalize: boolean,
  showGrid: boolean
): string {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
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
    return canvas.toDataURL('image/jpeg', 0.8)
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

  return canvas.toDataURL('image/jpeg', 0.8)
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  const position = (p / 100) * (sorted.length - 1)
  const low = Math.floor(position)
  const high = Math.ceil(position)
  if (low === high) return sorted[low]
  return sorted[low] * (high - position) + sorted[high] * (position - low)
}

export const geoGrainHistogram: NodeImpl = (inputs, params, ctx) => {
  const regions = inputs.regions as MeasuredRegion[] | undefined
  if (!regions || regions.length === 0) {
    ctx.emit('stats', null)
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

  const unit = calibrated ? (metric === 3 ? 'µm²' : 'µm') : metric === 3 ? 'px²' : 'px'
  const stats = {
    d10: percentile(values, 10),
    d50: percentile(values, 50),
    d90: percentile(values, 90),
    count: values.length,
    unit,
  }

  const bins = Math.max(4, Number(params.bins) || 30)
  ctx.emit('main_preview', renderHistogram(values, bins, stats).split(',')[1])
  ctx.emit('stats', stats)
  return {}
}

function renderHistogram(
  values: number[],
  bins: number,
  stats: { d10: number; d50: number; d90: number; unit: string }
): string {
  const width = 520
  const height = 300
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')!

  ctx.fillStyle = '#1e2530'
  ctx.fillRect(0, 0, width, height)

  const marginLeft = 40
  const marginBottom = 30
  const marginTop = 24
  const plotWidth = width - marginLeft - 12
  const plotHeight = height - marginTop - marginBottom

  const min = values[0]
  const max = values[values.length - 1]
  const binWidth = (max - min) / bins || 1
  const counts = new Array(bins).fill(0)
  for (const value of values) {
    const index = Math.min(bins - 1, Math.max(0, Math.floor((value - min) / binWidth)))
    counts[index]++
  }
  const maxCount = Math.max(...counts, 1)

  ctx.fillStyle = '#4f8cff'
  const barWidth = plotWidth / bins
  counts.forEach((count, i) => {
    const barHeight = (count / maxCount) * plotHeight
    ctx.fillRect(marginLeft + i * barWidth, marginTop + plotHeight - barHeight, Math.max(barWidth - 1, 1), barHeight)
  })

  // Cumulative curve — the D-values are read off this, not off the bars.
  ctx.strokeStyle = 'rgba(255,255,255,0.55)'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  let cumulative = 0
  counts.forEach((count, i) => {
    cumulative += count
    const x = marginLeft + (i + 1) * barWidth
    const y = marginTop + plotHeight - (cumulative / values.length) * plotHeight
    if (i === 0) ctx.moveTo(marginLeft, marginTop + plotHeight)
    ctx.lineTo(x, y)
  })
  ctx.stroke()

  ctx.strokeStyle = 'rgba(255,255,255,0.35)'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(marginLeft, marginTop)
  ctx.lineTo(marginLeft, marginTop + plotHeight)
  ctx.lineTo(marginLeft + plotWidth, marginTop + plotHeight)
  ctx.stroke()

  const span = max - min || 1
  const markers: [number, string, string][] = [
    [stats.d10, 'D10', '#f5a623'],
    [stats.d50, 'D50', '#ff5c5c'],
    [stats.d90, 'D90', '#7ee787'],
  ]
  ctx.font = '11px sans-serif'
  markers.forEach(([value, label, colour], i) => {
    const x = marginLeft + ((value - min) / span) * plotWidth
    ctx.strokeStyle = colour
    ctx.beginPath()
    ctx.moveTo(x, marginTop)
    ctx.lineTo(x, marginTop + plotHeight)
    ctx.stroke()
    ctx.fillStyle = colour
    ctx.fillText(`${label} ${value.toFixed(1)}`, 8 + i * 120, 15)
  })

  ctx.fillStyle = '#aab4c0'
  ctx.fillText(`${min.toFixed(0)} ${stats.unit}`, marginLeft, height - 10)
  ctx.fillText(`${max.toFixed(0)} ${stats.unit}`, marginLeft + plotWidth - 60, height - 10)

  return canvas.toDataURL('image/jpeg', 0.85)
}
