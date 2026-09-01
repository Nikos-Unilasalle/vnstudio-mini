import type { NodeDef } from '../engine/types'
import type { MeasuredRegion } from './regionProps'

function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0
  const idx = (p / 100) * (sorted.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]
  const frac = idx - lo
  return sorted[lo] * (1 - frac) + sorted[hi] * frac
}

export const grainHistogramNode: NodeDef = {
  typeId: 'geo_grain_histogram',
  label: 'Grain Size Histogram',
  category: 'Geology',
  description: 'Distribution des tailles + D10 / D50 / D90.',
  inputs: [{ id: 'main', label: 'regions', color: 'regions' }],
  outputs: [
    { id: 'main', label: 'chart', color: 'image' },
    { id: 'stats', label: 'stats', color: 'dict' },
  ],
  params: [
    { id: 'bins', label: 'Bins', type: 'number', default: 30, min: 4, max: 100, step: 1 },
    {
      id: 'metric',
      label: 'Metric',
      type: 'select',
      default: 0,
      options: [{ label: 'Equiv. Diameter', value: 0 }],
    },
  ],
  process(inputs, params) {
    const regions = inputs.main as MeasuredRegion[] | undefined
    if (!regions || !regions.length) return { main: undefined, stats: undefined }

    const useUm = regions[0].equivDiameterUm != null
    const values = regions
      .map((r) => (useUm ? r.equivDiameterUm! : r.equivDiameterPx))
      .sort((a, b) => a - b)

    const d10 = percentile(values, 10)
    const d50 = percentile(values, 50)
    const d90 = percentile(values, 90)
    const unit = useUm ? 'µm' : 'px'

    const bins = Number(params.bins)
    const min = values[0]
    const max = values[values.length - 1]
    const binWidth = (max - min) / bins || 1
    const counts = new Array(bins).fill(0)
    for (const v of values) {
      let idx = Math.floor((v - min) / binWidth)
      if (idx >= bins) idx = bins - 1
      if (idx < 0) idx = 0
      counts[idx]++
    }

    const chartDataUrl = renderChart(counts, min, binWidth, unit, { d10, d50, d90 })

    return {
      main: chartDataUrl,
      stats: { d10, d50, d90, unit, count: values.length },
    }
  },
}

function renderChart(
  counts: number[],
  min: number,
  binWidth: number,
  unit: string,
  pct: { d10: number; d50: number; d90: number }
): string {
  const W = 480
  const H = 300
  const marginL = 40
  const marginB = 30
  const marginT = 20
  const plotW = W - marginL - 10
  const plotH = H - marginT - marginB

  const canvas = document.createElement('canvas')
  canvas.width = W
  canvas.height = H
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#0b0f14'
  ctx.fillRect(0, 0, W, H)

  const maxCount = Math.max(...counts, 1)
  const barW = plotW / counts.length

  ctx.fillStyle = '#4f8cff'
  counts.forEach((c, i) => {
    const barH = (c / maxCount) * plotH
    ctx.fillRect(marginL + i * barW, marginT + plotH - barH, Math.max(barW - 1, 1), barH)
  })

  ctx.strokeStyle = 'rgba(255,255,255,0.35)'
  ctx.beginPath()
  ctx.moveTo(marginL, marginT)
  ctx.lineTo(marginL, marginT + plotH)
  ctx.lineTo(marginL + plotW, marginT + plotH)
  ctx.stroke()

  const total = counts.length * binWidth
  const xForValue = (v: number) => marginL + ((v - min) / (total || 1)) * plotW

  const lines: [number, string, string][] = [
    [pct.d10, 'D10', '#f5a623'],
    [pct.d50, 'D50', '#ff5c5c'],
    [pct.d90, 'D90', '#7ee787'],
  ]
  for (const [val, label, color] of lines) {
    const x = xForValue(val)
    ctx.strokeStyle = color
    ctx.beginPath()
    ctx.moveTo(x, marginT)
    ctx.lineTo(x, marginT + plotH)
    ctx.stroke()
    ctx.fillStyle = color
    ctx.font = '11px sans-serif'
    ctx.fillText(`${label} ${val.toFixed(1)}${unit}`, Math.min(x + 2, W - 90), marginT + 12)
  }

  ctx.fillStyle = '#aab4c0'
  ctx.font = '11px sans-serif'
  ctx.fillText(`${min.toFixed(0)} ${unit}`, marginL, H - 10)
  ctx.fillText(`${(min + total).toFixed(0)} ${unit}`, W - 60, H - 10)

  return canvas.toDataURL('image/png')
}
