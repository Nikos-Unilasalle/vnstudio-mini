import type { NodeDef } from '../engine/types'

interface PlotterState {
  history: { tick: number; value: number }[]
  lastReset: boolean
}

export const plotterProNode: NodeDef = {
  typeId: 'plotter_pro',
  label: 'Plotter Pro',
  category: 'Visualize',
  description: 'Empile une valeur à chaque exécution (ou à chaque frame en lecture vidéo) et en trace la courbe dans le temps.',
  inputs: [
    { id: 'ticks', label: 'ticks (x)', color: 'scalar' },
    { id: 'value', label: 'value (y)', color: 'scalar' },
  ],
  outputs: [{ id: 'main', label: 'chart', color: 'image' }],
  params: [
    { id: 'buffer_size', label: 'History Size', type: 'number', default: 300, min: 10, max: 10000, step: 10 },
    { id: 'reset', label: 'Reset History', type: 'boolean', default: false },
  ],
  process(inputs, params, ctx) {
    let state: PlotterState = ctx.nodeState.get(ctx.nodeId)
    if (!state) {
      state = { history: [], lastReset: false }
      ctx.nodeState.set(ctx.nodeId, state)
    }

    if (params.reset && !state.lastReset) state.history = []
    state.lastReset = !!params.reset

    const tick = typeof inputs.ticks === 'number' ? inputs.ticks : state.history.length
    const value = typeof inputs.value === 'number' ? inputs.value : null

    if (value !== null) {
      state.history.push({ tick, value })
      const maxLen = Number(params.buffer_size)
      if (state.history.length > maxLen) state.history.splice(0, state.history.length - maxLen)
    }

    return { main: renderChart(state.history) }
  },
}

function renderChart(history: { tick: number; value: number }[]): string {
  const W = 480
  const H = 260
  const marginL = 40
  const marginB = 24
  const marginT = 10
  const plotW = W - marginL - 10
  const plotH = H - marginT - marginB

  const canvas = document.createElement('canvas')
  canvas.width = W
  canvas.height = H
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#0b0f14'
  ctx.fillRect(0, 0, W, H)

  ctx.strokeStyle = 'rgba(255,255,255,0.35)'
  ctx.beginPath()
  ctx.moveTo(marginL, marginT)
  ctx.lineTo(marginL, marginT + plotH)
  ctx.lineTo(marginL + plotW, marginT + plotH)
  ctx.stroke()

  if (history.length < 2) return canvas.toDataURL('image/png')

  const ticks = history.map((h) => h.tick)
  const values = history.map((h) => h.value)
  const minT = Math.min(...ticks)
  const maxT = Math.max(...ticks)
  const minV = Math.min(...values)
  const maxV = Math.max(...values)
  const rangeT = maxT - minT || 1
  const rangeV = maxV - minV || 1

  const xFor = (t: number) => marginL + ((t - minT) / rangeT) * plotW
  const yFor = (v: number) => marginT + plotH - ((v - minV) / rangeV) * plotH

  ctx.strokeStyle = '#4f8cff'
  ctx.lineWidth = 2
  ctx.beginPath()
  history.forEach((h, i) => {
    const x = xFor(h.tick)
    const y = yFor(h.value)
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.stroke()

  ctx.fillStyle = '#aab4c0'
  ctx.font = '11px sans-serif'
  ctx.fillText(minV.toFixed(2), 4, marginT + plotH)
  ctx.fillText(maxV.toFixed(2), 4, marginT + 10)
  ctx.fillText(`n=${history.length}`, marginL, H - 6)

  return canvas.toDataURL('image/png')
}
