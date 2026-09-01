import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'
import { computeLabelStats } from './labelStats'

const BACKGROUND = 1

export const watershedNode: NodeDef = {
  typeId: 'sci_watershed',
  label: 'Watershed',
  category: 'Segmentation',
  description:
    'Découpe un masque fusionné en objets distincts à partir des graines (markers). Cell Mask borne la zone où l\'eau peut monter.',
  inputs: [
    { id: 'image', label: 'image', color: 'image' },
    { id: 'markers', label: 'markers', color: 'regions' },
    { id: 'cell_mask', label: 'Cell Mask', color: 'mask' },
  ],
  outputs: [{ id: 'main', label: 'regions', color: 'regions' }],
  params: [{ id: 'rescue_unseeded', label: 'Rescue Unseeded Regions', type: 'boolean', default: true }],
  process(inputs, params, ctx) {
    const image = inputs.image as any
    const seeds = inputs.markers as any
    const cellMask = inputs.cell_mask as any
    if (!image || !seeds || !cellMask) return { main: undefined }
    const cv = ctx.cv

    const bgr = trackMat(new cv.Mat())
    if (image.channels() === 1) cv.cvtColor(image, bgr, cv.COLOR_GRAY2BGR)
    else image.copyTo(bgr)

    // shift seed labels by +1 to reserve label 1 for background
    const markers = trackMat(new cv.Mat(seeds.rows, seeds.cols, cv.CV_32S, new cv.Scalar(0)))
    const seedData = seeds.data32S as Int32Array
    const markerData = markers.data32S as Int32Array
    for (let i = 0; i < seedData.length; i++) {
      markerData[i] = seedData[i] > 0 ? seedData[i] + 1 : 0
    }

    const maskData = cellMask.data as Uint8Array
    for (let i = 0; i < maskData.length; i++) {
      if (maskData[i] === 0) markerData[i] = BACKGROUND
    }

    if (params.rescue_unseeded) {
      const blobLabels = trackMat(new cv.Mat())
      cv.connectedComponents(cellMask, blobLabels, 8, cv.CV_32S)
      const blobData = blobLabels.data32S as Int32Array
      const seededBlobs = new Set<number>()
      for (let i = 0; i < blobData.length; i++) {
        const blob = blobData[i]
        if (blob > 0 && markerData[i] > BACKGROUND) seededBlobs.add(blob)
      }
      const blobStats = computeLabelStats(blobLabels)
      let nextLabel = 2
      const maxExisting = Math.max(1, ...Array.from(seedData).filter((v) => v > 0))
      nextLabel = maxExisting + 2
      for (const [blobId, s] of blobStats) {
        if (seededBlobs.has(blobId)) continue
        const cx = Math.round(s.cx)
        const cy = Math.round(s.cy)
        const idx = cy * blobLabels.cols + cx
        markerData[idx] = nextLabel
        nextLabel++
      }
    }

    cv.watershed(bgr, markers)

    // normalize: background(1) and boundary(-1) become 0, objects renumbered from 1
    const finalData = markers.data32S as Int32Array
    const remap = new Map<number, number>()
    let next = 1
    for (let i = 0; i < finalData.length; i++) {
      const v = finalData[i]
      if (v <= BACKGROUND) {
        finalData[i] = 0
        continue
      }
      let mapped = remap.get(v)
      if (mapped === undefined) {
        mapped = next++
        remap.set(v, mapped)
      }
      finalData[i] = mapped
    }

    const stats = computeLabelStats(markers)
    return { main: { labels: markers, count: stats.size, stats } }
  },
}
