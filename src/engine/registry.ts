import type { NodeDef } from './types'
import { inputImageNode } from '../nodes/inputImage'
import { grayscaleNode } from '../nodes/grayscale'
import { thresholdAdvNode } from '../nodes/thresholdAdv'
import { morphologyAdvNode } from '../nodes/morphologyAdv'
import { fillHolesNode } from '../nodes/fillHoles'
import { maskPolygonNode } from '../nodes/maskPolygon'
import { distanceTransformNode } from '../nodes/distanceTransform'
import { connectedComponentsNode } from '../nodes/connectedComponents'
import { connectedCompCv2Node } from '../nodes/connectedCompCv2'
import { markerFilterNode } from '../nodes/markerFilter'
import { watershedNode } from '../nodes/watershed'
import { regionFilterNode } from '../nodes/regionFilter'
import { regionPropsNode } from '../nodes/regionProps'
import { visualCalibrationNode } from '../nodes/visualCalibration'
import { grainHistogramNode } from '../nodes/grainHistogram'
import { displayNode } from '../nodes/display'
import { inspectorNode } from '../nodes/inspector'
import { csvExportNode } from '../nodes/csvExport'
import { canvasFrameNode, canvasNoteNode } from '../nodes/decorative'

export const NODE_DEFS: NodeDef[] = [
  inputImageNode,
  grayscaleNode,
  maskPolygonNode,
  thresholdAdvNode,
  morphologyAdvNode,
  fillHolesNode,
  distanceTransformNode,
  connectedComponentsNode,
  connectedCompCv2Node,
  markerFilterNode,
  watershedNode,
  regionFilterNode,
  regionPropsNode,
  visualCalibrationNode,
  grainHistogramNode,
  displayNode,
  inspectorNode,
  csvExportNode,
  canvasFrameNode,
  canvasNoteNode,
]

const byId = new Map(NODE_DEFS.map((d) => [d.typeId, d]))

export function getNodeDef(typeId: string): NodeDef | undefined {
  return byId.get(typeId)
}

export const CATEGORIES = [...new Set(NODE_DEFS.filter((d) => !d.decorative).map((d) => d.category))]
