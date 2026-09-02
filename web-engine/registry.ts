import type { NodeImpl, WebNodeSchema } from './types'
import rawSchemas from './schemas.json'

import { inputImage, inputMovie, inputSolidColor, inputWebcam } from './nodes/input'
import { cvColorspace, featClahe, filterBlur, filterCanny, filterGray, filterThreshold, pluginBrightnessContrast, pluginInvert, pluginSobel } from './nodes/image'
import {
  featConnectedComponents,
  featDistanceTransform,
  featMarkerFilter,
  featMorphologyAdv,
  featThresholdAdv,
  featWatershed,
  sciConnectedComponents,
} from './nodes/segmentation'
import { fillHoles, maskOperations } from './nodes/mask'
import { geomObb, utilRoiPolygon } from './nodes/geometry'
import { imageMoments, sciAnalysisReport, sciCalibration, sciInteractiveCalibration, sciMarkerFilter, sciRegionProps } from './nodes/measure'
import { forensicFootprint } from './nodes/analysis'
import { analysisFaceMp, analysisHandMp, geomTrackPoint } from './nodes/body'
import {
  dataCoordCombine,
  dataCoordSplitter,
  dataListSelector,
  mathAbs,
  mathAdd,
  mathClamp,
  mathCos,
  mathDistance,
  mathDiv,
  mathMax,
  mathMin,
  mathMod,
  mathMul,
  mathPow,
  mathRound,
  mathSin,
  mathSub,
  pluginDictGet,
  scalarInput,
} from './nodes/dataMath'
import { canvasFrame, canvasNote, logicPython } from './nodes/logic'
import { dataInspector, outputDisplay, utilCsvExport } from './nodes/output'
import { geoGrainHistogram, plotterPro } from './nodes/visualize'

export const SCHEMAS = rawSchemas as unknown as WebNodeSchema[]

export const IMPLEMENTATIONS: Record<string, NodeImpl> = {
  canvas_note: canvasNote,
  canvas_frame: canvasFrame,

  input_image: inputImage,
  input_movie: inputMovie,
  input_webcam: inputWebcam,
  input_solid_color: inputSolidColor,

  filter_gray: filterGray,
  filter_blur: filterBlur,
  filter_canny: filterCanny,
  filter_threshold: filterThreshold,
  plugin_invert: pluginInvert,
  plugin_brightness_contrast: pluginBrightnessContrast,
  plugin_sobel: pluginSobel,
  cv_colorspace: cvColorspace,
  feat_clahe: featClahe,

  feat_threshold_adv: featThresholdAdv,
  feat_morphology_adv: featMorphologyAdv,
  feat_distance_transform: featDistanceTransform,
  feat_connected_components: featConnectedComponents,
  sci_connected_components: sciConnectedComponents,
  feat_marker_filter: featMarkerFilter,
  feat_watershed: featWatershed,

  fill_holes: fillHoles,
  mask_operations: maskOperations,

  util_roi_polygon: utilRoiPolygon,
  geom_obb: geomObb,

  sci_marker_filter: sciMarkerFilter,
  sci_region_props: sciRegionProps,
  sci_interactive_calibration: sciInteractiveCalibration,
  sci_calibration: sciCalibration,
  image_moments: imageMoments,
  sci_analysis_report: sciAnalysisReport,

  geo_grain_histogram: geoGrainHistogram,
  forensic_footprint: forensicFootprint,

  analysis_face_mp: analysisFaceMp,
  analysis_hand_mp: analysisHandMp,
  geom_track_point: geomTrackPoint,

  data_coord_combine: dataCoordCombine,
  data_coord_splitter: dataCoordSplitter,
  data_list_selector: dataListSelector,
  plugin_dict_get: pluginDictGet,

  math_add: mathAdd,
  math_sub: mathSub,
  math_mul: mathMul,
  math_div: mathDiv,
  math_mod: mathMod,
  math_min: mathMin,
  math_max: mathMax,
  math_pow: mathPow,
  math_abs: mathAbs,
  math_round: mathRound,
  math_sin: mathSin,
  math_cos: mathCos,
  math_clamp: mathClamp,
  math_distance: mathDistance,
  scalar_input: scalarInput,

  logic_python: logicPython,

  plotter_pro: plotterPro,
  data_inspector: dataInspector,
  output_display: outputDisplay,
  util_csv_export: utilCsvExport,
}

const schemaByType = new Map(SCHEMAS.map((s) => [s.type, s]))

export function getSchema(type: string): WebNodeSchema | undefined {
  return schemaByType.get(type)
}
