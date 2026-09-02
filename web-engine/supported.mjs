/**
 * Node types the browser build ships.
 *
 * The desktop registry has ~470 nodes. Everything requiring a Python runtime,
 * a large model download, native GDAL/GEE access, hardware (serial, VNPad) or
 * a filesystem walk is left out — the point of the web build is that a student
 * opens a URL and it works, with no install and no server.
 *
 * Keep this list in sync with the implementations in web-engine/nodes/.
 */
export const SUPPORTED_TYPES = [
  // canvas / structural
  'canvas_note',
  'canvas_frame',

  // input
  'input_image',
  'input_movie',
  'input_webcam',
  'input_solid_color',

  // image
  'filter_gray',
  'filter_blur',
  'filter_canny',
  'filter_threshold',
  'plugin_invert',
  'plugin_brightness_contrast',
  'plugin_sobel',
  'cv_colorspace',
  'feat_clahe',

  // segmentation
  'feat_threshold_adv',
  'feat_morphology_adv',
  'feat_distance_transform',
  'feat_connected_components',
  'sci_connected_components',
  'feat_marker_filter',
  'feat_watershed',

  // mask
  'fill_holes',
  'mask_operations',

  // geometry
  'util_roi_polygon',
  'geom_obb',

  // measure
  'sci_marker_filter',
  'sci_region_props',
  'sci_interactive_calibration',
  'sci_calibration',
  'image_moments',
  'sci_analysis_report',

  // geology
  'geo_grain_histogram',

  // analysis
  'forensic_footprint',

  // body / tracking
  'analysis_face_mp',
  'analysis_hand_mp',
  'geom_track_point',

  // data
  'data_coord_combine',
  'data_coord_splitter',
  'data_list_selector',
  'plugin_dict_get',

  // math
  'math_add',
  'math_sub',
  'math_mul',
  'math_div',
  'math_mod',
  'math_min',
  'math_max',
  'math_pow',
  'math_abs',
  'math_round',
  'math_sin',
  'math_cos',
  'math_clamp',
  'math_distance',
  'scalar_input',

  // logic
  'logic_python',

  // visualize / output
  'plotter_pro',
  'data_inspector',
  'output_display',
  'util_csv_export',
]
