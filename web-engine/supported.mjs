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
  'cv_gamma',
  'cv_levels',
  'cv_shadow_highlight',
  'cv_adaptive_threshold',
  'cv_white_balance',
  'feat_bilateral',
  'filter_noise_gaussian',
  'filter_noise_salt_pepper',
  'filter_noise_speckle',
  'filter_gabor',
  'filter_high_pass',
  'filter_low_pass',
  'filter_laplacian',
  'filter_img_clamp',
  'plugin_gradient',
  'plugin_channel_split',
  'plugin_channel_merge',
  'plugin_blend_modes',
  'plugin_pixelate',
  'filter_glitch',

  // segmentation
  'feat_threshold_adv',
  'feat_morphology_adv',
  'feat_distance_transform',
  'feat_connected_components',
  'sci_connected_components',
  'feat_marker_filter',
  'feat_watershed',
  'feat_find_contours',
  'feat_contour_props',
  'feat_filter_contours',
  'feat_fill_contours',
  'feat_hough_circles',
  'feat_hough_lines',
  'cv_kmeans_segmentation',

  // measure (texture / quality)
  'cv_lbp',
  'cv_ssim',

  // mask
  'fill_holes',
  'mask_operations',
  'mask_circle',
  'mask_point_query',
  'plugin_mask_to_image',
  'cv_skeletonize',
  'cv_directional_dilate',
  'filter_float_threshold',

  // draw
  'draw_arrow',
  'draw_ellipse',
  'draw_line',
  'draw_point',
  'draw_rect',
  'draw_text',
  'draw_overlay',
  'draw_tint_mask',

  // geometry
  'util_roi_polygon',
  'geom_obb',
  'cv_undistort',
  'plugin_rotate',
  'plugin_offset',
  'geom_crop_rect',
  'geom_cropper',
  'remove_vignette',

  // measure
  'sci_marker_filter',
  'sci_region_props',
  'sci_interactive_calibration',
  'sci_calibration',
  'image_moments',
  'sci_analysis_report',
  'sci_first_order_stats',
  'sci_normalizer',
  'sci_roi_stats',
  'sci_color_distance',
  'sci_delta_e',
  'sci_channel_expr',
  'sci_histogram',
  'sci_hist_compare',
  'sci_mask_metrics',
  'sci_line_profile',
  'sci_scale_bar',
  'sci_focus_metric',
  'sci_noise_estimate',
  'sci_matrix_dist',
  'sci_colormap',
  'sci_range_checker',
  'sci_region_color_stats',
  'sci_frame_accumulator',
  'sci_glcm',
  'sci_hausdorff',
  'sci_boundary_f1',
  'sci_robust_bbox',
  'sci_region_classifier',
  'sci_cluster_heatmap',
  'sci_scalar_list',
  'sci_robust_location',
  'sci_mad_scale',
  'sci_m_estimator',
  'feat_seeds_from_boundaries',
  'sci_general_segmenter',
  'filter_linearity',
  'mask_region_sealer',
  'filter_linear_direction',
  'filter_directional_morphology',

  // generative / EVM
  'gen_canvas',
  'gen_feedback',
  'gen_gray_scott',
  'plugin_evm_color',
  'plugin_evm_motion',

  // geology
  'geo_grain_histogram',

  // analysis
  'forensic_footprint',

  // body / tracking
  'analysis_face_mp',
  'analysis_hand_mp',
  'analysis_pose_mp',
  'analysis_object_mp',
  'geom_track_point',

  // data
  'data_coord_combine',
  'data_coord_splitter',
  'data_list_selector',
  'plugin_dict_get',
  'list_region_select',
  'data_list_ops',
  'util_landmark_selector',
  'util_dict_merge',
  'data_group_dicts',
  'dict_builder',

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

  // signal filters
  'plugin_filter_ma',
  'plugin_filter_ema',
  'plugin_filter_kalman',
  'plugin_filter_median',
  'plugin_filter_savgol',
  'plugin_filter_lowpass',
  'plugin_filter_holt',
  'plugin_filter_gaussian',
  'plugin_filter_loess',
  'plugin_filter_particle',

  // logic
  'logic_python',
  'logic_collect',
  'signal_gate',
  'logic_latch',
  'logic_compare',
  'logic_presence',
  'logic_switch',
  'logic_gate',
  'util_filter_label',

  // text
  'string_concat',
  'string_split',
  'string_length',
  'string_case',
  'string_replace',
  'string_input',

  // math (extra)
  'math_expr',
  'math_map_range',
  'math_operation',

  // visualize / output
  'plotter_pro',
  'data_inspector',
  'output_display',
  'util_csv_export',
  // core extras
  'note',
  'reroute',
  'geom_flip',
  'geom_resize',
  'filter_color_mask',
  'filter_morphology',
  'util_coord_to_mask',
  'util_mask_blend',
  'analysis_flow',
  'analysis_flow_viz',
  'analysis_monitor',
  // dataframe
  'df_from_list',
  'df_to_dataframe',
  'df_column_list',
  'df_select',
  'df_sort',
  'df_sample',
  'df_slice',
  'df_groupby',
  'df_merge',
  'df_fillna',
  'df_new_col',
  'df_rename',
  'df_collect',
  'df_editor',
  'df_export',
  'df_plot',

  // keypoints / feature measures
  'feat_orb',
  'feat_sift',
  'feat_fast',
  'feat_harris',
  'feat_matcher',
  'feat_hog',
  'feat_gabor_bank',
  'feat_structure_tensor',
  'feat_ransac_line',
  'feat_skeleton',
  'feat_mask_stats',
  'feat_shape_gate',

  // tracking / motion
  'bg_sub_mog2',
  'bg_sub_knn',
  'filter_bg_subtraction',
  'optical_flow_lk',
  'tracker_sort',
  'tracker_visualize',
  'geom_track_line',
  'geom_track_polygon',

  // geometry (advanced) / boxes
  'geom_approx_poly',
  'geom_fit_shape',
  'geom_warp_affine',
  'geom_rotate_no_crop',
  'geom_perspective',
  'util_manual_points',
  'bbox_transform',
  'box_selection',
  'util_coord_center',

  // mask / utility
  'util_fill_holes',
  'util_colormap',
  'util_image_math',
  'util_draw_contours',
  'util_label_filter_area',
  'mask_filter_area',
  'util_image_masking',
  'mask_outline',
  'util_mask_band',
  'util_split_half',
  'util_compose',
  'plugin_image_to_mask',
  'plugin_invert_mask',
  'points_to_mask',
  'merge_points',

  // analytics / visualisation
  'sci_plotter',
  'sci_stats',
  'sci_heatmap',
  'feat_list_aggregator',
  'sci_kmeans_list',
  'sci_robust_stats',
  'viz_compare_grid',
  'phase_space',
  'imu_phase_dashboard',

  // io / misc
  'signal_clock',
  'signal_generator',
  'color_input',
  'variable_store',
  'output_save_frame',
  'util_snapshot',
  'export_crops',
  'obj_extractor',

  // advanced segmentation
  'filter_blob_filter',
  'filter_morphology_smart',
  'cv_stereo',
  'feat_grabcut',
  'feat_active_contour',
  'cv_ransac',
  'cv_montecarlo_cluster',

  // Fourier / cosine transforms
  'sci_fft',
  'sci_ifft',
  'sci_spectral_gain',
  'sci_dct',

  // indices / misc
  'feat_ndwi',
  'feat_spectral_index',
  'feat_water_refine',
  'mask_to_svg',
  'image_legend',
  'util_on_each',
  'math_3d_to_screen',
  'transform_eye_crop',

  // dataframe analytics / plots
  'ml_df_filter',
  'ml_df_stats',
  'ml_dataframe_join',
  'ml_scatter_plot',
  'ml_histogram',
  'ml_corr_heatmap',

  // supervised / unsupervised learning
  'ml_kmeans',
  'ml_pca',
  'ml_train_test_split',
  'ml_knn_classifier',
  'ml_linear_regression',
  'ml_cluster_validity',
  'ml_classification_report',
  'ml_decision_tree',
  'ml_random_forest',
  'ml_robust_line',
  'ml_best_params',
]









