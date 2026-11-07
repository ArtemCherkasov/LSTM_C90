#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "../include/utils.h"
#include "../include/data.h"
#include "../include/ohlc/data_ohlc.h"
#include "../include/direction/data_direction.h"
#include "../include/lstm.h"
#include "../include/model.h"
#include "../include/types.h"

/* ==================== Shared test constants ==================== */

/* Test output size for small tests */
#define TEST_NUM_OUT 4

/* Deterministic forward 64x64x6 */
#define DET_N 64
#define DET_F 6
#define DET_T 64
#define DET_W 0.1
#define DET_X 0.5

/* Deterministic forward 128x256x12 */
#define DET2_N 128
#define DET2_F 12
#define DET2_T 256
#define DET2_W 0.05
#define DET2_X 0.3

/* Deterministic forward 32x8x26 (full feature count) */
#define DET3_N 32
#define DET3_F 26
#define DET3_T 8
#define DET3_W 0.02
#define DET3_X 0.4

/* ==================== Shared helpers ==================== */

LSTMCellState *alloc_cell_states(int seq_len, int n, int combined);
void generate_synthetic_dataset(DataRow **out_rows, int *out_count);
void set_constant_weights(LSTMNetwork *net, double w);

/* ==================== Test function declarations ==================== */

int test_forward_pass(void);
int test_concatenated_storage(void);
int test_state_propagation(void);

int test_backward_pass(void);
int test_last_timestep_grad(void);
int test_output_grad_accumulation(void);

int test_update(void);
int test_batch_normalization(void);

int test_save_load(void);

int test_deterministic_forward_64(void);
int test_deterministic_forward_128(void);
int test_deterministic_forward_26(void);

int test_bptt_gradient_check_64(void);
int test_bptt_gradient_check_128(void);
int test_bptt_gradient_check_26(void);

/* 26-feature tests */
int test_forward_26_features(void);
int test_backward_26_features(void);

/* Data I/O tests */
int test_load_raw_basic(void);
int test_load_raw_chronological(void);
int test_aggregate_h2(void);
int test_aggregate_h4(void);
int test_aggregate_h8(void);
int test_aggregate_h1_identity(void);
int test_aggregate_double_h2(void);

/* Secondary-data (sdata) tests */
int test_intersect_raw_full_overlap(void);
int test_intersect_raw_partial(void);
int test_intersect_raw_empty(void);
int test_compute_raw_features_base_only(void);
int test_alloc_dataset_ext_null_secondary(void);
int test_alloc_dataset_ext_with_secondary(void);
int test_forward_39_features(void);
int test_backward_39_features(void);

/* OHLC-specific */
int test_training_loop_ohlc(void);
int test_volume_clip_and_label_norm_ohlc(void);
int test_deterministic_ohlc(void);

/* Direction-specific */
int test_direction_labels(void);
int test_training_loop_direction(void);
int test_sigmoid_gradient_check(void);
int test_deterministic_direction(void);

/* sdata fix tests */
int test_mtf_features_finite(void);
int test_mtf_features_bounded(void);
int test_mtf_minmax_no_garbage(void);
int test_sdata_intersection_chronological(void);
int test_sdata_intersection_size(void);
int test_sdata_39_features_finite(void);
int test_sdata_iteration_count(void);

/* timeframe alignment tests */
int test_aggregate_h4_aligned(void);
int test_aggregate_h2_aligned(void);
int test_aggregate_h8_aligned(void);
int test_aggregate_h4_after_gap(void);
int test_aggregate_partial_group_kept(void);

/* datetime shift tests */
int test_datetime_shift_h1(void);
int test_datetime_shift_h4(void);
int test_datetime_shift_month_boundary(void);
int test_datetime_shift_h1_identity(void);

/* parameters file tests */
int test_parameters_load(void);
int test_parameters_blank_defaults(void);
int test_parameters_cfg_loads(void);

#endif /* TEST_UTILS_H */
