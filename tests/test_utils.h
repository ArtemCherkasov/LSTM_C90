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

int test_bptt_gradient_check_64(void);
int test_bptt_gradient_check_128(void);

/* OHLC-specific */
int test_training_loop_ohlc(void);
int test_volume_clip_and_label_norm_ohlc(void);

/* Direction-specific */
int test_direction_labels(void);
int test_training_loop_direction(void);
int test_sigmoid_gradient_check(void);

#endif /* TEST_UTILS_H */
