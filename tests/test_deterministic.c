#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ==================== Test: Deterministic Forward 64x64x6 ==================== */

int test_deterministic_forward_64(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double *sequence;
    double *preds;
    int t, i, c, o;
    int n, combined;
    double h_scalar, c_scalar;
    double max_h_err, max_c_err;
    double tol;

    printf("[TEST] Deterministic forward: 64 cells, 64 steps, 6 features");
    fflush(stdout);

    n = DET_N;
    combined = n + DET_F;
    tol = 1e-10;

    net = lstm_init(n, DET_F, TEST_NUM_OUT, 42);
    set_constant_weights(net, DET_W);

    sequence = (double *)utils_safe_calloc((size_t)DET_T * DET_F, sizeof(double));
    for (t = 0; t < DET_T; t++) {
        for (i = 0; i < DET_F; i++) {
            sequence[t * DET_F + i] = DET_X;
        }
    }

    cell_states = alloc_cell_states(DET_T, n, combined);
    preds = (double *)utils_safe_calloc((size_t)(TEST_NUM_OUT * DET_T), sizeof(double));
    lstm_forward(net, sequence, cell_states, DET_T, preds);

    h_scalar = 0.0;
    c_scalar = 0.0;
    max_h_err = 0.0;
    max_c_err = 0.0;

    for (t = 0; t < DET_T; t++) {
        double concat_sum = (double)n * h_scalar + (double)DET_F * DET_X;
        double gate_pre = DET_W * concat_sum;
        double f_gate = utils_sigmoid(gate_pre);
        double i_gate = utils_sigmoid(gate_pre);
        double g_gate = utils_tanh_act(gate_pre);
        double o_gate = utils_sigmoid(gate_pre);
        double c_new = f_gate * c_scalar + i_gate * g_gate;
        double h_new = o_gate * utils_tanh_act(c_new);

        for (c = 0; c < n; c++) {
            double h_err = fabs(cell_states[t].hidden_state[c] - h_new);
            double c_err = fabs(cell_states[t].cell_state[c] - c_new);
            if (h_err > max_h_err) max_h_err = h_err;
            if (c_err > max_c_err) max_c_err = c_err;

            if (fabs(cell_states[t].forget_gate[c] - f_gate) > tol) {
                printf("\n  FAIL: forget_gate[%d][%d] mismatch\n", t, c);
                free(sequence);
                free(preds);
                lstm_free_cell_states(cell_states, DET_T);
                lstm_free(net);
                return 0;
            }
        }

        /* Check outputs at this timestep */
        {
            double expected_pred = DET_W * (double)n * h_new;
            for (o = 0; o < TEST_NUM_OUT; o++) {
                if (fabs(preds[t * TEST_NUM_OUT + o] - expected_pred) > 1e-8) {
                    printf("\n  FAIL: pred[%d][%d]=%.10f expected=%.10f\n", t, o, preds[t * TEST_NUM_OUT + o], expected_pred);
                    free(sequence);
                    free(preds);
                    lstm_free_cell_states(cell_states, DET_T);
                    lstm_free(net);
                    return 0;
                }
            }
        }

        h_scalar = h_new;
        c_scalar = c_new;
    }

    printf("\n  max_h_err=%.2e  max_c_err=%.2e  pred[0][0]=%.8f", max_h_err, max_c_err, preds[0]);

    if (max_h_err < tol && max_c_err < tol) {
        printf("\n  PASS: H and C vectors match analytical solution at all %d steps\n", DET_T);
        free(sequence);
        free(preds);
        lstm_free_cell_states(cell_states, DET_T);
        lstm_free(net);
        return 1;
    } else {
        printf("\n  FAIL: H or C error exceeds tolerance %.2e\n", tol);
        free(sequence);
        free(preds);
        lstm_free_cell_states(cell_states, DET_T);
        lstm_free(net);
        return 0;
    }
}

/* ==================== Test: Deterministic Forward 128x256x12 ==================== */

int test_deterministic_forward_128(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double *sequence;
    double *preds;
    int t, i, c, o;
    int n, combined;
    double h_scalar, c_scalar;
    double max_h_err, max_c_err;
    double tol;

    printf("[TEST] Deterministic forward: 128 cells, 256 steps, 12 features");
    fflush(stdout);

    n = DET2_N;
    combined = n + DET2_F;
    tol = 1e-10;

    net = lstm_init(n, DET2_F, TEST_NUM_OUT, 42);
    set_constant_weights(net, DET2_W);

    sequence = (double *)utils_safe_calloc((size_t)DET2_T * DET2_F, sizeof(double));
    for (t = 0; t < DET2_T; t++) {
        for (i = 0; i < DET2_F; i++) {
            sequence[t * DET2_F + i] = DET2_X;
        }
    }

    cell_states = alloc_cell_states(DET2_T, n, combined);
    preds = (double *)utils_safe_calloc((size_t)(TEST_NUM_OUT * DET2_T), sizeof(double));
    lstm_forward(net, sequence, cell_states, DET2_T, preds);

    h_scalar = 0.0;
    c_scalar = 0.0;
    max_h_err = 0.0;
    max_c_err = 0.0;

    for (t = 0; t < DET2_T; t++) {
        double concat_sum = (double)n * h_scalar + (double)DET2_F * DET2_X;
        double gate_pre = DET2_W * concat_sum;
        double f_gate = utils_sigmoid(gate_pre);
        double i_gate = utils_sigmoid(gate_pre);
        double g_gate = utils_tanh_act(gate_pre);
        double o_gate = utils_sigmoid(gate_pre);
        double c_new = f_gate * c_scalar + i_gate * g_gate;
        double h_new = o_gate * utils_tanh_act(c_new);

        for (c = 0; c < n; c++) {
            double h_err = fabs(cell_states[t].hidden_state[c] - h_new);
            double c_err = fabs(cell_states[t].cell_state[c] - c_new);
            if (h_err > max_h_err) max_h_err = h_err;
            if (c_err > max_c_err) max_c_err = c_err;

            if (fabs(cell_states[t].forget_gate[c] - f_gate) > tol) {
                printf("\n  FAIL: forget_gate[%d][%d] mismatch\n", t, c);
                free(sequence);
                free(preds);
                lstm_free_cell_states(cell_states, DET2_T);
                lstm_free(net);
                return 0;
            }
            if (fabs(cell_states[t].input_gate[c] - i_gate) > tol) {
                printf("\n  FAIL: input_gate[%d][%d] mismatch\n", t, c);
                free(sequence);
                free(preds);
                lstm_free_cell_states(cell_states, DET2_T);
                lstm_free(net);
                return 0;
            }
            if (fabs(cell_states[t].output_gate[c] - o_gate) > tol) {
                printf("\n  FAIL: output_gate[%d][%d] mismatch\n", t, c);
                free(sequence);
                free(preds);
                lstm_free_cell_states(cell_states, DET2_T);
                lstm_free(net);
                return 0;
            }
            if (fabs(cell_states[t].candidate[c] - g_gate) > tol) {
                printf("\n  FAIL: candidate[%d][%d] mismatch\n", t, c);
                free(sequence);
                free(preds);
                lstm_free_cell_states(cell_states, DET2_T);
                lstm_free(net);
                return 0;
            }
        }

        /* Check outputs at this timestep */
        {
            double expected_pred = DET2_W * (double)n * h_new;
            for (o = 0; o < TEST_NUM_OUT; o++) {
                if (fabs(preds[t * TEST_NUM_OUT + o] - expected_pred) > 1e-8) {
                    printf("\n  FAIL: pred[%d][%d]=%.10f expected=%.10f\n", t, o, preds[t * TEST_NUM_OUT + o], expected_pred);
                    free(sequence);
                    free(preds);
                    lstm_free_cell_states(cell_states, DET2_T);
                    lstm_free(net);
                    return 0;
                }
            }
        }

        h_scalar = h_new;
        c_scalar = c_new;
    }

    printf("\n  max_h_err=%.2e  max_c_err=%.2e  pred[0][0]=%.8f", max_h_err, max_c_err, preds[0]);

    if (max_h_err < tol && max_c_err < tol) {
        printf("\n  PASS: H and C vectors match analytical solution at all %d steps\n", DET2_T);
        free(sequence);
        free(preds);
        lstm_free_cell_states(cell_states, DET2_T);
        lstm_free(net);
        return 1;
    } else {
        printf("\n  FAIL: H or C error exceeds tolerance %.2e\n", tol);
        free(sequence);
        free(preds);
        lstm_free_cell_states(cell_states, DET2_T);
        lstm_free(net);
        return 0;
    }
}
