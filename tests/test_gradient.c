#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ==================== Test: BPTT Numerical Gradient Check 64x5x6 ==================== */

#define GRAD_N 64
#define GRAD_F 6
#define GRAD_T 5
#define GRAD_W 0.01

int test_bptt_gradient_check_64(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double *sequence;
    double *preds;
    double *targets;
    double eps;
    int n, combined;
    int t, i, c, o;
    int num_checks;
    int num_passed;
    double max_rel_err;
    double rel_tol;

    int check_gate[8];
    int check_idx[8];
    double anal_grad[8];
    double num_grad[8];
    double *weight_ptr[8];

    printf("[TEST] BPTT numerical gradient check: 64 cells, %d steps, %d outputs", GRAD_T, TEST_NUM_OUT);
    fflush(stdout);

    n = GRAD_N;
    combined = n + GRAD_F;
    eps = 1e-5;
    rel_tol = 1e-3;

    net = lstm_init(n, GRAD_F, TEST_NUM_OUT, 42);
    set_constant_weights(net, GRAD_W);

    sequence = (double *)utils_safe_calloc((size_t)GRAD_T * GRAD_F, sizeof(double));
    for (t = 0; t < GRAD_T; t++) {
        for (i = 0; i < GRAD_F; i++) {
            sequence[t * GRAD_F + i] = DET_X;
        }
    }

    preds = (double *)utils_safe_calloc((size_t)(TEST_NUM_OUT * GRAD_T), sizeof(double));
    targets = (double *)utils_safe_calloc((size_t)(TEST_NUM_OUT * GRAD_T), sizeof(double));
    for (t = 0; t < GRAD_T; t++)
        for (o = 0; o < TEST_NUM_OUT; o++)
            targets[t * TEST_NUM_OUT + o] = 0.1;

    cell_states = alloc_cell_states(GRAD_T, n, combined);

    /* Step 1: Forward + Backward to get analytical gradients */
    lstm_forward(net, sequence, cell_states, GRAD_T, preds);
    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, preds, targets, GRAD_T);

    /* Select 8 weights to check */
    num_checks = 8;

    check_gate[0] = 0; check_idx[0] = 0;
    weight_ptr[0] = &net->weight_forget_recurrent->data[0];
    anal_grad[0] = net->dweight_forget_recurrent->data[0];

    check_gate[1] = 0; check_idx[1] = 10 * combined + 35;
    weight_ptr[1] = &net->weight_forget_recurrent->data[10 * combined + 35];
    anal_grad[1] = net->dweight_forget_recurrent->data[10 * combined + 35];

    check_gate[2] = 1; check_idx[2] = 5 * combined + 60;
    weight_ptr[2] = &net->weight_input_recurrent->data[5 * combined + 60];
    anal_grad[2] = net->dweight_input_recurrent->data[5 * combined + 60];

    check_gate[3] = 2; check_idx[3] = 20 * combined + 10;
    weight_ptr[3] = &net->weight_output_recurrent->data[20 * combined + 10];
    anal_grad[3] = net->dweight_output_recurrent->data[20 * combined + 10];

    check_gate[4] = 3; check_idx[4] = 30 * combined + 50;
    weight_ptr[4] = &net->weight_candidate_recurrent->data[30 * combined + 50];
    anal_grad[4] = net->dweight_candidate_recurrent->data[30 * combined + 50];

    /* Output layer weights: [TEST_NUM_OUT x n] */
    check_gate[5] = 4; check_idx[5] = 0 * n + 0;
    weight_ptr[5] = &net->weight_output_layer->data[0 * n + 0];
    anal_grad[5] = net->dweight_output_layer->data[0 * n + 0];

    check_gate[6] = 4; check_idx[6] = 1 * n + 32;
    weight_ptr[6] = &net->weight_output_layer->data[1 * n + 32];
    anal_grad[6] = net->dweight_output_layer->data[1 * n + 32];

    check_gate[7] = 0; check_idx[7] = 63 * combined + 69;
    weight_ptr[7] = &net->weight_forget_recurrent->data[63 * combined + 69];
    anal_grad[7] = net->dweight_forget_recurrent->data[63 * combined + 69];

    /* Step 2: Numerical gradients via finite differences */
    for (i = 0; i < num_checks; i++) {
        double orig_w, loss_plus, loss_minus;

        orig_w = *weight_ptr[i];

        *weight_ptr[i] = orig_w + eps;
        lstm_forward(net, sequence, cell_states, GRAD_T, preds);
        loss_plus = 0.0;
        for (t = 0; t < GRAD_T; t++)
            for (o = 0; o < TEST_NUM_OUT; o++)
                loss_plus += 0.5 * (preds[t*TEST_NUM_OUT+o] - targets[t*TEST_NUM_OUT+o]) * (preds[t*TEST_NUM_OUT+o] - targets[t*TEST_NUM_OUT+o]);

        *weight_ptr[i] = orig_w - eps;
        lstm_forward(net, sequence, cell_states, GRAD_T, preds);
        loss_minus = 0.0;
        for (t = 0; t < GRAD_T; t++)
            for (o = 0; o < TEST_NUM_OUT; o++)
                loss_minus += 0.5 * (preds[t*TEST_NUM_OUT+o] - targets[t*TEST_NUM_OUT+o]) * (preds[t*TEST_NUM_OUT+o] - targets[t*TEST_NUM_OUT+o]);

        *weight_ptr[i] = orig_w;
        num_grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
    }

    /* Step 3: Compare */
    num_passed = 0;
    max_rel_err = 0.0;

    printf("\n  %-6s %-8s %-14s %-14s %-10s", "gate", "idx", "analytical", "numerical", "rel_err");
    for (i = 0; i < num_checks; i++) {
        double diff, rel_err;
        char gate_name[8];

        diff = fabs(anal_grad[i] - num_grad[i]);
        if (fabs(num_grad[i]) > 1e-12) {
            rel_err = diff / fabs(num_grad[i]);
        } else {
            rel_err = diff;
        }

        if (rel_err > max_rel_err) max_rel_err = rel_err;

        switch (check_gate[i]) {
            case 0: strcpy(gate_name, "f_rec"); break;
            case 1: strcpy(gate_name, "i_rec"); break;
            case 2: strcpy(gate_name, "o_rec"); break;
            case 3: strcpy(gate_name, "c_rec"); break;
            case 4: strcpy(gate_name, "w_out"); break;
            default: strcpy(gate_name, "?????"); break;
        }

        printf("\n  %-6s %-8d %-14.8e %-14.8e %-10.2e",
               gate_name, check_idx[i], anal_grad[i], num_grad[i], rel_err);

        if (rel_err < rel_tol || diff < 1e-10) {
            num_passed++;
        }
    }

    printf("\n  passed %d/%d checks, max_rel_err=%.2e (tol=%.2e)",
           num_passed, num_checks, max_rel_err, rel_tol);

    free(sequence);
    free(preds);
    free(targets);
    lstm_free_cell_states(cell_states, GRAD_T);
    lstm_free(net);

    if (num_passed == num_checks) {
        printf("\n  PASS: all gradient checks passed\n");
        return 1;
    } else {
        printf("\n  FAIL: %d gradient checks failed\n", num_checks - num_passed);
        return 0;
    }
}

/* ==================== Test: BPTT Numerical Gradient Check 128x4x12 ==================== */

#define GRAD2_N 128
#define GRAD2_F 12
#define GRAD2_T 4
#define GRAD2_W 0.02

int test_bptt_gradient_check_128(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double *sequence;
    double *preds;
    double *targets;
    double eps;
    int n, combined;
    int t, i, c, o;
    int num_checks;
    int num_passed;
    double max_rel_err;
    double rel_tol;

    int check_gate[10];
    int check_idx[10];
    double anal_grad[10];
    double num_grad[10];
    double *weight_ptr[10];

    printf("[TEST] BPTT numerical gradient check: 128 cells, %d steps, 12 features, %d outputs",
           GRAD2_T, TEST_NUM_OUT);
    fflush(stdout);

    n = GRAD2_N;
    combined = n + GRAD2_F;
    eps = 1e-4;
    rel_tol = 1e-3;

    net = lstm_init(n, GRAD2_F, TEST_NUM_OUT, 42);
    set_constant_weights(net, GRAD2_W);

    sequence = (double *)utils_safe_calloc((size_t)GRAD2_T * GRAD2_F, sizeof(double));
    for (t = 0; t < GRAD2_T; t++) {
        for (i = 0; i < GRAD2_F; i++) {
            sequence[t * GRAD2_F + i] = DET2_X;
        }
    }

    preds = (double *)utils_safe_calloc((size_t)(TEST_NUM_OUT * GRAD2_T), sizeof(double));
    targets = (double *)utils_safe_calloc((size_t)(TEST_NUM_OUT * GRAD2_T), sizeof(double));
    for (t = 0; t < GRAD2_T; t++)
        for (o = 0; o < TEST_NUM_OUT; o++)
            targets[t * TEST_NUM_OUT + o] = 1.0;

    cell_states = alloc_cell_states(GRAD2_T, n, combined);

    /* Step 1: Forward + Backward */
    lstm_forward(net, sequence, cell_states, GRAD2_T, preds);
    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, preds, targets, GRAD2_T);

    num_checks = 10;

    check_gate[0] = 0; check_idx[0] = 0;
    weight_ptr[0] = &net->weight_forget_recurrent->data[0];
    anal_grad[0] = net->dweight_forget_recurrent->data[0];

    check_gate[1] = 0; check_idx[1] = 20 * combined + 70;
    weight_ptr[1] = &net->weight_forget_recurrent->data[20 * combined + 70];
    anal_grad[1] = net->dweight_forget_recurrent->data[20 * combined + 70];

    check_gate[2] = 1; check_idx[2] = 10 * combined + 120;
    weight_ptr[2] = &net->weight_input_recurrent->data[10 * combined + 120];
    anal_grad[2] = net->dweight_input_recurrent->data[10 * combined + 120];

    check_gate[3] = 2; check_idx[3] = 40 * combined + 20;
    weight_ptr[3] = &net->weight_output_recurrent->data[40 * combined + 20];
    anal_grad[3] = net->dweight_output_recurrent->data[40 * combined + 20];

    check_gate[4] = 3; check_idx[4] = 60 * combined + 100;
    weight_ptr[4] = &net->weight_candidate_recurrent->data[60 * combined + 100];
    anal_grad[4] = net->dweight_candidate_recurrent->data[60 * combined + 100];

    /* Output layer: [TEST_NUM_OUT x n] */
    check_gate[5] = 4; check_idx[5] = 0 * n + 0;
    weight_ptr[5] = &net->weight_output_layer->data[0 * n + 0];
    anal_grad[5] = net->dweight_output_layer->data[0 * n + 0];

    check_gate[6] = 4; check_idx[6] = 1 * n + 64;
    weight_ptr[6] = &net->weight_output_layer->data[1 * n + 64];
    anal_grad[6] = net->dweight_output_layer->data[1 * n + 64];

    check_gate[7] = 4; check_idx[7] = 3 * n + 127;
    weight_ptr[7] = &net->weight_output_layer->data[3 * n + 127];
    anal_grad[7] = net->dweight_output_layer->data[3 * n + 127];

    check_gate[8] = 0; check_idx[8] = 100 * combined + 130;
    weight_ptr[8] = &net->weight_forget_recurrent->data[100 * combined + 130];
    anal_grad[8] = net->dweight_forget_recurrent->data[100 * combined + 130];

    check_gate[9] = 3; check_idx[9] = 127 * combined + (combined - 1);
    weight_ptr[9] = &net->weight_candidate_recurrent->data[127 * combined + (combined - 1)];
    anal_grad[9] = net->dweight_candidate_recurrent->data[127 * combined + (combined - 1)];

    /* Step 2: Numerical gradients */
    for (i = 0; i < num_checks; i++) {
        double orig_w, loss_plus, loss_minus;

        orig_w = *weight_ptr[i];

        *weight_ptr[i] = orig_w + eps;
        lstm_forward(net, sequence, cell_states, GRAD2_T, preds);
        loss_plus = 0.0;
        for (t = 0; t < GRAD2_T; t++)
            for (o = 0; o < TEST_NUM_OUT; o++)
                loss_plus += 0.5 * (preds[t*TEST_NUM_OUT+o] - targets[t*TEST_NUM_OUT+o]) * (preds[t*TEST_NUM_OUT+o] - targets[t*TEST_NUM_OUT+o]);

        *weight_ptr[i] = orig_w - eps;
        lstm_forward(net, sequence, cell_states, GRAD2_T, preds);
        loss_minus = 0.0;
        for (t = 0; t < GRAD2_T; t++)
            for (o = 0; o < TEST_NUM_OUT; o++)
                loss_minus += 0.5 * (preds[t*TEST_NUM_OUT+o] - targets[t*TEST_NUM_OUT+o]) * (preds[t*TEST_NUM_OUT+o] - targets[t*TEST_NUM_OUT+o]);

        *weight_ptr[i] = orig_w;
        num_grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
    }

    /* Step 3: Compare */
    num_passed = 0;
    max_rel_err = 0.0;

    printf("\n  %-6s %-8s %-14s %-14s %-10s", "gate", "idx", "analytical", "numerical", "rel_err");
    for (i = 0; i < num_checks; i++) {
        double diff, rel_err;
        char gate_name[8];

        diff = fabs(anal_grad[i] - num_grad[i]);
        if (fabs(num_grad[i]) > 1e-12) {
            rel_err = diff / fabs(num_grad[i]);
        } else {
            rel_err = diff;
        }

        if (rel_err > max_rel_err) max_rel_err = rel_err;

        switch (check_gate[i]) {
            case 0: strcpy(gate_name, "f_rec"); break;
            case 1: strcpy(gate_name, "i_rec"); break;
            case 2: strcpy(gate_name, "o_rec"); break;
            case 3: strcpy(gate_name, "c_rec"); break;
            case 4: strcpy(gate_name, "w_out"); break;
            default: strcpy(gate_name, "?????"); break;
        }

        printf("\n  %-6s %-8d %-14.8e %-14.8e %-10.2e",
               gate_name, check_idx[i], anal_grad[i], num_grad[i], rel_err);

        if (rel_err < rel_tol || diff < 1e-10) {
            num_passed++;
        }
    }

    printf("\n  passed %d/%d checks, max_rel_err=%.2e (tol=%.2e)",
           num_passed, num_checks, max_rel_err, rel_tol);

    free(sequence);
    free(preds);
    free(targets);
    lstm_free_cell_states(cell_states, GRAD2_T);
    lstm_free(net);

    if (num_passed == num_checks) {
        printf("\n  PASS: all gradient checks passed\n");
        return 1;
    } else {
        printf("\n  FAIL: %d gradient checks failed\n", num_checks - num_passed);
        return 0;
    }
}
