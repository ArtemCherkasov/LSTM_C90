#include "../test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ==================== Test: Sigmoid gradient check ==================== */
/* Verifies BPTT gradients with sigmoid output activation */

#define SG_N 16
#define SG_F 6
#define SG_T 5
#define SG_W 0.05
#define SG_NUM_OUT 3

int test_sigmoid_gradient_check(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double *sequence;
    double *preds;
    double *targets;
    double eps;
    int n, combined;
    int t, i, c, o;
    int num_checks = 6;
    int num_passed;
    double max_rel_err;
    double rel_tol;

    int check_gate[6];
    int check_idx[6];
    double anal_grad[6];
    double num_grad[6];
    double *weight_ptr[6];

    printf("[TEST] Sigmoid gradient check: %d cells, %d steps, %d outputs", SG_N, SG_T, SG_NUM_OUT);
    fflush(stdout);

    n = SG_N;
    combined = n + SG_F;
    eps = 1e-5;
    rel_tol = 1e-3;

    net = lstm_init(n, SG_F, SG_NUM_OUT, 42);
    net->output_activation = ACTIVATION_SIGMOID;
    set_constant_weights(net, SG_W);

    sequence = (double *)utils_safe_calloc((size_t)SG_T * SG_F, sizeof(double));
    for (t = 0; t < SG_T; t++)
        for (i = 0; i < SG_F; i++)
            sequence[t * SG_F + i] = 0.5;

    preds = (double *)utils_safe_calloc((size_t)(SG_NUM_OUT * SG_T), sizeof(double));
    targets = (double *)utils_safe_calloc((size_t)(SG_NUM_OUT * SG_T), sizeof(double));
    for (t = 0; t < SG_T; t++)
        for (o = 0; o < SG_NUM_OUT; o++)
            targets[t * SG_NUM_OUT + o] = (o == 0) ? 1.0 : 0.0;

    cell_states = alloc_cell_states(SG_T, n, combined);

    /* Forward + Backward */
    lstm_forward(net, sequence, cell_states, SG_T, preds);
    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, preds, targets, SG_T);

    /* Select weights to check */
    check_gate[0] = 0; check_idx[0] = 0;
    weight_ptr[0] = &net->weight_forget_recurrent->data[0];
    anal_grad[0] = net->dweight_forget_recurrent->data[0];

    check_gate[1] = 1; check_idx[1] = 5 * combined + 10;
    weight_ptr[1] = &net->weight_input_recurrent->data[5 * combined + 10];
    anal_grad[1] = net->dweight_input_recurrent->data[5 * combined + 10];

    check_gate[2] = 2; check_idx[2] = 3 * combined + 5;
    weight_ptr[2] = &net->weight_output_recurrent->data[3 * combined + 5];
    anal_grad[2] = net->dweight_output_recurrent->data[3 * combined + 5];

    check_gate[3] = 3; check_idx[3] = 10 * combined + 15;
    weight_ptr[3] = &net->weight_candidate_recurrent->data[10 * combined + 15];
    anal_grad[3] = net->dweight_candidate_recurrent->data[10 * combined + 15];

    check_gate[4] = 4; check_idx[4] = 0 * n + 0;
    weight_ptr[4] = &net->weight_output_layer->data[0 * n + 0];
    anal_grad[4] = net->dweight_output_layer->data[0 * n + 0];

    check_gate[5] = 4; check_idx[5] = 2 * n + 8;
    weight_ptr[5] = &net->weight_output_layer->data[2 * n + 8];
    anal_grad[5] = net->dweight_output_layer->data[2 * n + 8];

    /* Numerical gradients */
    for (i = 0; i < num_checks; i++) {
        double orig_w, loss_plus, loss_minus;
        orig_w = *weight_ptr[i];

        *weight_ptr[i] = orig_w + eps;
        lstm_forward(net, sequence, cell_states, SG_T, preds);
        loss_plus = 0.0;
        for (t = 0; t < SG_T; t++)
            for (o = 0; o < SG_NUM_OUT; o++)
                loss_plus += 0.5 * (preds[t*SG_NUM_OUT+o] - targets[t*SG_NUM_OUT+o]) * (preds[t*SG_NUM_OUT+o] - targets[t*SG_NUM_OUT+o]);

        *weight_ptr[i] = orig_w - eps;
        lstm_forward(net, sequence, cell_states, SG_T, preds);
        loss_minus = 0.0;
        for (t = 0; t < SG_T; t++)
            for (o = 0; o < SG_NUM_OUT; o++)
                loss_minus += 0.5 * (preds[t*SG_NUM_OUT+o] - targets[t*SG_NUM_OUT+o]) * (preds[t*SG_NUM_OUT+o] - targets[t*SG_NUM_OUT+o]);

        *weight_ptr[i] = orig_w;
        num_grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
    }

    /* Compare */
    num_passed = 0;
    max_rel_err = 0.0;

    printf("\n  %-6s %-8s %-14s %-14s %-10s", "gate", "idx", "analytical", "numerical", "rel_err");
    for (i = 0; i < num_checks; i++) {
        double diff, rel_err;
        char gate_name[8];
        diff = fabs(anal_grad[i] - num_grad[i]);
        if (fabs(num_grad[i]) > 1e-12) rel_err = diff / fabs(num_grad[i]);
        else rel_err = diff;
        if (rel_err > max_rel_err) max_rel_err = rel_err;

        switch (check_gate[i]) {
            case 0: strcpy(gate_name, "f_rec"); break;
            case 1: strcpy(gate_name, "i_rec"); break;
            case 2: strcpy(gate_name, "o_rec"); break;
            case 3: strcpy(gate_name, "c_rec"); break;
            case 4: strcpy(gate_name, "w_out"); break;
            default: strcpy(gate_name, "?"); break;
        }
        printf("\n  %-6s %-8d %-14.8e %-14.8e %-10.2e", gate_name, check_idx[i], anal_grad[i], num_grad[i], rel_err);
        if (rel_err < rel_tol || diff < 1e-10) num_passed++;
    }

    printf("\n  passed %d/%d, max_rel_err=%.2e (tol=%.2e)", num_passed, num_checks, max_rel_err, rel_tol);

    free(sequence);
    free(preds);
    free(targets);
    lstm_free_cell_states(cell_states, SG_T);
    lstm_free(net);

    if (num_passed == num_checks) {
        printf("\n  PASS: all sigmoid gradient checks passed\n");
        return 1;
    } else {
        printf("\n  FAIL: %d sigmoid gradient checks failed\n", num_checks - num_passed);
        return 0;
    }
}
