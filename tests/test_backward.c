#include "test_utils.h"
#include <stdio.h>
#include <math.h>

/* ==================== Test: Backward Pass — Non-zero Gradients ==================== */

int test_backward_pass(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double preds[5 * TEST_NUM_OUT];
    double targets[5 * TEST_NUM_OUT];
    int i, t, r, c, o;
    int n, combined;
    int has_grad;

    printf("[TEST] Backward pass (gradient check)...");
    fflush(stdout);

    net = lstm_init(8, 3, TEST_NUM_OUT, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.5;
        }
    }
    for (t = 0; t < 5; t++)
        for (o = 0; o < TEST_NUM_OUT; o++)
            targets[t * TEST_NUM_OUT + o] = 1.0;

    cell_states = alloc_cell_states(5, n, combined);
    lstm_forward(net, sequence, cell_states, 5, preds);

    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, preds, targets, 5);

    has_grad = 0;
    for (c = 0; c < n; c++) {
        for (r = 0; r < combined; r++) {
            if (fabs(net->dweight_forget_recurrent->data[c * combined + r]) > 1e-15) {
                has_grad = 1;
                break;
            }
        }
        if (has_grad) break;
    }

    if (!has_grad) {
        printf("  FAIL: all forget gate gradients are zero!\n");
        lstm_free_cell_states(cell_states, 5);
        lstm_free(net);
        return 0;
    }

    printf("  PASS: gradients are non-zero\n");
    lstm_free_cell_states(cell_states, 5);
    lstm_free(net);
    return 1;
}

/* ==================== Test: Last Timestep Gradients (Bug #2) ==================== */

int test_last_timestep_grad(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double preds[5 * TEST_NUM_OUT];
    double targets[5 * TEST_NUM_OUT];
    int i, t, c, o;
    int n, combined;
    int has_input_grad;
    int has_forget_grad;
    int has_candidate_grad;

    printf("[TEST] Last timestep gradients (Bug #2)...");
    fflush(stdout);

    net = lstm_init(8, 3, TEST_NUM_OUT, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.3 * (double)(t + 1);
        }
    }
    for (t = 0; t < 5; t++)
        for (o = 0; o < TEST_NUM_OUT; o++)
            targets[t * TEST_NUM_OUT + o] = 0.5;

    cell_states = alloc_cell_states(5, n, combined);
    lstm_forward(net, sequence, cell_states, 5, preds);

    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, preds, targets, 5);

    has_forget_grad = 0;
    has_input_grad = 0;
    has_candidate_grad = 0;

    for (c = 0; c < n; c++) {
        if (fabs(net->dbias_forget[c]) > 1e-15) has_forget_grad = 1;
        if (fabs(net->dbias_input[c]) > 1e-15) has_input_grad = 1;
        if (fabs(net->dbias_candidate[c]) > 1e-15) has_candidate_grad = 1;
    }

    if (!has_forget_grad || !has_input_grad || !has_candidate_grad) {
        printf("\n  FAIL: forget=%d input=%d candidate=%d (all should be 1)\n",
               has_forget_grad, has_input_grad, has_candidate_grad);
        lstm_free_cell_states(cell_states, 5);
        lstm_free(net);
        return 0;
    }

    printf("  PASS: all gate biases have non-zero gradients\n");
    lstm_free_cell_states(cell_states, 5);
    lstm_free(net);
    return 1;
}

/* ==================== Test: Output Layer Gradient Accumulation (Bug #3) ==================== */

int test_output_grad_accumulation(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double preds[5 * TEST_NUM_OUT];
    double targets[5 * TEST_NUM_OUT];
    int i, t, o;
    int n, combined;
    double grad_single;
    double grad_double;

    printf("[TEST] Output layer gradient accumulation (Bug #3)...");
    fflush(stdout);

    net = lstm_init(8, 3, TEST_NUM_OUT, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.5;
        }
    }
    for (t = 0; t < 5; t++)
        for (o = 0; o < TEST_NUM_OUT; o++)
            targets[t * TEST_NUM_OUT + o] = 1.0;

    cell_states = alloc_cell_states(5, n, combined);
    lstm_forward(net, sequence, cell_states, 5, preds);

    /* Single backward call */
    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, preds, targets, 5);
    grad_single = net->dbias_output_layer[0];

    /* Two backward calls (same sample) — should accumulate */
    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, preds, targets, 5);
    lstm_backward(net, sequence, cell_states, preds, targets, 5);
    grad_double = net->dbias_output_layer[0];

    if (fabs(grad_double - 2.0 * grad_single) > 1e-10) {
        printf("\n  FAIL: grad_single=%.6f, grad_double=%.6f (expected %.6f)\n",
               grad_single, grad_double, 2.0 * grad_single);
        lstm_free_cell_states(cell_states, 5);
        lstm_free(net);
        return 0;
    }

    printf("  PASS: gradients accumulate correctly (%.6f + %.6f = %.6f)\n",
           grad_single, grad_single, grad_double);
    lstm_free_cell_states(cell_states, 5);
    lstm_free(net);
    return 1;
}
