#include "test_utils.h"
#include <stdio.h>
#include <math.h>

/* ==================== Test: Weight Update ==================== */

int test_update(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double preds[5 * TEST_NUM_OUT];
    double targets[5 * TEST_NUM_OUT];
    double old_val;
    int i, t, o;
    int n, combined;

    printf("[TEST] Weight update...");
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

    old_val = net->weight_forget_recurrent->data[0];

    lstm_update(net, 0.01, 0.9, 1);

    if (net->weight_forget_recurrent->data[0] == old_val) {
        printf("  FAIL: weights did not change!\n");
        lstm_free_cell_states(cell_states, 5);
        lstm_free(net);
        return 0;
    }

    printf("  PASS: weights changed (%.6f -> %.6f)\n",
           old_val, net->weight_forget_recurrent->data[0]);
    lstm_free_cell_states(cell_states, 5);
    lstm_free(net);
    return 1;
}

/* ==================== Test: Batch Gradient Normalization (Bug #6) ==================== */

int test_batch_normalization(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double preds[5 * TEST_NUM_OUT];
    double targets[5 * TEST_NUM_OUT];
    int i, t, o;
    int n, combined;
    double saved_weight;
    double saved_grad;
    double weight_after_1;
    double weight_after_4;

    printf("[TEST] Batch gradient normalization (Bug #6)...");
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

    saved_weight = net->weight_forget_recurrent->data[0];
    saved_grad = net->dweight_forget_recurrent->data[0];

    net->mweight_forget_recurrent->data[0] = 0.0;
    lstm_update(net, 0.01, 0.0, 1);
    weight_after_1 = net->weight_forget_recurrent->data[0];

    net->weight_forget_recurrent->data[0] = saved_weight;
    net->mweight_forget_recurrent->data[0] = 0.0;
    net->dweight_forget_recurrent->data[0] = saved_grad * 4.0;

    lstm_update(net, 0.01, 0.0, 4);
    weight_after_4 = net->weight_forget_recurrent->data[0];

    if (fabs(weight_after_1 - weight_after_4) > 1e-6) {
        printf("\n  FAIL: batch=1 -> %.8f, batch=4 -> %.8f\n",
               weight_after_1, weight_after_4);
        lstm_free_cell_states(cell_states, 5);
        lstm_free(net);
        return 0;
    }

    printf("  PASS: batch=1 and batch=4 produce same update (%.8f)\n", weight_after_1);
    lstm_free_cell_states(cell_states, 5);
    lstm_free(net);
    return 1;
}
