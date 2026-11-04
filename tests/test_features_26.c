#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ==================== Test: Forward Pass with 26 Features ==================== */

int test_forward_26_features(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double *sequence;
    double *preds;
    int i, t, o;
    int n, combined;
    int seq_len = 5;
    int num_features = 26;
    int bad = 0;

    printf("[TEST] Forward pass with 26 features...");
    fflush(stdout);

    net = lstm_init(8, num_features, TEST_NUM_OUT, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    sequence = (double *)utils_safe_calloc((size_t)(num_features * seq_len), sizeof(double));
    preds = (double *)utils_safe_calloc((size_t)(TEST_NUM_OUT * seq_len), sizeof(double));

    for (t = 0; t < seq_len; t++) {
        for (i = 0; i < num_features; i++) {
            sequence[t * num_features + i] = 0.5;
        }
    }

    cell_states = alloc_cell_states(seq_len, n, combined);
    lstm_forward(net, sequence, cell_states, seq_len, preds);

    for (t = 0; t < seq_len; t++) {
        for (o = 0; o < TEST_NUM_OUT; o++) {
            double p = preds[t * TEST_NUM_OUT + o];
            if (p != p || fabs(p) > 1e6) bad = 1;
        }
    }

    if (bad) {
        printf("\n  FAIL: prediction is NaN or Inf with 26 features!\n");
        free(sequence);
        free(preds);
        lstm_free_cell_states(cell_states, seq_len);
        lstm_free(net);
        return 0;
    }

    /* Verify hidden states are non-trivial */
    {
        double h_sum = 0.0;
        for (i = 0; i < n; i++)
            h_sum += fabs(cell_states[seq_len - 1].hidden_state[i]);
        if (h_sum < 1e-15) {
            printf("\n  FAIL: hidden states are all zero!\n");
            free(sequence);
            free(preds);
            lstm_free_cell_states(cell_states, seq_len);
            lstm_free(net);
            return 0;
        }
    }

    printf("\n  PASS: forward pass with 26 features OK (pred[0][0]=%.6f)\n", preds[0]);
    free(sequence);
    free(preds);
    lstm_free_cell_states(cell_states, seq_len);
    lstm_free(net);
    return 1;
}

/* ==================== Test: Backward Pass with 26 Features ==================== */

int test_backward_26_features(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double *sequence;
    double *preds;
    double *targets;
    int i, t, r, c, o;
    int n, combined;
    int seq_len = 5;
    int num_features = 26;
    int has_grad;

    printf("[TEST] Backward pass with 26 features (gradient check)...");
    fflush(stdout);

    net = lstm_init(8, num_features, TEST_NUM_OUT, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    sequence = (double *)utils_safe_calloc((size_t)(num_features * seq_len), sizeof(double));
    preds = (double *)utils_safe_calloc((size_t)(TEST_NUM_OUT * seq_len), sizeof(double));
    targets = (double *)utils_safe_calloc((size_t)(TEST_NUM_OUT * seq_len), sizeof(double));

    for (t = 0; t < seq_len; t++) {
        for (i = 0; i < num_features; i++) {
            sequence[t * num_features + i] = 0.5;
        }
    }
    for (t = 0; t < seq_len; t++)
        for (o = 0; o < TEST_NUM_OUT; o++)
            targets[t * TEST_NUM_OUT + o] = 1.0;

    cell_states = alloc_cell_states(seq_len, n, combined);
    lstm_forward(net, sequence, cell_states, seq_len, preds);

    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, preds, targets, seq_len);

    /* Check forget gate recurrent gradients */
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
        printf("\n  FAIL: all forget gate gradients are zero with 26 features!\n");
        free(sequence);
        free(preds);
        free(targets);
        lstm_free_cell_states(cell_states, seq_len);
        lstm_free(net);
        return 0;
    }

    /* Check input weight gradients in the recurrent (concatenated) matrix */
    /* The input portion starts at column n in the [n x (n+f)] matrix */
    has_grad = 0;
    for (c = 0; c < n; c++) {
        for (r = n; r < combined; r++) {
            if (fabs(net->dweight_forget_recurrent->data[c * combined + r]) > 1e-15) {
                has_grad = 1;
                break;
            }
        }
        if (has_grad) break;
    }

    if (!has_grad) {
        printf("\n  FAIL: all forget input gradients are zero with 26 features!\n");
        free(sequence);
        free(preds);
        free(targets);
        lstm_free_cell_states(cell_states, seq_len);
        lstm_free(net);
        return 0;
    }

    printf("\n  PASS: gradients are non-zero with 26 features\n");
    free(sequence);
    free(preds);
    free(targets);
    lstm_free_cell_states(cell_states, seq_len);
    lstm_free(net);
    return 1;
}
