#include "test_utils.h"
#include <stdio.h>
#include <math.h>

/* ==================== Test: Forward Pass ==================== */

int test_forward_pass(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double preds[5 * TEST_NUM_OUT];
    int i, t, o;
    int n, combined;
    int bad = 0;

    printf("[TEST] Forward pass...");
    fflush(stdout);

    net = lstm_init(8, 3, TEST_NUM_OUT, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.5;
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    lstm_forward(net, sequence, cell_states, 5, preds);

    for (t = 0; t < 5; t++) {
        for (o = 0; o < TEST_NUM_OUT; o++) {
            printf("  pred[%d][%d]=%.6f", t, o, preds[t * TEST_NUM_OUT + o]);
            if (preds[t * TEST_NUM_OUT + o] != preds[t * TEST_NUM_OUT + o] || fabs(preds[t * TEST_NUM_OUT + o]) > 1e6) bad = 1;
        }
    }
    printf("\n");

    if (bad) {
        printf("  FAIL: prediction is NaN or Inf!\n");
        lstm_free_cell_states(cell_states, 5);
        lstm_free(net);
        return 0;
    }

    for (t = 0; t < 5; t++) {
        if (!cell_states[t].hidden_state || !cell_states[t].cell_state) {
            printf("  FAIL: cell_states[%d] has NULL pointers!\n", t);
            lstm_free_cell_states(cell_states, 5);
            lstm_free(net);
            return 0;
        }
    }

    printf("  PASS: forward pass OK\n");
    lstm_free_cell_states(cell_states, 5);
    lstm_free(net);
    return 1;
}

/* ==================== Test: Concatenated Storage (Bug #1) ==================== */

int test_concatenated_storage(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double preds[5 * TEST_NUM_OUT];
    int i, t, c;
    int n, combined;
    int all_stored;

    printf("[TEST] Concatenated storage (Bug #1)...");
    fflush(stdout);

    net = lstm_init(8, 3, TEST_NUM_OUT, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.1 * (double)(t * 3 + i + 1);
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    lstm_forward(net, sequence, cell_states, 5, preds);

    all_stored = 1;
    for (t = 0; t < 5; t++) {
        for (c = 0; c < net->num_features; c++) {
            double expected = sequence[t * net->num_features + c];
            double stored = cell_states[t].concatenated[n + c];
            if (fabs(expected - stored) > 1e-10) {
                printf("\n  FAIL: concatenated[%d][%d] = %.6f, expected %.6f\n",
                       t, n + c, stored, expected);
                all_stored = 0;
                break;
            }
        }
        if (!all_stored) break;
    }

    if (all_stored) {
        printf("  PASS: all %d concatenated elements stored correctly\n", combined);
    }

    lstm_free_cell_states(cell_states, 5);
    lstm_free(net);
    return all_stored;
}

/* ==================== Test: State Propagation ==================== */

int test_state_propagation(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double preds[5 * TEST_NUM_OUT];
    int i, t;
    int n, combined;
    int states_different;
    double h_diff, c_diff;

    printf("[TEST] Hidden/Cell state propagation...");
    fflush(stdout);

    net = lstm_init(8, 3, TEST_NUM_OUT, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = (double)(t + 1) * 0.1;
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    lstm_forward(net, sequence, cell_states, 5, preds);

    states_different = 0;
    for (t = 1; t < 5; t++) {
        h_diff = 0.0;
        c_diff = 0.0;
        for (i = 0; i < n; i++) {
            h_diff += fabs(cell_states[t].hidden_state[i] - cell_states[t-1].hidden_state[i]);
            c_diff += fabs(cell_states[t].cell_state[i] - cell_states[t-1].cell_state[i]);
        }
        if (h_diff > 1e-6 || c_diff > 1e-6) {
            states_different = 1;
            break;
        }
    }

    if (!states_different) {
        printf("  FAIL: all hidden/cell states are identical!\n");
        lstm_free_cell_states(cell_states, 5);
        lstm_free(net);
        return 0;
    }

    printf("  PASS: states propagate correctly\n");
    lstm_free_cell_states(cell_states, 5);
    lstm_free(net);
    return 1;
}
