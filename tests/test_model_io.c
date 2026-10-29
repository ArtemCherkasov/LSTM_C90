#include "test_utils.h"
#include <stdio.h>
#include <math.h>

/* ==================== Test: Save/Load ==================== */

int test_save_load(void) {
    LSTMNetwork *net1, *net2;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double preds1[5 * TEST_NUM_OUT];
    double preds2[5 * TEST_NUM_OUT];
    int i, t, o;
    int n, combined;
    double max_diff;

    printf("[TEST] Save/Load model...");
    fflush(stdout);

    net1 = lstm_init(8, 3, TEST_NUM_OUT, 42);
    n = net1->num_cells;
    combined = n + net1->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.5;
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    lstm_forward(net1, sequence, cell_states, 5, preds1);
    lstm_free_cell_states(cell_states, 5);

    if (model_save("test_model.txt", net1, 10, 0.5, 0.7, NULL) != 0) {
        printf("  FAIL: could not save model\n");
        lstm_free(net1);
        return 0;
    }

    net2 = lstm_init(8, 3, TEST_NUM_OUT, 99);

    if (model_load("test_model.txt", net2) != 0) {
        printf("  FAIL: could not load model\n");
        lstm_free(net1);
        lstm_free(net2);
        return 0;
    }

    lstm_predict(net2, sequence, 5, preds2);

    max_diff = 0.0;
    for (o = 0; o < 5*TEST_NUM_OUT; o++) {
        double diff = fabs(preds1[o] - preds2[o]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("  max_diff=%.6f\n", max_diff);

    if (max_diff < 1e-4) {
        printf("  PASS: predictions match after save/load\n");
    } else {
        printf("  FAIL: predictions differ by %.6f\n", max_diff);
        lstm_free(net1);
        lstm_free(net2);
        return 0;
    }

    lstm_free(net1);
    lstm_free(net2);
    remove("test_model.txt");
    return 1;
}
