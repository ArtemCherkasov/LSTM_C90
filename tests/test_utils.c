#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==================== alloc_cell_states ==================== */

LSTMCellState *alloc_cell_states(int seq_len, int n, int combined) {
    LSTMCellState *cs;
    int t;
    cs = (LSTMCellState *)utils_safe_calloc((size_t)seq_len, sizeof(LSTMCellState));
    for (t = 0; t < seq_len; t++) {
        cs[t].hidden_state = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cs[t].cell_state   = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cs[t].forget_gate  = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cs[t].input_gate   = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cs[t].output_gate  = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cs[t].candidate    = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cs[t].concatenated = (double *)utils_safe_calloc((size_t)combined, sizeof(double));
    }
    return cs;
}

/* ==================== generate_synthetic_dataset ==================== */

void generate_synthetic_dataset(DataRow **out_rows, int *out_count) {
    int count = 300;
    DataRow *rows;
    int i;
    double base_price;

    base_price = 1.10000;
    rows = (DataRow *)utils_safe_malloc((size_t)count * sizeof(DataRow));

    for (i = 0; i < count; i++) {
        double sign = (i % 6 < 3) ? 1.0 : -1.0;
        double noise = (utils_rand() - 0.5) * 0.0002;
        double cret = sign * 0.003 + noise;

        rows[i].open = base_price;
        rows[i].close = base_price * (1.0 + cret);
        rows[i].high = (rows[i].close > rows[i].open) ?
            rows[i].close + 0.0005 : rows[i].open + 0.0005;
        rows[i].low = (rows[i].close < rows[i].open) ?
            rows[i].close - 0.0005 : rows[i].open - 0.0005;

        rows[i].tickvol = 1000 + (int)(utils_rand() * 500);
        rows[i].vol = 0;
        rows[i].spread = 1;

        snprintf(rows[i].date, 16, "2024.01.%02d", (i % 28) + 1);
        snprintf(rows[i].time_, 16, "%02d:00", i % 24);

        base_price = rows[i].close;
    }

    *out_rows = rows;
    *out_count = count;
}

/* ==================== set_constant_weights ==================== */

void set_constant_weights(LSTMNetwork *net, double w) {
    int i, n, combined, num_out;
    n = net->num_cells;
    combined = n + net->num_features;
    num_out = net->num_outputs;

    #define SET_GATE(name) do { \
        for (i = 0; i < n * combined; i++) \
            net->weight_##name##_recurrent->data[i] = w; \
        for (i = 0; i < n * net->num_features; i++) \
            net->weight_##name##_input->data[i] = w; \
        for (i = 0; i < n; i++) \
            net->bias_##name[i] = 0.0; \
    } while (0)

    SET_GATE(forget);
    SET_GATE(input);
    SET_GATE(output);
    SET_GATE(candidate);
    #undef SET_GATE

    for (i = 0; i < num_out * n; i++) {
        net->weight_output_layer->data[i] = w;
    }
    for (i = 0; i < num_out; i++) {
        net->bias_output_layer[i] = 0.0;
    }
}
