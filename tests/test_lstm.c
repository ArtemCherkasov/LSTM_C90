#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/utils.h"
#include "../include/data.h"
#include "../include/lstm.h"
#include "../include/model.h"
#include "../include/types.h"

/* ==================== Helpers ==================== */

static LSTMCellState *alloc_cell_states(int seq_len, int n, int combined) {
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

/* ==================== Synthetic Dataset Generator ==================== */

static void generate_synthetic_dataset(DataRow **out_rows, int *out_count) {
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

/* ==================== Test: Forward Pass ==================== */

static int test_forward_pass(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double pred;
    int i, t;
    int n, combined;

    printf("[TEST] Forward pass...");
    fflush(stdout);

    net = lstm_init(8, 3, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.5;
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    pred = lstm_forward(net, sequence, cell_states, 5);

    printf("  prediction=%.6f\n", pred);

    if (pred != pred || fabs(pred) > 1e6) {
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

static int test_concatenated_storage(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    int i, t, c;
    int n, combined;
    int all_stored;

    printf("[TEST] Concatenated storage (Bug #1)...");
    fflush(stdout);

    net = lstm_init(8, 3, 42);
    n = net->num_cells;
    combined = n + net->num_features;  /* 8 + 3 = 11 */

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            /* Use distinct values so we can verify storage */
            sequence[t * 3 + i] = 0.1 * (double)(t * 3 + i + 1);
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    lstm_forward(net, sequence, cell_states, 5);

    /* Verify that concatenated[n..combined-1] matches input values */
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

static int test_state_propagation(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    int i, t;
    int n, combined;
    int states_different;
    double h_diff, c_diff;

    printf("[TEST] Hidden/Cell state propagation...");
    fflush(stdout);

    net = lstm_init(8, 3, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = (double)(t + 1) * 0.1;
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    lstm_forward(net, sequence, cell_states, 5);

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

/* ==================== Test: Backward Pass — Non-zero Gradients ==================== */

static int test_backward_pass(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double pred;
    int i, t, r, c;
    int n, combined;
    int has_grad;

    printf("[TEST] Backward pass (gradient check)...");
    fflush(stdout);

    net = lstm_init(8, 3, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.5;
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    pred = lstm_forward(net, sequence, cell_states, 5);

    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, pred, 1.0, 5);

    /* Check that recurrent weight gradients are non-zero */
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

static int test_last_timestep_grad(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double pred;
    int i, t, c;
    int n, combined;
    int has_input_grad;
    int has_forget_grad;
    int has_candidate_grad;

    printf("[TEST] Last timestep gradients (Bug #2)...");
    fflush(stdout);

    net = lstm_init(8, 3, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.3 * (double)(t + 1);
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    pred = lstm_forward(net, sequence, cell_states, 5);

    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, pred, 0.5, 5);

    /* After the fix, the last timestep should contribute non-zero gradients
     * to forget/input/candidate gates (via dcell).
     * We check that bias gradients are non-zero (they accumulate over all timesteps). */
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

static int test_output_grad_accumulation(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double pred;
    int i, t, c;
    int n, combined;
    double grad_single;
    double grad_double;

    printf("[TEST] Output layer gradient accumulation (Bug #3)...");
    fflush(stdout);

    net = lstm_init(8, 3, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.5;
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    pred = lstm_forward(net, sequence, cell_states, 5);

    /* Single backward call */
    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, pred, 1.0, 5);
    grad_single = net->dbias_output_layer;

    /* Two backward calls (same sample) — should accumulate */
    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, pred, 1.0, 5);
    lstm_backward(net, sequence, cell_states, pred, 1.0, 5);
    grad_double = net->dbias_output_layer;

    /* grad_double should be ~2 * grad_single */
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

/* ==================== Test: Weight Update ==================== */

static int test_update(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double pred;
    double old_val;
    int i, t;
    int n, combined;

    printf("[TEST] Weight update...");
    fflush(stdout);

    net = lstm_init(8, 3, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.5;
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    pred = lstm_forward(net, sequence, cell_states, 5);
    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, pred, 1.0, 5);

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

static int test_batch_normalization(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double pred;
    int i, t;
    int n, combined;
    double saved_weight;
    double saved_grad;
    double weight_after_1;
    double weight_after_4;

    printf("[TEST] Batch gradient normalization (Bug #6)...");
    fflush(stdout);

    net = lstm_init(8, 3, 42);
    n = net->num_cells;
    combined = n + net->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.5;
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    pred = lstm_forward(net, sequence, cell_states, 5);

    /* Compute gradient once */
    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, pred, 1.0, 5);

    /* Save original weight and gradient */
    saved_weight = net->weight_forget_recurrent->data[0];
    saved_grad = net->dweight_forget_recurrent->data[0];

    /* Update with batch_size=1, momentum=0 */
    net->mweight_forget_recurrent->data[0] = 0.0;
    lstm_update(net, 0.01, 0.0, 1);
    weight_after_1 = net->weight_forget_recurrent->data[0];

    /* Restore weight, set gradient to 4x (simulating 4 samples) */
    net->weight_forget_recurrent->data[0] = saved_weight;
    net->mweight_forget_recurrent->data[0] = 0.0;
    net->dweight_forget_recurrent->data[0] = saved_grad * 4.0;

    /* Update with batch_size=4 — should normalize by 4, producing same result */
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

/* ==================== Test: Training Loop ==================== */

static int test_training_loop(void) {
    DataRow *raw_data;
    int num_raw;
    Dataset *full_dataset;
    DatasetSplit split;
    Dataset *train;
    Dataset *test;
    LSTMNetwork *net;
    int epoch;
    double train_loss, test_loss;
    double best_train_loss;
    double initial_loss;
    int loss_decreased;
    int num_epochs;
    int num_cells;
    int seq_len;
    double lr;
    double momentum;
    int batch_size;
    int seed;
    int *train_indices;
    int sample, batch_end, actual_batch, b, idx;
    double pred;
    LSTMCellState *cell_states;
    int n, combined;
    int i, j;

    printf("[TEST] Training loop (MSE regression)...");
    fflush(stdout);

    num_epochs = 50;
    num_cells = 16;
    seq_len = 5;
    lr = 0.05;
    momentum = 0.9;
    batch_size = 8;
    seed = 42;
    loss_decreased = 0;

    utils_srand(seed);
    generate_synthetic_dataset(&raw_data, &num_raw);

    full_dataset = data_build_dataset(raw_data, num_raw, seq_len);
    if (!full_dataset) {
        printf("  FAIL: could not build dataset\n");
        utils_safe_free(raw_data);
        return 0;
    }

    data_normalize(full_dataset);
    split = data_split_train_test(full_dataset, 0.8);
    train = split.train;
    test = split.test;

    if (!train || !test) {
        printf("  FAIL: could not split data\n");
        data_free_split(&split);
        data_free_dataset(full_dataset);
        utils_safe_free(raw_data);
        return 0;
    }

    net = lstm_init(num_cells, full_dataset->num_features, seed);
    n = net->num_cells;
    combined = n + net->num_features;

    train_indices = (int *)utils_safe_calloc((size_t)train->num_samples, sizeof(int));
    for (epoch = 0; epoch < train->num_samples; epoch++) {
        train_indices[epoch] = epoch;
    }

    best_train_loss = 1e30;
    initial_loss = 0.0;

    for (epoch = 0; epoch < num_epochs; epoch++) {
        /* Shuffle */
        for (i = train->num_samples - 1; i > 0; i--) {
            j = (int)(utils_rand() * (double)(i + 1));
            if (j > i) j = i;
            { int tmp = train_indices[i]; train_indices[i] = train_indices[j]; train_indices[j] = tmp; }
        }

        for (sample = 0; sample < train->num_samples; sample += batch_size) {
            batch_end = sample + batch_size;
            if (batch_end > train->num_samples) batch_end = train->num_samples;
            actual_batch = batch_end - sample;

            cell_states = alloc_cell_states(seq_len, n, combined);

            lstm_zero_gradients(net);

            for (b = 0; b < actual_batch; b++) {
                idx = train_indices[sample + b];
                pred = lstm_forward(net, train->features[idx], cell_states, seq_len);
                lstm_backward(net, train->features[idx], cell_states, pred, train->labels[idx], seq_len);
            }

            lstm_update(net, lr, momentum, actual_batch);

            lstm_free_cell_state_buffers(cell_states, seq_len);
            free(cell_states);
        }

        train_loss = lstm_compute_loss(net, train);
        test_loss = lstm_compute_loss(net, test);

        if (epoch == 0) initial_loss = train_loss;
        if (train_loss < best_train_loss) best_train_loss = train_loss;

        if (epoch % 10 == 0 || epoch == num_epochs - 1) {
            printf("\n    epoch %3d: train=%.8f test=%.8f", epoch + 1, train_loss, test_loss);
            fflush(stdout);
        }
    }

    printf("\n  initial=%.8f  best=%.8f\n", initial_loss, best_train_loss);

    if (best_train_loss < initial_loss * 0.5) {
        loss_decreased = 1;
    }

    lstm_free(net);
    data_free_split(&split);
    data_free_dataset(full_dataset);
    utils_safe_free(raw_data);
    utils_safe_free(train_indices);

    if (loss_decreased) {
        printf("  PASS: loss decreased by >50%%\n");
        return 1;
    } else {
        printf("  FAIL: loss did not decrease sufficiently\n");
        return 0;
    }
}

/* ==================== Test: Save/Load ==================== */

static int test_save_load(void) {
    LSTMNetwork *net1, *net2;
    LSTMCellState *cell_states;
    double sequence[3 * 5];
    double pred1, pred2;
    int i, t;
    int n, combined;

    printf("[TEST] Save/Load model...");
    fflush(stdout);

    net1 = lstm_init(8, 3, 42);
    n = net1->num_cells;
    combined = n + net1->num_features;

    for (t = 0; t < 5; t++) {
        for (i = 0; i < 3; i++) {
            sequence[t * 3 + i] = 0.5;
        }
    }

    cell_states = alloc_cell_states(5, n, combined);
    pred1 = lstm_forward(net1, sequence, cell_states, 5);
    lstm_free_cell_states(cell_states, 5);

    if (model_save("test_model.txt", net1, 10, 0.5, 0.7, NULL) != 0) {
        printf("  FAIL: could not save model\n");
        lstm_free(net1);
        return 0;
    }

    net2 = lstm_init(8, 3, 99);

    if (model_load("test_model.txt", net2) != 0) {
        printf("  FAIL: could not load model\n");
        lstm_free(net1);
        lstm_free(net2);
        return 0;
    }

    pred2 = lstm_predict(net2, sequence, 5);

    printf("  pred1=%.6f  pred2=%.6f  diff=%.6f\n", pred1, pred2, fabs(pred1 - pred2));

    if (fabs(pred1 - pred2) < 1e-4) {
        printf("  PASS: predictions match after save/load\n");
    } else {
        printf("  FAIL: predictions differ by %.6f\n", fabs(pred1 - pred2));
        lstm_free(net1);
        lstm_free(net2);
        return 0;
    }

    lstm_free(net1);
    lstm_free(net2);
    remove("test_model.txt");
    return 1;
}

/* ==================== Test: Deterministic Forward (64 cells, 64 steps) ===== */

#define DET_N 64
#define DET_F 6
#define DET_T 64
#define DET_W 0.1    /* constant weight value */
#define DET_X 0.5    /* constant input value */

static void set_constant_weights(LSTMNetwork *net, double w) {
    int i, n, combined;
    n = net->num_cells;
    combined = n + net->num_features;

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

    for (i = 0; i < n; i++) {
        net->weight_output_layer->data[i] = w;
    }
    net->bias_output_layer = 0.0;
}

static int test_deterministic_forward_64(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double *sequence;
    double pred;
    int t, i, c;
    int n, combined;
    double h_scalar, c_scalar;
    double max_h_err, max_c_err;
    double tol;

    printf("[TEST] Deterministic forward: 64 cells, 64 steps, 6 features");
    fflush(stdout);

    n = DET_N;
    combined = n + DET_F;
    tol = 1e-10;

    net = lstm_init(n, DET_F, 42);
    set_constant_weights(net, DET_W);

    /* Allocate constant input sequence */
    sequence = (double *)utils_safe_calloc((size_t)DET_T * DET_F, sizeof(double));
    for (t = 0; t < DET_T; t++) {
        for (i = 0; i < DET_F; i++) {
            sequence[t * DET_F + i] = DET_X;
        }
    }

    cell_states = alloc_cell_states(DET_T, n, combined);
    pred = lstm_forward(net, sequence, cell_states, DET_T);

    /* With constant weights and inputs, all cells are symmetric.
     * Compute expected h_t and c_t via scalar recurrence. */
    h_scalar = 0.0;
    c_scalar = 0.0;
    max_h_err = 0.0;
    max_c_err = 0.0;

    for (t = 0; t < DET_T; t++) {
        double concat_sum = (double)n * h_scalar + (double)DET_F * DET_X;
        double gate_pre = DET_W * concat_sum;  /* bias=0 */
        double f_gate = utils_sigmoid(gate_pre);
        double i_gate = utils_sigmoid(gate_pre);
        double g_gate = utils_tanh_act(gate_pre);
        double o_gate = utils_sigmoid(gate_pre);
        double c_new = f_gate * c_scalar + i_gate * g_gate;
        double h_new = o_gate * utils_tanh_act(c_new);

        /* Compare with actual cell_states for ALL cells */
        for (c = 0; c < n; c++) {
            double h_err = fabs(cell_states[t].hidden_state[c] - h_new);
            double c_err = fabs(cell_states[t].cell_state[c] - c_new);
            if (h_err > max_h_err) max_h_err = h_err;
            if (c_err > max_c_err) max_c_err = c_err;

            /* Verify gates are also correct */
            if (fabs(cell_states[t].forget_gate[c] - f_gate) > tol) {
                printf("\n  FAIL: forget_gate[%d][%d] mismatch\n", t, c);
                free(sequence);
                lstm_free_cell_states(cell_states, DET_T);
                lstm_free(net);
                return 0;
            }
        }

        /* Verify output layer at this timestep (h_t is the output) */
        if (t == DET_T - 1) {
            double expected_pred = DET_W * (double)n * h_new;  /* W_out * h + bias=0 */
            if (fabs(pred - expected_pred) > 1e-8) {
                printf("\n  FAIL: pred=%.10f expected=%.10f\n", pred, expected_pred);
                free(sequence);
                lstm_free_cell_states(cell_states, DET_T);
                lstm_free(net);
                return 0;
            }
        }

        h_scalar = h_new;
        c_scalar = c_new;
    }

    printf("\n  max_h_err=%.2e  max_c_err=%.2e  pred=%.8f", max_h_err, max_c_err, pred);

    if (max_h_err < tol && max_c_err < tol) {
        printf("\n  PASS: H and C vectors match analytical solution at all %d steps\n", DET_T);
        free(sequence);
        lstm_free_cell_states(cell_states, DET_T);
        lstm_free(net);
        return 1;
    } else {
        printf("\n  FAIL: H or C error exceeds tolerance %.2e\n", tol);
        free(sequence);
        lstm_free_cell_states(cell_states, DET_T);
        lstm_free(net);
        return 0;
    }
}

/* ==================== Test: BPTT Numerical Gradient Check (64x64x6) ======== */

#define GRAD_N 64
#define GRAD_F 6
#define GRAD_T 5     /* fewer timesteps to avoid gradient explosion */
#define GRAD_W 0.01  /* small weight to keep gradients bounded */

static int test_bptt_gradient_check_64(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double *sequence;
    double pred, target;
    double eps;
    int n, combined;
    int t, i, c;
    int num_checks;
    int num_passed;
    double max_rel_err;
    double rel_tol;

    /* Indices of weights to check: (gate, row, col) */
    int check_gate[8];
    int check_idx[8];
    double anal_grad[8];
    double num_grad[8];
    double *weight_ptr[8];

    printf("[TEST] BPTT numerical gradient check: 64 cells, %d steps", GRAD_T);
    fflush(stdout);

    n = GRAD_N;
    combined = n + GRAD_F;
    eps = 1e-5;
    rel_tol = 1e-3;  /* 0.1% relative tolerance */
    target = 0.1;

    net = lstm_init(n, GRAD_F, 42);
    set_constant_weights(net, GRAD_W);

    sequence = (double *)utils_safe_calloc((size_t)GRAD_T * GRAD_F, sizeof(double));
    for (t = 0; t < GRAD_T; t++) {
        for (i = 0; i < GRAD_F; i++) {
            sequence[t * GRAD_F + i] = DET_X;
        }
    }

    cell_states = alloc_cell_states(GRAD_T, n, combined);

    /* Step 1: Forward + Backward to get analytical gradients */
    pred = lstm_forward(net, sequence, cell_states, GRAD_T);
    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, pred, target, GRAD_T);

    /* Select 8 weights to check: spread across gates and positions */
    num_checks = 8;

    /* forget_recurrent[0][0] */
    check_gate[0] = 0; check_idx[0] = 0;
    weight_ptr[0] = &net->weight_forget_recurrent->data[0];
    anal_grad[0] = net->dweight_forget_recurrent->data[0];

    /* forget_recurrent[10][35] */
    check_gate[1] = 0; check_idx[1] = 10 * combined + 35;
    weight_ptr[1] = &net->weight_forget_recurrent->data[10 * combined + 35];
    anal_grad[1] = net->dweight_forget_recurrent->data[10 * combined + 35];

    /* input_recurrent[5][60] */
    check_gate[2] = 1; check_idx[2] = 5 * combined + 60;
    weight_ptr[2] = &net->weight_input_recurrent->data[5 * combined + 60];
    anal_grad[2] = net->dweight_input_recurrent->data[5 * combined + 60];

    /* output_recurrent[20][10] */
    check_gate[3] = 2; check_idx[3] = 20 * combined + 10;
    weight_ptr[3] = &net->weight_output_recurrent->data[20 * combined + 10];
    anal_grad[3] = net->dweight_output_recurrent->data[20 * combined + 10];

    /* candidate_recurrent[30][50] */
    check_gate[4] = 3; check_idx[4] = 30 * combined + 50;
    weight_ptr[4] = &net->weight_candidate_recurrent->data[30 * combined + 50];
    anal_grad[4] = net->dweight_candidate_recurrent->data[30 * combined + 50];

    /* output_layer[0] */
    check_gate[5] = 4; check_idx[5] = 0;
    weight_ptr[5] = &net->weight_output_layer->data[0];
    anal_grad[5] = net->dweight_output_layer->data[0];

    /* output_layer[32] */
    check_gate[6] = 4; check_idx[6] = 32;
    weight_ptr[6] = &net->weight_output_layer->data[32];
    anal_grad[6] = net->dweight_output_layer->data[32];

    /* forget_recurrent[63][69] (last cell, last col) */
    check_gate[7] = 0; check_idx[7] = 63 * combined + 69;
    weight_ptr[7] = &net->weight_forget_recurrent->data[63 * combined + 69];
    anal_grad[7] = net->dweight_forget_recurrent->data[63 * combined + 69];

    /* Step 2: Numerical gradients via finite differences */
    for (i = 0; i < num_checks; i++) {
        double orig_w, loss_plus, loss_minus, p;

        orig_w = *weight_ptr[i];

        *weight_ptr[i] = orig_w + eps;
        p = lstm_forward(net, sequence, cell_states, GRAD_T);
        loss_plus = 0.5 * (p - target) * (p - target);

        *weight_ptr[i] = orig_w - eps;
        p = lstm_forward(net, sequence, cell_states, GRAD_T);
        loss_minus = 0.5 * (p - target) * (p - target);

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

/* ==================== Test: Deterministic Forward 128x256x12 ================= */

#define DET2_N 128
#define DET2_F 12
#define DET2_T 256
#define DET2_W 0.05
#define DET2_X 0.3

static int test_deterministic_forward_128(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double *sequence;
    double pred;
    int t, i, c;
    int n, combined;
    double h_scalar, c_scalar;
    double max_h_err, max_c_err;
    double tol;

    printf("[TEST] Deterministic forward: 128 cells, 256 steps, 12 features");
    fflush(stdout);

    n = DET2_N;
    combined = n + DET2_F;
    tol = 1e-10;

    net = lstm_init(n, DET2_F, 42);
    set_constant_weights(net, DET2_W);

    sequence = (double *)utils_safe_calloc((size_t)DET2_T * DET2_F, sizeof(double));
    for (t = 0; t < DET2_T; t++) {
        for (i = 0; i < DET2_F; i++) {
            sequence[t * DET2_F + i] = DET2_X;
        }
    }

    cell_states = alloc_cell_states(DET2_T, n, combined);
    pred = lstm_forward(net, sequence, cell_states, DET2_T);

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
                lstm_free_cell_states(cell_states, DET2_T);
                lstm_free(net);
                return 0;
            }
            if (fabs(cell_states[t].input_gate[c] - i_gate) > tol) {
                printf("\n  FAIL: input_gate[%d][%d] mismatch\n", t, c);
                free(sequence);
                lstm_free_cell_states(cell_states, DET2_T);
                lstm_free(net);
                return 0;
            }
            if (fabs(cell_states[t].output_gate[c] - o_gate) > tol) {
                printf("\n  FAIL: output_gate[%d][%d] mismatch\n", t, c);
                free(sequence);
                lstm_free_cell_states(cell_states, DET2_T);
                lstm_free(net);
                return 0;
            }
            if (fabs(cell_states[t].candidate[c] - g_gate) > tol) {
                printf("\n  FAIL: candidate[%d][%d] mismatch\n", t, c);
                free(sequence);
                lstm_free_cell_states(cell_states, DET2_T);
                lstm_free(net);
                return 0;
            }
        }

        if (t == DET2_T - 1) {
            double expected_pred = DET2_W * (double)n * h_new;
            if (fabs(pred - expected_pred) > 1e-8) {
                printf("\n  FAIL: pred=%.10f expected=%.10f\n", pred, expected_pred);
                free(sequence);
                lstm_free_cell_states(cell_states, DET2_T);
                lstm_free(net);
                return 0;
            }
        }

        h_scalar = h_new;
        c_scalar = c_new;
    }

    printf("\n  max_h_err=%.2e  max_c_err=%.2e  pred=%.8f", max_h_err, max_c_err, pred);

    if (max_h_err < tol && max_c_err < tol) {
        printf("\n  PASS: H and C vectors match analytical solution at all %d steps\n", DET2_T);
        free(sequence);
        lstm_free_cell_states(cell_states, DET2_T);
        lstm_free(net);
        return 1;
    } else {
        printf("\n  FAIL: H or C error exceeds tolerance %.2e\n", tol);
        free(sequence);
        lstm_free_cell_states(cell_states, DET2_T);
        lstm_free(net);
        return 0;
    }
}

/* ==================== Test: BPTT Gradient Check 128x256x12 ================== */

#define GRAD2_N 128
#define GRAD2_F 12
#define GRAD2_T 4       /* few timesteps to avoid gradient explosion */
#define GRAD2_W 0.02

static int test_bptt_gradient_check_128(void) {
    LSTMNetwork *net;
    LSTMCellState *cell_states;
    double *sequence;
    double pred, target;
    double eps;
    int n, combined;
    int t, i, c;
    int num_checks;
    int num_passed;
    double max_rel_err;
    double rel_tol;

    int check_gate[10];
    int check_idx[10];
    double anal_grad[10];
    double num_grad[10];
    double *weight_ptr[10];

    printf("[TEST] BPTT numerical gradient check: 128 cells, %d steps, 12 features", GRAD2_T);
    fflush(stdout);

    n = GRAD2_N;
    combined = n + GRAD2_F;
    eps = 1e-4;
    rel_tol = 1e-3;  /* 0.1% relative tolerance */
    target = 1.0;

    net = lstm_init(n, GRAD2_F, 42);
    set_constant_weights(net, GRAD2_W);

    sequence = (double *)utils_safe_calloc((size_t)GRAD2_T * GRAD2_F, sizeof(double));
    for (t = 0; t < GRAD2_T; t++) {
        for (i = 0; i < GRAD2_F; i++) {
            sequence[t * GRAD2_F + i] = DET2_X;
        }
    }

    cell_states = alloc_cell_states(GRAD2_T, n, combined);

    /* Step 1: Forward + Backward to get analytical gradients */
    pred = lstm_forward(net, sequence, cell_states, GRAD2_T);
    lstm_zero_gradients(net);
    lstm_backward(net, sequence, cell_states, pred, target, GRAD2_T);

    num_checks = 10;

    /* forget_recurrent[0][0] */
    check_gate[0] = 0; check_idx[0] = 0;
    weight_ptr[0] = &net->weight_forget_recurrent->data[0];
    anal_grad[0] = net->dweight_forget_recurrent->data[0];

    /* forget_recurrent[20][70] */
    check_gate[1] = 0; check_idx[1] = 20 * combined + 70;
    weight_ptr[1] = &net->weight_forget_recurrent->data[20 * combined + 70];
    anal_grad[1] = net->dweight_forget_recurrent->data[20 * combined + 70];

    /* input_recurrent[10][120] */
    check_gate[2] = 1; check_idx[2] = 10 * combined + 120;
    weight_ptr[2] = &net->weight_input_recurrent->data[10 * combined + 120];
    anal_grad[2] = net->dweight_input_recurrent->data[10 * combined + 120];

    /* output_recurrent[40][20] */
    check_gate[3] = 2; check_idx[3] = 40 * combined + 20;
    weight_ptr[3] = &net->weight_output_recurrent->data[40 * combined + 20];
    anal_grad[3] = net->dweight_output_recurrent->data[40 * combined + 20];

    /* candidate_recurrent[60][100] */
    check_gate[4] = 3; check_idx[4] = 60 * combined + 100;
    weight_ptr[4] = &net->weight_candidate_recurrent->data[60 * combined + 100];
    anal_grad[4] = net->dweight_candidate_recurrent->data[60 * combined + 100];

    /* output_layer[0] */
    check_gate[5] = 4; check_idx[5] = 0;
    weight_ptr[5] = &net->weight_output_layer->data[0];
    anal_grad[5] = net->dweight_output_layer->data[0];

    /* output_layer[64] */
    check_gate[6] = 4; check_idx[6] = 64;
    weight_ptr[6] = &net->weight_output_layer->data[64];
    anal_grad[6] = net->dweight_output_layer->data[64];

    /* output_layer[127] (last) */
    check_gate[7] = 4; check_idx[7] = 127;
    weight_ptr[7] = &net->weight_output_layer->data[127];
    anal_grad[7] = net->dweight_output_layer->data[127];

    /* forget_recurrent[100][130] */
    check_gate[8] = 0; check_idx[8] = 100 * combined + 130;
    weight_ptr[8] = &net->weight_forget_recurrent->data[100 * combined + 130];
    anal_grad[8] = net->dweight_forget_recurrent->data[100 * combined + 130];

    /* candidate_recurrent[127][139] (last cell, last col) */
    check_gate[9] = 3; check_idx[9] = 127 * combined + (combined - 1);
    weight_ptr[9] = &net->weight_candidate_recurrent->data[127 * combined + (combined - 1)];
    anal_grad[9] = net->dweight_candidate_recurrent->data[127 * combined + (combined - 1)];

    /* Step 2: Numerical gradients via finite differences */
    for (i = 0; i < num_checks; i++) {
        double orig_w, loss_plus, loss_minus, p;

        orig_w = *weight_ptr[i];

        *weight_ptr[i] = orig_w + eps;
        p = lstm_forward(net, sequence, cell_states, GRAD2_T);
        loss_plus = 0.5 * (p - target) * (p - target);

        *weight_ptr[i] = orig_w - eps;
        p = lstm_forward(net, sequence, cell_states, GRAD2_T);
        loss_minus = 0.5 * (p - target) * (p - target);

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

/* ==================== Main ==================== */

int main(void) {
    int passed = 0;
    int total = 0;

    printf("========================================\n");
    printf("LSTM Network Tests (C90, MSE regression)\n");
    printf("========================================\n\n");

    utils_srand(12345);

    total++; passed += test_forward_pass();
    printf("\n");

    total++; passed += test_concatenated_storage();
    printf("\n");

    total++; passed += test_state_propagation();
    printf("\n");

    total++; passed += test_backward_pass();
    printf("\n");

    total++; passed += test_last_timestep_grad();
    printf("\n");

    total++; passed += test_output_grad_accumulation();
    printf("\n");

    total++; passed += test_update();
    printf("\n");

    total++; passed += test_batch_normalization();
    printf("\n");

    total++; passed += test_training_loop();
    printf("\n");

    total++; passed += test_save_load();
    printf("\n");

    total++; passed += test_deterministic_forward_64();
    printf("\n");

    total++; passed += test_bptt_gradient_check_64();
    printf("\n");

    total++; passed += test_deterministic_forward_128();
    printf("\n");

    total++; passed += test_bptt_gradient_check_128();
    printf("\n");

    printf("========================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}
