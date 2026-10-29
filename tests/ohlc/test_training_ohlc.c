#include "../test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ==================== Test: Training Loop ==================== */

int test_training_loop_ohlc(void) {
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
    double *preds;
    int o;
    LSTMCellState *cell_states;
    int n, combined;
    int i, j;

    printf("[TEST] Training loop OHLC (MSE regression, multi-output)...");
    fflush(stdout);

    num_epochs = 50;
    num_cells = 16;
    seq_len = 5;
    lr = 0.001;
    momentum = 0.9;
    batch_size = 8;
    seed = 42;
    loss_decreased = 0;

    utils_srand(seed);
    generate_synthetic_dataset(&raw_data, &num_raw);

    full_dataset = data_build_dataset_ohlc(raw_data, num_raw, seq_len, 5);
    if (!full_dataset) {
        printf("  FAIL: could not build dataset\n");
        utils_safe_free(raw_data);
        return 0;
    }

    data_normalize_features(full_dataset);
    data_normalize_labels_ohlc(full_dataset);;
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

    net = lstm_init(num_cells, full_dataset->num_features, full_dataset->num_outputs, seed);
    n = net->num_cells;
    combined = n + net->num_features;

    preds = (double *)utils_safe_calloc((size_t)(full_dataset->num_outputs * seq_len), sizeof(double));

    train_indices = (int *)utils_safe_calloc((size_t)train->num_samples, sizeof(int));
    for (epoch = 0; epoch < train->num_samples; epoch++) {
        train_indices[epoch] = epoch;
    }

    best_train_loss = 1e30;
    initial_loss = 0.0;

    for (epoch = 0; epoch < num_epochs; epoch++) {
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
                lstm_forward(net, train->features[idx], cell_states, seq_len, preds);
                lstm_backward(net, train->features[idx], cell_states, preds,
                              train->labels[idx], seq_len);
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

    utils_safe_free(preds);
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
