#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include "include/utils.h"
#include "include/data.h"
#include "include/lstm.h"
#include "include/model.h"
#include "include/types.h"

/* ==================== Global for signal handler ==================== */

static LSTMNetwork *g_saved_network = NULL;
static const char *g_model_path = NULL;
static double g_epochs_trained = 0.0;
static double g_best_train_mse = 0.0;
static double g_best_test_accuracy = 0.0;
static Dataset *g_full_dataset = NULL;

static void signal_handler(int signum) {
    (void)signum;
    printf("\n\nSignal received, saving model...\n");
    if (g_saved_network != NULL && g_model_path != NULL) {
        model_save(g_model_path, g_saved_network,
                   g_epochs_trained, g_best_train_mse, g_best_test_accuracy, g_full_dataset);
        printf("Model saved to %s\n", g_model_path);
    }
    exit(1);
}

/* ==================== Shuffle indices ==================== */

static void shuffle_indices(int *indices, int count, unsigned int seed) {
    int i, j;
    unsigned int tmp;
    for (i = count - 1; i > 0; i--) {
        tmp = (unsigned int)(utils_rand() * (double)(i + 1));
        if (tmp > (unsigned int)i) tmp = (unsigned int)i;
        j = indices[tmp];
        indices[tmp] = indices[i];
        indices[i] = j;
    }
}

/* ==================== Constants ==================== */
#define DEFAULT_NUM_LSTM_CELLS 128
#define DEFAULT_NUM_EPOCHS 100
#define DEFAULT_LEARNING_RATE 0.0005
#define DEFAULT_SEQUENCE_LENGTH 64
#define DEFAULT_BATCH_SIZE 32
#define DEFAULT_MOMENTUM 0.9
#define DEFAULT_SEED 42
#define TRAIN_TEST_RATIO 0.8
#define PROGRESS_BAR_WIDTH 30
#define LOSS_SCALE 1000
#define MODEL_PATH "../resources/model.txt"

int main(int argc, char *argv[]) {
    int num_lstm_cells, num_epochs, sequence_length, batch_size;
    double learning_rate, momentum;
    unsigned int seed;
    const char *data_path;
    const char *model_path;
    DataRow *raw_data;
    int num_raw;
    Dataset *full_dataset;
    int j;
    DatasetSplit split;
    Dataset *train;
    Dataset *test;
    LSTMNetwork *network;
    int epoch;
    LSTMCellState *cell_states;
    double best_train_loss;
    double best_test_accuracy;
    int *train_indices;
    int sample;
    int batch_end, actual_batch;
    int b;
    int idx;
    int i;
    double train_loss;
    double train_acc;
    double test_loss;
    double test_acc;
    int demo_start;
    int d;
    double pred_demo;
    int actual;
    int predicted;
    const char *actual_str;
    const char *pred_str;
    double pred;
    int predict_mode = 0;
    int raw_line = 0;
    int predict_count = 0;
    int real_mode = 0;
    int real_count = 0;
    double batch_loss_sum;
    double batch_loss_avg;
    int bar_width;
    int progress;

    /* Parse CLI arguments */
    num_lstm_cells = DEFAULT_NUM_LSTM_CELLS;
    num_epochs = DEFAULT_NUM_EPOCHS;
    learning_rate = DEFAULT_LEARNING_RATE;
    sequence_length = DEFAULT_SEQUENCE_LENGTH;
    batch_size = DEFAULT_BATCH_SIZE;
    momentum = DEFAULT_MOMENTUM;
    seed = DEFAULT_SEED;
    predict_mode = 0;
    raw_line = 0;
    predict_count = 0;
    real_mode = 0;
    real_count = 0;
    data_path = "../resources/EURUSD_H1_200906120000_202509251100.csv";
    model_path = "../resources/model.txt";

    /* Allow overriding data_path and model_path via CLI */
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[i + 1];
            i++;
        }
    }

    /* Register signal handler for graceful shutdown */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--line") == 0 && i + 1 < argc) {
            raw_line = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--predict") == 0 && i + 1 < argc) {
            predict_mode = 1;
            predict_count = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--real") == 0 && i + 1 < argc) {
            real_mode = 1;
            real_count = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--cells") == 0 && i + 1 < argc) {
            num_lstm_cells = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            num_epochs = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            sequence_length = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[i + 1];
            i++;
        }
    }

    /* ==================== REAL DATA MODE ==================== */
    if (real_mode && raw_line > 0 && real_count > 0) {
        int k;
        int end_row;

        /* Load data */
        raw_data = NULL;
        num_raw = 0;
        if (data_load_raw(data_path, &raw_data, &num_raw) != 0) {
            fprintf(stderr, "Failed to load data.\n");
            return 1;
        }

        end_row = raw_line + real_count - 1;
        if (end_row > num_raw - 1) end_row = num_raw - 1;

        for (k = raw_line; k <= end_row; k++) {
            printf("%.5f %.5f %.5f %.5f\n",
                   raw_data[k].open,
                   raw_data[k].high,
                   raw_data[k].low,
                   raw_data[k].close);
        }
        utils_safe_free(raw_data);
        return 0;
    }

    /* ==================== PREDICT MODE ==================== */
    if (predict_mode && raw_line > 0 && predict_count > 0) {
        int seq_len, num_features, num_available, start_row, end_row;
        int f, fi, s, ref_steps, p, norm_count;
        double cr, rng_val, cret, vol_ret, ma_diff, vol;
        double feat_min[6], feat_max[6], mmin, mmax;
        double **features;
        double *seq_buf;
        LSTMNetwork *pred_network;
        char mline[512], mname[64];

        seq_len = sequence_length;
        num_features = 6;
        start_row = 1;
        end_row = raw_line;
        ref_steps = 64;
        norm_count = 0;

        /* Load data */
        raw_data = NULL;
        num_raw = 0;
        if (data_load_raw(data_path, &raw_data, &num_raw) != 0) {
            fprintf(stderr, "Failed to load data.\n");
            return 1;
        }

        /* Load model */
        pred_network = lstm_init(num_lstm_cells, num_features, seed);
        if (model_load(model_path, pred_network) != 0) {
            fprintf(stderr, "Failed to load model from %s\n", model_path);
            utils_safe_free(raw_data);
            lstm_free(pred_network);
            return 1;
        }

        /* Load normalization params from model file */
        {
            FILE *mf = fopen(model_path, "r");
            if (mf) {
                while (fgets(mline, sizeof(mline), mf) && norm_count < 6) {
                    utils_trim(mline);
                    if (strncmp(mline, "feature_name:", strlen("feature_name:")) == 0) {
                        /* Parse: feature_name: name | min: value | max: value */
                        char *p = mline;
                        char *min_pos = strstr(p, "min:");
                        char *max_pos = strstr(p, "max:");
                        if (min_pos && max_pos) {
                            mmin = strtod(min_pos + 4, NULL);
                            mmax = strtod(max_pos + 4, NULL);
                            feat_min[norm_count] = mmin;
                            feat_max[norm_count] = mmax;
                            norm_count++;
                            printf("NORM[%d]: min=%.8f max=%.8f\n", norm_count - 1, mmin, mmax);
                        }
                    }
                }
                fclose(mf);
            }
        } /* end normalization params block */

        end_row = raw_line;
        if (end_row > num_raw - 1) end_row = num_raw - 1;

        /* Compute features up to end_row + predict_count - 1 for all prediction windows */
        {
            int predict_end_row = end_row + predict_count - 1;
            if (predict_end_row >= num_raw) predict_end_row = num_raw - 1;
            num_available = predict_end_row - start_row + 1;
        } /* end predict_end_row block */
        /* num_available is now set */

        if (num_available < seq_len) {
            fprintf(stderr, "Error: Need at least %d rows for sequence, have %d\n",
                    seq_len, num_available);
            utils_safe_free(raw_data);
            lstm_free(pred_network);
            return 1;
        }
        if (predict_count > num_raw - end_row) {
            fprintf(stderr, "Error: Can predict at most %d steps ahead (line %d, data has %d rows)\n",
                    num_raw - end_row, end_row, num_raw);
            utils_safe_free(raw_data);
            lstm_free(pred_network);
            return 1;
        }

        /* Allocate feature matrix dynamically */
        features = (double **)utils_safe_calloc((size_t)num_available, sizeof(double *));
        for (f = 0; f < num_available; f++) {
            features[f] = (double *)utils_safe_calloc((size_t)10, sizeof(double));
        }

        /* Allocate sequence buffer for max reference steps */
        seq_buf = (double *)utils_safe_calloc((size_t)(ref_steps * num_features), sizeof(double));

        /* Compute and normalize features */
        {
            int predict_end_row = end_row + predict_count - 1;
            int pi;
            if (predict_end_row >= num_raw) predict_end_row = num_raw - 1;
            for (pi = start_row; pi <= predict_end_row; pi++) {
                int f = pi - start_row;
                cr = (raw_data[pi].close - raw_data[pi].open) / raw_data[pi].open;
                rng_val = (raw_data[pi].high - raw_data[pi].low) / raw_data[pi].open;
                cret = (pi > start_row) ? (raw_data[pi].close - raw_data[pi-1].close) / raw_data[pi-1].close : 0.0;
                vol_ret = (pi > start_row && raw_data[pi-1].tickvol > 0) ?
                    (raw_data[pi].tickvol - raw_data[pi-1].tickvol) / (double)raw_data[pi-1].tickvol : 0.0;
                ma_diff = 0.0;
                if (pi >= 20) {
                    double sum_short = 0.0, sum_long = 0.0;
                    int kk;
                    for (kk = 0; kk < 5; kk++) sum_short += raw_data[pi - kk].close;
                    for (kk = 0; kk < 20; kk++) sum_long += raw_data[pi - kk].close;
                    sum_short /= 5.0;
                    sum_long /= 20.0;
                    ma_diff = (sum_short - sum_long) / sum_long;
                }
                vol = 0.0;
                if (pi >= 12) {
                    double sum_ret = 0.0;
                    int ret_count;
                    int mm;
                    ret_count = 0;
                    for (mm = pi - 12; mm < pi; mm++) {
                        if (mm > 0) {
                            sum_ret += (raw_data[mm].close - raw_data[mm-1].close) / raw_data[mm-1].close;
                            ret_count++;
                        }
                    }
                    if (ret_count > 0) {
                        double mean_val = sum_ret / (double)ret_count;
                        double sum_sq = 0.0;
                        for (mm = pi - 12; mm < pi; mm++) {
                            if (mm > 0) {
                                double r_val = (raw_data[mm].close - raw_data[mm-1].close) / raw_data[mm-1].close;
                                sum_sq += (r_val - mean_val) * (r_val - mean_val);
                            }
                        }
                        vol = sqrt(sum_sq / (double)ret_count);
                    }
                }
                features[f][0] = cr;
                features[f][1] = rng_val;
                features[f][2] = cret;
                features[f][3] = vol_ret;
                features[f][4] = ma_diff;
                features[f][5] = vol;

                for (fi = 0; fi < num_features; fi++) {
                    double r = feat_max[fi] - feat_min[fi];
                    if (r < 1e-15) r = 1.0;
                    features[f][fi] = (features[f][fi] - feat_min[fi]) / r;
                }
            }
        }

        /* Predict */
        /* Full output: 64 reference (real) + 16 prediction (autoregressive) */
        {
            double prev_predicted_close;
            int first_pred;
            double **extended_features;
            int extended_count;
            int pred_steps = predict_count;
            int p, s, fi, f;
            double predicted_close_return;
            double predicted_close;
            double predicted_high;
            double predicted_low;
            double predicted_open;
            double range;
            double norm_range;
            double raw_pred;
            double last_cr, last_crng, last_cret, last_vol_ret, last_ma_diff, last_vol;
            int feat_idx;

            prev_predicted_close = raw_data[end_row].close;
            first_pred = 1;

            /* Allocate extended feature matrix for autoregressive prediction */
            extended_count = num_available + pred_steps + 1;
            extended_features = (double **)utils_safe_calloc((size_t)extended_count, sizeof(double *));
            for (f = 0; f < extended_count; f++) {
                extended_features[f] = (double *)utils_safe_calloc((size_t)num_features, sizeof(double));
            }

            /* Copy real features to extended buffer */
            for (f = 0; f < num_available; f++) {
                for (fi = 0; fi < num_features; fi++) {
                    extended_features[f][fi] = features[f][fi];
                }
            }

            printf("\n  === Full Output: %d reference + %d prediction ===\n", ref_steps, pred_steps);

            /* Stage 1: Output reference (real data) steps */
            for (p = 0; p < ref_steps; p++) {
                int current_row = end_row - ref_steps + 1 + p;
                int data_row = current_row;
                if (current_row < 0 || current_row >= num_available) continue;
                if (data_row < 0 || data_row >= num_raw) continue;

                printf("REF [%03d] %.5f %.5f %.5f %.5f\n",
                       p + 1, raw_data[data_row].open, raw_data[data_row].high,
                       raw_data[data_row].low, raw_data[data_row].close);
            }

            /* Stage 2: Autoregressive prediction steps */
            for (p = 0; p < pred_steps; p++) {
                int current_row;
                int data_row;

                /* Use last ref_steps features (including predicted) */
                current_row = num_available - 1 + p;
                data_row = end_row + p;

                if (current_row < ref_steps - 1) continue;

                /* Build sequence: last ref_steps features */
                for (s = 0; s < ref_steps; s++) {
                    for (fi = 0; fi < num_features; fi++) {
                        seq_buf[s * num_features + fi] =
                            extended_features[current_row - ref_steps + 1 + s][fi];
                    }
                }

                pred = lstm_predict(pred_network, seq_buf, ref_steps);

                /* Denormalize close_return (feature index 2) */
                norm_range = feat_max[2] - feat_min[2];
                if (norm_range < 1e-15) norm_range = 1.0;
                raw_pred = pred;
                predicted_close_return = raw_pred * norm_range + feat_min[2];

                /* Clip return to realistic range (±2%) to prevent runaway predictions */
                if (predicted_close_return > 0.02) predicted_close_return = 0.02;
                if (predicted_close_return < -0.02) predicted_close_return = -0.02;

                /* Use real open for first step, predicted for autoregressive */
                if (first_pred) {
                    predicted_open = raw_data[end_row].open;
                    first_pred = 0;
                } else {
                    predicted_open = prev_predicted_close;
                }

                predicted_close = predicted_open * (1.0 + predicted_close_return);

                /* Additional safety: limit max price change per step to ±1% */
                {
                    double max_change = predicted_open * 0.01;
                    double change = predicted_close - predicted_open;
                    if (change > max_change) {
                        predicted_close = predicted_open + max_change;
                    } else if (change < -max_change) {
                        predicted_close = predicted_open - max_change;
                    }
                }

                /* Use a fraction of the real range for estimated high/low. */
                range = (data_row < num_raw) ?
                    (raw_data[data_row].high - raw_data[data_row].low) :
                    (raw_data[end_row].high - raw_data[end_row].low);
                /* Use 0.25 multiplier to avoid extreme candles */
                predicted_high = (predicted_close > predicted_open) ?
                    (predicted_close + range * 0.25) : (predicted_open + range * 0.25);
                predicted_low = (predicted_close < predicted_open) ?
                    (predicted_close - range * 0.25) : (predicted_open - range * 0.25);

                prev_predicted_close = predicted_close;

                printf("PRE [%03d] %.5f %.5f %.5f %.5f\n",
                       p + 1, predicted_open, predicted_high, predicted_low, predicted_close);

                /* Compute features from predicted candle for next autoregressive step */
                feat_idx = num_available + p;

                /* Feature 0: candle_return */
                last_cr = (predicted_close - predicted_open) / predicted_open;

                /* Feature 1: candle_range */
                last_crng = (predicted_high - predicted_low) / predicted_open;

                /* Feature 2: close_return — use real price difference, not normalized */
                if (feat_idx > 0) {
                    double prev_close;
                    /* Get previous close from extended features or real data */
                    if (feat_idx < num_available) {
                        /* Previous step was real data */
                        int prev_data_row = end_row - ref_steps + 1 + (feat_idx - (num_available - ref_steps));
                        if (prev_data_row > 0 && prev_data_row < num_raw) {
                            prev_close = raw_data[prev_data_row].close;
                        } else {
                            prev_close = raw_data[end_row].close;
                        }
                    } else {
                        /* Previous step was predicted — use prev_predicted_close */
                        prev_close = prev_predicted_close;
                    }
                    if (prev_close > 1e-10) {
                        last_cret = (predicted_close - prev_close) / prev_close;
                    } else {
                        last_cret = 0.0;
                    }
                } else {
                    last_cret = 0.0;
                }

                /* Feature 3: volume_return — approximate (no volume in autoregressive) */
                last_vol_ret = 0.0;

                /* Feature 4: ma_diff — approximate using recent prices */
                last_ma_diff = 0.0;
                if (feat_idx >= 20) {
                    double sum_short = 0.0, sum_long = 0.0;
                    int kk;
                    /* Use last 5 predicted/real closes */
                    for (kk = 0; kk < 5; kk++) {
                        int fidx = feat_idx - kk;
                        if (fidx < num_available) {
                            int drow = end_row - ref_steps + 1 + (fidx - (num_available - ref_steps));
                            if (drow > 0 && drow < num_raw) {
                                sum_short += raw_data[drow].close;
                            }
                        } else {
                            sum_short += prev_predicted_close;
                        }
                    }
                    sum_short /= 5.0;
                    /* Use last 20 predicted/real closes */
                    for (kk = 0; kk < 20; kk++) {
                        int fidx = feat_idx - kk;
                        if (fidx < num_available) {
                            int drow = end_row - ref_steps + 1 + (fidx - (num_available - ref_steps));
                            if (drow > 0 && drow < num_raw) {
                                sum_long += raw_data[drow].close;
                            }
                        } else {
                            sum_long += prev_predicted_close;
                        }
                    }
                    sum_long /= 20.0;
                    if (sum_long > 1e-10) {
                        last_ma_diff = (sum_short - sum_long) / sum_long;
                    }
                }

                /* Feature 5: volatility — approximate using recent returns */
                last_vol = 0.0;

                /* Normalize features using same params as real data */
                extended_features[feat_idx][0] = last_cr;
                extended_features[feat_idx][1] = last_crng;
                extended_features[feat_idx][2] = last_cret;
                extended_features[feat_idx][3] = last_vol_ret;
                extended_features[feat_idx][4] = last_ma_diff;
                extended_features[feat_idx][5] = last_vol;

                for (fi = 0; fi < num_features; fi++) {
                    double r = feat_max[fi] - feat_min[fi];
                    if (r < 1e-15) r = 1.0;
                    extended_features[feat_idx][fi] = (extended_features[feat_idx][fi] - feat_min[fi]) / r;
                }
            }

            /* Cleanup extended features */
            for (f = 0; f < extended_count; f++) utils_safe_free(extended_features[f]);
            utils_safe_free(extended_features);
        } /* end predict loop */

        /* Cleanup */
        for (f = 0; f < num_available; f++) utils_safe_free(features[f]);
        utils_safe_free(features);
        utils_safe_free(seq_buf);
        utils_safe_free(raw_data);
        lstm_free(pred_network);
        return 0;
    }

    /* ==================== TRAIN MODE ==================== */
    printf("========================================\n");
    printf("LSTM Forex Predictor (C90)\n");
    printf("========================================\n");
    printf("Parameters:\n");
    printf("  LSTM cells:       %d\n", num_lstm_cells);
    printf("  Epochs:           %d\n", num_epochs);
    printf("  Learning rate:    %.6f\n", learning_rate);
    printf("  Sequence length:  %d\n", sequence_length);
    printf("  Batch size:       %d\n", batch_size);
    printf("  Momentum:         %.2f\n", momentum);
    printf("  Seed:             %u\n", seed);
    printf("========================================\n\n");

    /* Step 1: Load raw data */
    printf("[1/6] Loading data from %s ...\n", data_path);
    raw_data = NULL;
    num_raw = 0;
    if (data_load_raw(data_path, &raw_data, &num_raw) != 0) {
        fprintf(stderr, "Failed to load data. Exiting.\n");
        return 1;
    }
    printf("  Loaded %d rows.\n\n", num_raw);

    /* Step 2: Build dataset */
    printf("[2/6] Building dataset (features, labels, normalization) ...\n");
    full_dataset = data_build_dataset(raw_data, num_raw, sequence_length);
    if (!full_dataset) {
        fprintf(stderr, "Failed to build dataset. Exiting.\n");
        utils_safe_free(raw_data);
        return 1;
    }
    printf("  Samples: %d\n", full_dataset->num_samples);
    printf("  Features: %d\n", full_dataset->num_features);
    printf("  Feature names:\n");
    for (j = 0; j < full_dataset->num_features; j++) {
        printf("    [%d] %s (min=%.6f, max=%.6f)\n",
               j, full_dataset->feat_names[j],
               full_dataset->feat_min[j],
               full_dataset->feat_max[j]);
    }
    printf("\n");

    /* Step 3: Normalize */
    printf("[3/6] Normalizing features ...\n");
    data_normalize(full_dataset);
    printf("  Normalization complete.\n\n");

    /* Step 4: Split train/test */
    printf("[4/6] Splitting train/test (80/20) ...\n");
    split = data_split_train_test(full_dataset, 0.8);
    train = split.train;
    test = split.test;
    if (!train || !test) {
        fprintf(stderr, "Failed to split data. Exiting.\n");
        data_free_dataset(full_dataset);
        utils_safe_free(raw_data);
        return 1;
    }
    printf("  Train: %d samples\n", train->num_samples);
    printf("  Test:  %d samples\n\n", test->num_samples);

    /* Step 5: Initialize network */
    printf("[5/6] Initializing LSTM network ...\n");
    network = lstm_init(num_lstm_cells, full_dataset->num_features, seed);
    printf("  Network initialized with %d cells, %d features.\n\n",
           num_lstm_cells, full_dataset->num_features);

    /* Set global variables for signal handler */
    g_saved_network = network;
    g_model_path = model_path;
    g_full_dataset = full_dataset;
    g_best_train_mse = 1e30;
    g_best_test_accuracy = 0.0;

    /* Step 6: Train */
    printf("[6/6] Training for %d epochs ...\n", num_epochs);
    printf("========================================\n");

    best_train_loss = 1e30;
    best_test_accuracy = 0.0;

    /* Sync globals */
    g_epochs_trained = 0.0;
    g_best_train_mse = 1e30;
    g_best_test_accuracy = 0.0;

    train_indices = (int *)utils_safe_calloc((size_t)train->num_samples, sizeof(int));
    for (epoch = 0; epoch < train->num_samples; epoch++) {
        train_indices[epoch] = epoch;
    }

    for (epoch = 0; epoch < num_epochs; epoch++) {
        shuffle_indices(train_indices, train->num_samples, (unsigned int)(seed + epoch));

        for (sample = 0; sample < train->num_samples; sample += batch_size) {
            batch_end = sample + batch_size;
            if (batch_end > train->num_samples) batch_end = train->num_samples;
            actual_batch = batch_end - sample;

            /* Allocate cell_states and per-timestep buffers ONCE per batch */
            cell_states = (LSTMCellState *)utils_safe_calloc(
                (size_t)sequence_length, sizeof(LSTMCellState));
            {
                int alloc_t;
                int lstm_cells = num_lstm_cells;
                int lstm_combined = num_lstm_cells + full_dataset->num_features;
                for (alloc_t = 0; alloc_t < sequence_length; alloc_t++) {
                    cell_states[alloc_t].hidden_state = (double *)utils_safe_calloc((size_t)lstm_cells, sizeof(double));
                    cell_states[alloc_t].cell_state = (double *)utils_safe_calloc((size_t)lstm_cells, sizeof(double));
                    cell_states[alloc_t].forget_gate = (double *)utils_safe_calloc((size_t)lstm_cells, sizeof(double));
                    cell_states[alloc_t].input_gate = (double *)utils_safe_calloc((size_t)lstm_cells, sizeof(double));
                    cell_states[alloc_t].output_gate = (double *)utils_safe_calloc((size_t)lstm_cells, sizeof(double));
                    cell_states[alloc_t].candidate = (double *)utils_safe_calloc((size_t)lstm_cells, sizeof(double));
                    cell_states[alloc_t].concatenated = (double *)utils_safe_calloc((size_t)lstm_combined, sizeof(double));
                }
            }

            /* Zero gradients ONCE before accumulating */
            lstm_zero_gradients(network);

            for (b = 0; b < actual_batch; b++) {
                idx = train_indices[sample + b];
                pred = lstm_forward(network, train->features[idx],
                                    cell_states, sequence_length);
                lstm_backward(network, train->features[idx],
                              cell_states, pred, train->labels[idx], sequence_length);
            }

            lstm_update(network, learning_rate, momentum, actual_batch);
            batch_loss_sum = 0.0;
            for (b = 0; b < actual_batch; b++) {
                idx = train_indices[sample + b];
                pred = lstm_forward(network, train->features[idx],
                                    cell_states, sequence_length);
                /* MSE loss for regression */
                batch_loss_sum += 0.5 * (pred - train->labels[idx]) * (pred - train->labels[idx]);
            }
            batch_loss_avg = batch_loss_sum / (double)actual_batch;
            printf("loss=%.8f ", batch_loss_avg);
            fflush(stdout);

            /* Free cell_states buffers and array after each batch */
            lstm_free_cell_state_buffers(cell_states, sequence_length);
            free(cell_states);
        }

        train_loss = lstm_compute_loss(network, train);
        train_acc = lstm_compute_accuracy(network, train);
        test_loss = lstm_compute_loss(network, test);
        test_acc = lstm_compute_accuracy(network, test);

        if (train_loss < best_train_loss) {
            best_train_loss = train_loss;
            best_test_accuracy = test_acc;
            /* Sync globals for signal handler */
            g_epochs_trained = (double)(epoch + 1);
            g_best_train_mse = best_train_loss;
            g_best_test_accuracy = best_test_accuracy;
        }

        /* Progress bar */
        bar_width = PROGRESS_BAR_WIDTH;
        progress = (int)((double)(epoch + 1) / (double)num_epochs * (double)bar_width);
        printf("Epoch %3d/%d [", epoch + 1, num_epochs);
        for (i = 0; i < bar_width; i++) {
            if (i < progress) printf("# ");
            else if (i == progress) printf("> ");
            else printf("  ");
        }
        printf("] %.1f%% | loss: %.8f | acc: %.2f%% | test_acc: %.2f%%\n",
               (double)(epoch + 1) / (double)num_epochs * 100.0,
               train_loss, train_acc * 100.0, test_acc * 100.0);
        fflush(stdout);

        /* Save model every epoch */
        model_save(model_path, network, (double)(epoch + 1),
                  train_loss, test_acc, full_dataset);
        fflush(stdout);  /* Ensure model is flushed to disk */
    }

    utils_safe_free(train_indices);

    printf("========================================\n");
    printf("Training complete!\n");
    printf("Best train MSE:      %.6f\n", best_train_loss);
    printf("Best test direction accuracy: %.4f\n", best_test_accuracy);
    printf("\n");

    /* Demo: predict on last few candles */
    printf("Demo: Predicting on last 5 sequences ...\n");
    demo_start = full_dataset->num_samples - 5;
    if (demo_start < 0) demo_start = 0;
    for (d = demo_start; d < full_dataset->num_samples; d++) {
        pred_demo = lstm_predict(network, full_dataset->features[d], sequence_length);
        actual = (int)(full_dataset->labels[d] * 10000); /* scale to basis points */
        predicted = (int)(pred_demo * 10000);
        printf("  Sample %d: actual=%.4f%%, predicted=%.4f%%\n",
               d, full_dataset->labels[d] * 100.0, pred_demo * 100.0);
    }

    /* Save final model */
    printf("\nSaving final model to %s ...\n", model_path);
    model_save(model_path, network, (double)num_epochs,
               best_train_loss, best_test_accuracy, full_dataset);

    /* Cleanup */
    g_saved_network = NULL;
    g_full_dataset = NULL;
    lstm_free(network);
    data_free_split(&split);
    data_free_dataset(full_dataset);
    utils_safe_free(raw_data);

    printf("\nDone.\n");
    return 0;
}
