#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include "include/utils.h"
#include "include/data.h"
#include "include/ohlc/data_ohlc.h"
#include "include/direction/data_direction.h"
#include "include/lstm.h"
#include "include/model.h"
#include "include/ohlc/predict_ohlc.h"
#include "include/direction/predict_direction.h"
#include "include/types.h"
#include "include/params.h"

/* ==================== Global for signal handler ==================== */

static LSTMNetwork *g_saved_network = NULL;
static const char *g_model_path = NULL;
static double g_epochs_trained = 0.0;
static double g_best_train_mse = 0.0;
static double g_best_test_loss = 0.0;
static double g_best_test_accuracy = 0.0;
static Dataset *g_full_dataset = NULL;

static void signal_handler(int signum) {
    (void)signum;
    printf("\n\nSignal received, saving model...\n");
    if (g_saved_network != NULL && g_model_path != NULL) {
        model_save(g_model_path, g_saved_network,
                   g_epochs_trained, g_best_train_mse,
                   g_best_test_loss, g_best_test_accuracy, g_full_dataset);
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
#define DEFAULT_LEARNING_RATE 0.005
#define DEFAULT_SEQUENCE_LENGTH 64
#define DEFAULT_BATCH_SIZE 32
#define DEFAULT_MOMENTUM 0.9
#define DEFAULT_SEED 42
#define PROGRESS_BAR_WIDTH 30

/* ==================== Currency pair extraction ==================== */

static void extract_currency_pair(const char *data_path, char *pair_out) {
    const char *base;
    int i;
    base = data_path;
    for (i = 0; data_path[i]; i++) {
        if (data_path[i] == '/' || data_path[i] == '\\')
            base = data_path + i + 1;
    }
    for (i = 0; base[i] && base[i] != '_' && i < 7; i++)
        pair_out[i] = base[i];
    pair_out[i] = 0;
}

/* ==================== DateTime to row search ==================== */

static int find_row_by_datetime(DataRow *raw, int num_raw, const char *datetime_str) {
    char date_part[16];
    char time_part[16];
    char target_dt[32];
    int i;
    const char *t_pos;
    int lo, hi, mid, cmp;
    char row_dt[32];

    t_pos = strchr(datetime_str, 'T');
    if (!t_pos) return -1;

    {
        int dlen = (int)(t_pos - datetime_str);
        if (dlen >= 16) dlen = 15;
        memcpy(date_part, datetime_str, (size_t)dlen);
        date_part[dlen] = 0;
    }
    {
        int tlen = (int)strlen(t_pos + 1);
        if (tlen >= 16) tlen = 15;
        memcpy(time_part, t_pos + 1, (size_t)tlen);
        time_part[tlen] = 0;
    }

    snprintf(target_dt, sizeof(target_dt), "%s %s", date_part, time_part);

    /* Binary search for insertion point: first row >= target */
    lo = 0;
    hi = num_raw;
    while (lo < hi) {
        mid = (lo + hi) / 2;
        snprintf(row_dt, sizeof(row_dt), "%s %s", raw[mid].date, raw[mid].time_);
        if (strcmp(row_dt, target_dt) < 0)
            lo = mid + 1;
        else
            hi = mid;
    }

    /* lo is the first row >= target; lo-1 is the last row < target */
    if (lo < num_raw) {
        snprintf(row_dt, sizeof(row_dt), "%s %s", raw[lo].date, raw[lo].time_);
        if (strcmp(row_dt, target_dt) == 0)
            return lo;  /* exact match */
    }

    /* Compare neighbors lo-1 and lo, pick closest */
    if (lo == 0)
        return (num_raw > 0) ? 0 : -1;
    if (lo >= num_raw)
        return num_raw - 1;

    /* Both lo-1 and lo are valid; compare string distances */
    {
        char prev_dt[32];
        int cmp_prev, cmp_next;
        snprintf(prev_dt, sizeof(prev_dt), "%s %s", raw[lo - 1].date, raw[lo - 1].time_);
        snprintf(row_dt, sizeof(row_dt), "%s %s", raw[lo].date, raw[lo].time_);
        cmp_prev = strcmp(target_dt, prev_dt);  /* > 0 since prev < target */
        cmp_next = strcmp(row_dt, target_dt);   /* > 0 since next >= target */
        if (cmp_prev <= cmp_next)
            return lo - 1;
        else
            return lo;
    }
}

/* ==================== DateTime shift (in utils.c) ==================== */

/* ==================== Secondary data helpers ==================== */

static int prepare_secondary(const char *sdata_path, int h_period,
                             DataRow **io_raw, int *io_num_raw,
                             double ***out_sec_feat, int *out_sec_nfeat,
                             double **out_sec_min, double **out_sec_max,
                             char ***out_sec_names) {
    DataRow *sec_raw = NULL, *new_pri = NULL, *new_sec = NULL;
    int sec_count = 0, new_n = 0;
    double **sec_feat;
    double *sec_min, *sec_max;
    char **sec_names;
    int j;

    *out_sec_feat = NULL; *out_sec_nfeat = 0;
    *out_sec_min = NULL; *out_sec_max = NULL; *out_sec_names = NULL;

    if (data_load_raw(sdata_path, &sec_raw, &sec_count) != 0) {
        fprintf(stderr, "Failed to load secondary data: %s\n", sdata_path);
        return -1;
    }

    if (h_period > 1) {
        DataRow *agg; int agg_count;
        data_aggregate_raw(sec_raw, sec_count, h_period, &agg, &agg_count);
        utils_safe_free(sec_raw);
        sec_raw = agg;
        sec_count = agg_count;
    }

    data_intersect_raw(*io_raw, *io_num_raw, sec_raw, sec_count,
                       &new_pri, &new_sec, &new_n);
    utils_safe_free(sec_raw);

    if (new_n == 0) {
        fprintf(stderr, "Error: --sdata intersection with primary is empty\n");
        utils_safe_free(new_pri);
        utils_safe_free(new_sec);
        return -1;
    }

    utils_safe_free(*io_raw);
    *io_raw = new_pri;
    *io_num_raw = new_n;

    sec_min = (double *)utils_safe_calloc((size_t)NUM_BASE_FEATURES, sizeof(double));
    sec_max = (double *)utils_safe_calloc((size_t)NUM_BASE_FEATURES, sizeof(double));
    sec_names = (char **)utils_safe_calloc((size_t)NUM_BASE_FEATURES, sizeof(char *));
    sec_feat = data_compute_raw_features(new_sec, new_n, NUM_BASE_FEATURES,
                                          sec_min, sec_max, sec_names);
    utils_safe_free(new_sec);

    for (j = 0; j < NUM_BASE_FEATURES; j++) {
        char tmp[40];
        snprintf(tmp, sizeof(tmp), "sec_%s", sec_names[j]);
        free(sec_names[j]);
        sec_names[j] = utils_strdup(tmp);
    }

    *out_sec_feat = sec_feat;
    *out_sec_nfeat = NUM_BASE_FEATURES;
    *out_sec_min = sec_min;
    *out_sec_max = sec_max;
    *out_sec_names = sec_names;
    return 0;
}

static void free_secondary(double **sec_feat, int num_raw,
                           double *sec_min, double *sec_max,
                           char **sec_names, int nfeat) {
    int j;
    if (sec_feat) data_free_raw_features(sec_feat, num_raw);
    if (sec_names) {
        for (j = 0; j < nfeat; j++) utils_safe_free(sec_names[j]);
        utils_safe_free(sec_names);
    }
    utils_safe_free(sec_min);
    utils_safe_free(sec_max);
}

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
    double best_test_loss;
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
    int output_count;
    int is_prediction;
    int is_realdata;
    char datetime_str[32];
    int has_datetime;
    char from_str[32];
    int has_from;
    char to_str[32];
    int has_to;
    char currency_pair[8];
    double batch_loss_sum;
    double batch_loss_avg;
    int bar_width;
    int progress;
    double *preds;
    int o;
    int predict_count;
    int num_outputs;
    char model_path_buf[256];
    int model_path_overridden;
    const char *mode_str;
    int mode;
    int h_period;
    double weight_decay;
    int patience;
    int patience_counter;
    const char *sdata_path;
    const char *params_path;
    ParamsFile pf;
    double param_clip_min, param_clip_max, param_forget_bias_init;

    /* Parse CLI arguments */
    num_lstm_cells = DEFAULT_NUM_LSTM_CELLS;
    num_epochs = DEFAULT_NUM_EPOCHS;
    learning_rate = DEFAULT_LEARNING_RATE;
    sequence_length = DEFAULT_SEQUENCE_LENGTH;
    batch_size = DEFAULT_BATCH_SIZE;
    momentum = DEFAULT_MOMENTUM;
    seed = DEFAULT_SEED;
    output_count = 1;
    is_prediction = 0;
    is_realdata = 0;
    has_datetime = 0;
    datetime_str[0] = 0;
    has_from = 0;
    from_str[0] = 0;
    has_to = 0;
    to_str[0] = 0;
    data_path = "../resources/EURUSD_H1_201001040000_202611022100.csv";
    model_path = NULL;
    predict_count = DEFAULT_OHLC_PREDICT_COUNT;
    model_path_overridden = 0;
    h_period = 1;
    weight_decay = 1e-6;
    patience = 0;
    patience_counter = 0;
    sdata_path = NULL;
    params_path = NULL;
    param_clip_min = CLIP_MIN;
    param_clip_max = CLIP_MAX;
    param_forget_bias_init = 1.0;
    mode_str = "ohlc";
    mode = MODE_OHLC;

    /* First pass: find --parameters file and load it before other CLI args */
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--parameters") == 0 && i + 1 < argc) {
            params_path = argv[i + 1];
            i++;
        }
    }

    if (params_path) {
        parse_parameters_file(params_path, &pf);
        if (pf.has_cells) num_lstm_cells = pf.cells;
        if (pf.has_epochs) num_epochs = pf.epochs;
        if (pf.has_lr) learning_rate = pf.lr;
        if (pf.has_seq) sequence_length = pf.seq;
        if (pf.has_output) output_count = pf.output;
        if (pf.has_datetime) { strcpy(datetime_str, pf.datetime); has_datetime = 1; }
        if (pf.has_from) { strcpy(from_str, pf.from); has_from = 1; }
        if (pf.has_to) { strcpy(to_str, pf.to); has_to = 1; }
        if (pf.has_data) data_path = pf.data;
        if (pf.has_model) { model_path = pf.model; model_path_overridden = 1; }
        if (pf.has_h) h_period = pf.h;
        if (pf.has_wd) weight_decay = pf.wd;
        if (pf.has_patience) patience = pf.patience;
        if (pf.has_sdata) sdata_path = pf.sdata;
        if (pf.has_batch_size) batch_size = pf.batch_size;
        if (pf.has_momentum) momentum = pf.momentum;
        if (pf.has_seed) seed = pf.seed;
        if (pf.has_clip_min) param_clip_min = pf.clip_min;
        if (pf.has_clip_max) param_clip_max = pf.clip_max;
        if (pf.has_train_ratio) g_train_ratio = pf.train_ratio;
        if (pf.has_volume_return_clip) g_volume_return_clip = pf.volume_return_clip;
        if (pf.has_forget_gate_bias_init) param_forget_bias_init = pf.forget_gate_bias_init;
        if (pf.has_mode) {
            mode_str = pf.mode;
            if (strcmp(mode_str, "direction") == 0) {
                mode = MODE_DIRECTION; is_prediction = 0;
            } else if (strcmp(mode_str, "directionpredict") == 0) {
                mode = MODE_DIRECTION; mode_str = "direction"; is_prediction = 1;
            } else if (strcmp(mode_str, "ohlcpredict") == 0) {
                mode = MODE_OHLC; mode_str = "ohlc"; is_prediction = 1;
            } else if (strcmp(mode_str, "realdata") == 0) {
                mode = MODE_OHLC; mode_str = "ohlc"; is_realdata = 1;
            } else {
                mode = MODE_OHLC; is_prediction = 0;
            }
        }
    }

    /* Allow overriding data_path and model_path via CLI */
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[i + 1];
            model_path_overridden = 1;
            i++;
        }
    }

    /* Register signal handler for graceful shutdown */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cells") == 0 && i + 1 < argc) {
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
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_count = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--datetime") == 0 && i + 1 < argc) {
            strncpy(datetime_str, argv[i + 1], sizeof(datetime_str) - 1);
            datetime_str[sizeof(datetime_str) - 1] = 0;
            has_datetime = 1;
            i++;
        } else if (strcmp(argv[i], "--from") == 0 && i + 1 < argc) {
            strncpy(from_str, argv[i + 1], sizeof(from_str) - 1);
            from_str[sizeof(from_str) - 1] = 0;
            has_from = 1;
            i++;
        } else if (strcmp(argv[i], "--to") == 0 && i + 1 < argc) {
            strncpy(to_str, argv[i + 1], sizeof(to_str) - 1);
            to_str[sizeof(to_str) - 1] = 0;
            has_to = 1;
            i++;
        } else if (strcmp(argv[i], "--h") == 0 && i + 1 < argc) {
            h_period = atoi(argv[i + 1]);
            if (h_period < 1) h_period = 1;
            i++;
        } else if (strcmp(argv[i], "--wd") == 0 && i + 1 < argc) {
            weight_decay = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--patience") == 0 && i + 1 < argc) {
            patience = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--sdata") == 0 && i + 1 < argc) {
            sdata_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode_str = argv[i + 1];
            if (strcmp(mode_str, "direction") == 0) {
                mode = MODE_DIRECTION;
                is_prediction = 0;
            } else if (strcmp(mode_str, "directionpredict") == 0) {
                mode = MODE_DIRECTION;
                mode_str = "direction";
                is_prediction = 1;
            } else if (strcmp(mode_str, "ohlcpredict") == 0) {
                mode = MODE_OHLC;
                mode_str = "ohlc";
                is_prediction = 1;
            } else if (strcmp(mode_str, "realdata") == 0) {
                mode = MODE_OHLC;
                mode_str = "ohlc";
                is_realdata = 1;
            } else {
                mode = MODE_OHLC;
                is_prediction = 0;
            }
            i++;
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[i + 1];
            model_path_overridden = 1;
            i++;
        }
    }

    /* Compute derived values based on mode */
    predict_count = output_count;
    if (mode == MODE_OHLC) {
        num_outputs = predict_count * OUTPUTS_PER_CANDLE;
    } else {
        num_outputs = predict_count;
    }

    /* Extract currency pair from data filename */
    extract_currency_pair(data_path, currency_pair);

    /* Build default model path if not overridden */
    if (!model_path_overridden) {
        snprintf(model_path_buf, sizeof(model_path_buf),
                 "../resources/model_%s_%s_cells_%d_seq_%d_count_%d%s.txt",
                 currency_pair, mode_str, num_lstm_cells, sequence_length, predict_count,
                 sdata_path ? "_sdata" : "");
        model_path = model_path_buf;
    }

    /* ==================== REAL DATA MODE ==================== */
    if (is_realdata) {
        int from_row, to_row, k;

        if (sdata_path)
            fprintf(stderr, "Note: --sdata is ignored in realdata mode.\n");

        /* Load data */
        raw_data = NULL;
        num_raw = 0;
        if (data_load_raw(data_path, &raw_data, &num_raw) != 0) {
            fprintf(stderr, "Failed to load data.\n");
            return 1;
        }
        if (h_period > 1) {
            DataRow *agg; int agg_count;
            data_aggregate_raw(raw_data, num_raw, h_period, &agg, &agg_count);
            utils_safe_free(raw_data);
            raw_data = agg;
            num_raw = agg_count;
        }

        from_row = 0;
        to_row = num_raw - 1;

        if (has_from) {
            char shifted_from[32];
            utils_shift_datetime_back(from_str, h_period, shifted_from, sizeof(shifted_from));
            from_row = find_row_by_datetime(raw_data, num_raw, shifted_from);
            if (from_row < 0) {
                fprintf(stderr, "Error: no data available\n");
                utils_safe_free(raw_data);
                return 1;
            }
        }
        if (has_to) {
            char shifted_to[32];
            utils_shift_datetime_back(to_str, h_period, shifted_to, sizeof(shifted_to));
            to_row = find_row_by_datetime(raw_data, num_raw, shifted_to);
            if (to_row < 0) {
                fprintf(stderr, "Error: no data available\n");
                utils_safe_free(raw_data);
                return 1;
            }
        }

        if (from_row > to_row) {
            fprintf(stderr, "Error: --from is after --to\n");
            utils_safe_free(raw_data);
            return 1;
        }

        for (k = from_row; k <= to_row; k++) {
            printf("%sT%s %.5f %.5f %.5f %.5f %d\n",
                   raw_data[k].date, raw_data[k].time_,
                   raw_data[k].open, raw_data[k].high,
                   raw_data[k].low, raw_data[k].close,
                   raw_data[k].tickvol);
        }
        utils_safe_free(raw_data);
        return 0;
    }

    /* ==================== PREDICTION MODE ==================== */
    if (is_prediction) {
        int seq_len, num_features, end_row;
        int fi, s, norm_count;
        double *seq_buf;
        double *pred_outputs;
        double *feat_min_arr;
        double *feat_max_arr;
        double *lbl_min_arr;
        double *lbl_max_arr;
        double **features;
        LSTMNetwork *pred_network;
        char mline[8192];
        double ref_close;

        if (!has_datetime) {
            fprintf(stderr, "Error: --datetime is required for prediction modes\n");
            return 1;
        }

        seq_len = sequence_length;
        num_features = NUM_RAW_FEATURES + (sdata_path ? NUM_BASE_FEATURES : 0);
        norm_count = 0;

        /* Load data */
        raw_data = NULL;
        num_raw = 0;
        if (data_load_raw(data_path, &raw_data, &num_raw) != 0) {
            fprintf(stderr, "Failed to load data.\n");
            return 1;
        }
        if (h_period > 1) {
            DataRow *agg; int agg_count;
            data_aggregate_raw(raw_data, num_raw, h_period, &agg, &agg_count);
            utils_safe_free(raw_data);
            raw_data = agg;
            num_raw = agg_count;
        }

        /* Prepare secondary data if --sdata specified */
        {
            double **sec_feat_p = NULL; int sec_nfeat_p = 0, sec_nrows_p = 0;
            double *sec_min_p = NULL, *sec_max_p = NULL; char **sec_names_p = NULL;
            if (sdata_path) {
                if (prepare_secondary(sdata_path, h_period, &raw_data, &num_raw,
                                      &sec_feat_p, &sec_nfeat_p,
                                      &sec_min_p, &sec_max_p, &sec_names_p) != 0) {
                    fprintf(stderr, "Failed to prepare secondary data.\n");
                    utils_safe_free(raw_data);
                    return 1;
                }
                sec_nrows_p = num_raw;
            }

        /* Find row by datetime (nearest match if exact not found).
         * Shift back by h_period hours so --datetime refers to the close
         * of the last fully formed candle, not its open time. */
        {
            char shifted_dt[32];
            utils_shift_datetime_back(datetime_str, h_period, shifted_dt, sizeof(shifted_dt));
            end_row = find_row_by_datetime(raw_data, num_raw, shifted_dt);
        }
        if (end_row < 0) {
            fprintf(stderr, "Error: no data available\n");
            utils_safe_free(raw_data);
            return 1;
        }

        /* Validate predict_count against model file */
        {
            FILE *mf = fopen(model_path, "r");
            int model_ohlc_count = -1;
            char vline[512];
            if (mf) {
                while (fgets(vline, sizeof(vline), mf)) {
                    utils_trim(vline);
                    if (strncmp(vline, "# predict_count:", strlen("# predict_count:")) == 0) {
                        model_ohlc_count = atoi(vline + strlen("# predict_count:"));
                        break;
                    }
                }
                fclose(mf);
            }
            if (model_ohlc_count != predict_count) {
                fprintf(stderr, "Error: predict_count %d does not match model's predict_count %d\n",
                        predict_count, model_ohlc_count);
                utils_safe_free(raw_data);
                return 1;
            }
        }

        /* Validate mode against model file */
        {
            FILE *mf = fopen(model_path, "r");
            char model_mode_str[32] = "";
            char vline2[512];
            if (mf) {
                while (fgets(vline2, sizeof(vline2), mf)) {
                    utils_trim(vline2);
                    if (strncmp(vline2, "# mode:", 7) == 0) {
                        char *p = vline2 + 7;
                        while (*p == ' ') p++;
                        strncpy(model_mode_str, p, sizeof(model_mode_str) - 1);
                        model_mode_str[sizeof(model_mode_str) - 1] = 0;
                        break;
                    }
                }
                fclose(mf);
            }
            if (model_mode_str[0] != 0 && strcmp(model_mode_str, mode_str) != 0) {
                fprintf(stderr, "Error: --mode %s does not match model's mode %s\n", mode_str, model_mode_str);
                utils_safe_free(raw_data);
                return 1;
            }
        }

        /* Validate seq_len against model file */
        {
            FILE *mf = fopen(model_path, "r");
            int model_seq_len = -1;
            char vline3[512];
            if (mf) {
                while (fgets(vline3, sizeof(vline3), mf)) {
                    utils_trim(vline3);
                    if (strncmp(vline3, "# seq_len:", strlen("# seq_len:")) == 0) {
                        model_seq_len = atoi(vline3 + strlen("# seq_len:"));
                        break;
                    }
                }
                fclose(mf);
            }
            if (model_seq_len >= 0 && model_seq_len != seq_len) {
                fprintf(stderr, "Error: --seq %d does not match model's seq_len %d\n", seq_len, model_seq_len);
                utils_safe_free(raw_data);
                return 1;
            }
        }

        /* Load model */
        pred_network = lstm_init(num_lstm_cells, num_features, num_outputs, seed);
        /* Apply configurable hardcoded parameters */
        pred_network->clip_min = param_clip_min;
        pred_network->clip_max = param_clip_max;
        pred_network->forget_gate_bias_init = param_forget_bias_init;
        /* Direction uses sigmoid output + BCE loss */
        if (mode == MODE_DIRECTION)
            pred_network->output_activation = ACTIVATION_SIGMOID;
        if (model_load(model_path, pred_network) != 0) {
            fprintf(stderr, "Failed to load model from %s\n", model_path);
            utils_safe_free(raw_data);
            lstm_free(pred_network);
            return 1;
        }

        /* Allocate normalization arrays */
        feat_min_arr = (double *)utils_safe_calloc((size_t)num_features, sizeof(double));
        feat_max_arr = (double *)utils_safe_calloc((size_t)num_features, sizeof(double));
        lbl_min_arr = (double *)utils_safe_calloc((size_t)num_outputs, sizeof(double));
        lbl_max_arr = (double *)utils_safe_calloc((size_t)num_outputs, sizeof(double));

        /* Load normalization params from model file */
        {
            FILE *mf = fopen(model_path, "r");
            if (mf) {
                while (fgets(mline, sizeof(mline), mf)) {
                    utils_trim(mline);
                    if (strncmp(mline, "[lstm_weights]", strlen("[lstm_weights]")) == 0) break;
                    if (strncmp(mline, "feature_name:", strlen("feature_name:")) == 0) {
                        char *min_pos = strstr(mline, "min:");
                        char *max_pos = strstr(mline, "max:");
                        if (min_pos && max_pos && norm_count < num_features) {
                            feat_min_arr[norm_count] = strtod(min_pos + 4, NULL);
                            feat_max_arr[norm_count] = strtod(max_pos + 4, NULL);
                            norm_count++;
                        }
                    } else if (strncmp(mline, "label_min:", strlen("label_min:")) == 0) {
                        char *p = mline + strlen("label_min:");
                        for (o = 0; o < num_outputs && *p; o++) {
                            lbl_min_arr[o] = strtod(p, &p);
                        }
                    } else if (strncmp(mline, "label_max:", strlen("label_max:")) == 0) {
                        char *p = mline + strlen("label_max:");
                        for (o = 0; o < num_outputs && *p; o++) {
                            lbl_max_arr[o] = strtod(p, &p);
                        }
                    }
                }
                fclose(mf);
            }
        }

        if (end_row < seq_len) {
            fprintf(stderr, "Error: Need at least %d rows for sequence, have %d\n",
                    seq_len, end_row);
            utils_safe_free(raw_data);
            lstm_free(pred_network);
            utils_safe_free(feat_min_arr);
            utils_safe_free(feat_max_arr);
            utils_safe_free(lbl_min_arr);
            utils_safe_free(lbl_max_arr);
            return 1;
        }

        /* Allocate feature and sequence buffers */
        features = (double **)utils_safe_calloc((size_t)seq_len, sizeof(double *));
        for (s = 0; s < seq_len; s++) {
            features[s] = (double *)utils_safe_calloc((size_t)num_features, sizeof(double));
        }
        seq_buf = (double *)utils_safe_calloc((size_t)(seq_len * num_features), sizeof(double));
        pred_outputs = (double *)utils_safe_calloc((size_t)(num_outputs * seq_len), sizeof(double));

        /* Compute features for all bars up to end_row, then extract last seq_len */
        {
            double *temp_feat_min = (double *)utils_safe_calloc((size_t)num_features, sizeof(double));
            double *temp_feat_max = (double *)utils_safe_calloc((size_t)num_features, sizeof(double));
            char **temp_feat_names = (char **)utils_safe_calloc((size_t)num_features, sizeof(char *));
            double **all_features;
            int fi2;

            all_features = data_compute_raw_features(raw_data, end_row + 1, NUM_RAW_FEATURES,
                                                      temp_feat_min, temp_feat_max, temp_feat_names);

            if (sec_feat_p) {
                double **combined = (double **)utils_safe_malloc((size_t)(end_row + 1) * sizeof(double *));
                int rr2, cc2;
                for (rr2 = 0; rr2 <= end_row; rr2++) {
                    combined[rr2] = (double *)utils_safe_malloc((size_t)num_features * sizeof(double));
                    for (cc2 = 0; cc2 < NUM_RAW_FEATURES; cc2++)
                        combined[rr2][cc2] = all_features[rr2][cc2];
                    for (cc2 = 0; cc2 < NUM_BASE_FEATURES; cc2++)
                        combined[rr2][NUM_RAW_FEATURES + cc2] = sec_feat_p[rr2][cc2];
                }
                data_free_raw_features(all_features, end_row + 1);
                all_features = combined;
            }

            for (s = 0; s < seq_len; s++) {
                int pi = end_row - seq_len + 1 + s;
                for (fi2 = 0; fi2 < num_features; fi2++) {
                    double r = feat_max_arr[fi2] - feat_min_arr[fi2];
                    if (r < 1e-15) r = 1.0;
                    features[s][fi2] = (all_features[pi][fi2] - feat_min_arr[fi2]) / r;
                }
            }

            data_free_raw_features(all_features, end_row + 1);
            for (fi2 = 0; fi2 < num_features; fi2++) free(temp_feat_names[fi2]);
            free(temp_feat_names);
            free(temp_feat_min);
            free(temp_feat_max);
        }

        /* Build flat sequence buffer */
        for (s = 0; s < seq_len; s++) {
            for (fi = 0; fi < num_features; fi++) {
                seq_buf[s * num_features + fi] = features[s][fi];
            }
        }

        /* Single forward pass */
        lstm_predict(pred_network, seq_buf, seq_len, pred_outputs);

        /* Reference close for denormalization */
        ref_close = raw_data[end_row].close;

        /* Output only prediction results */
        if (mode == MODE_OHLC) {
            predict_ohlc(pred_outputs, num_outputs, seq_len,
                         lbl_min_arr, lbl_max_arr, ref_close, predict_count,
                         raw_data, end_row);
        } else {
            predict_direction(pred_outputs, num_outputs, seq_len, predict_count,
                             raw_data, end_row);
        }

        /* Cleanup */
        for (s = 0; s < seq_len; s++) utils_safe_free(features[s]);
        utils_safe_free(features);
        utils_safe_free(seq_buf);
        utils_safe_free(pred_outputs);
        utils_safe_free(feat_min_arr);
        utils_safe_free(feat_max_arr);
        utils_safe_free(lbl_min_arr);
        utils_safe_free(lbl_max_arr);
        utils_safe_free(raw_data);
        lstm_free(pred_network);
        free_secondary(sec_feat_p, sec_nrows_p, sec_min_p, sec_max_p, sec_names_p, sec_nfeat_p);
        return 0;
        }  /* end of sdata scope */
    }

    /* ==================== TRAIN MODE ==================== */
    printf("========================================\n");
    printf("LSTM Forex Predictor (C90) - Multi-Output OHLC\n");
    printf("========================================\n");
    printf("Parameters:\n");
    printf("  LSTM cells:       %d\n", num_lstm_cells);
    printf("  Epochs:           %d\n", num_epochs);
    printf("  Learning rate:    %.6f\n", learning_rate);
    printf("  Sequence length:  %d\n", sequence_length);
    printf("  Batch size:       %d\n", batch_size);
    printf("  Momentum:         %.2f\n", momentum);
    printf("  Seed:             %u\n", seed);
    printf("  Prediction horizon: %d candles (%d outputs)\n", predict_count, num_outputs);
    printf("  Mode:              %s\n", mode_str);
    printf("  Timeframe:         H%d\n", h_period);
    printf("========================================\n\n");

    /* Step 1: Load raw data */
    printf("[1/6] Loading data from %s ...\n", data_path);
    raw_data = NULL;
    num_raw = 0;
    if (data_load_raw(data_path, &raw_data, &num_raw) != 0) {
        fprintf(stderr, "Failed to load data. Exiting.\n");
        return 1;
    }
    if (h_period > 1) {
        DataRow *agg; int agg_count;
        data_aggregate_raw(raw_data, num_raw, h_period, &agg, &agg_count);
        utils_safe_free(raw_data);
        raw_data = agg;
        num_raw = agg_count;
    }
    printf("  Loaded %d rows (H%d).\n\n", num_raw, h_period);

    /* Prepare secondary data if --sdata specified */
    {
        double **sec_feat = NULL; int sec_nfeat = 0, sec_nrows = 0;
        double *sec_min = NULL, *sec_max = NULL; char **sec_names = NULL;
        if (sdata_path) {
            if (prepare_secondary(sdata_path, h_period, &raw_data, &num_raw,
                                  &sec_feat, &sec_nfeat,
                                  &sec_min, &sec_max, &sec_names) != 0) {
                fprintf(stderr, "Failed to prepare secondary data. Exiting.\n");
                utils_safe_free(raw_data);
                return 1;
            }
            sec_nrows = num_raw;
            printf("  After --sdata intersection: %d rows, %d total features.\n\n",
                   num_raw, NUM_RAW_FEATURES + sec_nfeat);
        }

    /* Step 2: Build dataset */
    printf("[2/6] Building dataset (features, labels, normalization) ...\n");
    if (mode == MODE_OHLC) {
        full_dataset = sec_feat
            ? data_build_dataset_ohlc_ext(raw_data, num_raw, sequence_length, predict_count,
                                          sec_feat, sec_nfeat, sec_min, sec_max, sec_names)
            : data_build_dataset_ohlc(raw_data, num_raw, sequence_length, predict_count);
    } else {
        full_dataset = sec_feat
            ? data_build_dataset_direction_ext(raw_data, num_raw, sequence_length, predict_count,
                                               sec_feat, sec_nfeat, sec_min, sec_max, sec_names)
            : data_build_dataset_direction(raw_data, num_raw, sequence_length, predict_count);
    }
    if (sec_feat) {
        free_secondary(sec_feat, sec_nrows, sec_min, sec_max, sec_names, sec_nfeat);
        sec_feat = NULL;
    }
    if (!full_dataset) {
        fprintf(stderr, "Failed to build dataset. Exiting.\n");
        utils_safe_free(raw_data);
        return 1;
    }
    printf("  Samples: %d\n", full_dataset->num_samples);
    printf("  Features: %d\n", full_dataset->num_features);
    printf("  Outputs: %d\n", full_dataset->num_outputs);
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
    data_normalize_features(full_dataset);
    if (mode == MODE_OHLC) data_normalize_labels_ohlc(full_dataset);
    else data_normalize_labels_direction(full_dataset);
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
    network = lstm_init(num_lstm_cells, full_dataset->num_features, num_outputs, seed);
    /* Apply configurable hardcoded parameters */
    network->clip_min = param_clip_min;
    network->clip_max = param_clip_max;
    network->forget_gate_bias_init = param_forget_bias_init;
    /* Direction uses sigmoid output + BCE loss for binary classification */
    if (mode == MODE_DIRECTION)
        network->output_activation = ACTIVATION_SIGMOID;
    network->weight_decay = weight_decay;
    printf("  Network initialized with %d cells, %d features, %d outputs.\n",
           num_lstm_cells, full_dataset->num_features, num_outputs);
    printf("  Weight decay (L2): %.2e\n", weight_decay);
    if (patience > 0)
        printf("  Early stopping patience: %d epochs\n", patience);

    /* Set global variables for signal handler */
    g_saved_network = network;
    g_model_path = model_path;
    g_full_dataset = full_dataset;
    g_best_train_mse = 1e30;
    g_best_test_loss = 1e30;
    g_best_test_accuracy = 0.0;

    /* Step 6: Train */
    printf("[6/6] Training for %d epochs ...\n", num_epochs);
    printf("========================================\n");

    best_train_loss = 1e30;
    best_test_loss = 1e30;
    best_test_accuracy = 0.0;

    /* Sync globals */
    g_epochs_trained = 0.0;
    g_best_train_mse = 1e30;
    g_best_test_loss = 1e30;
    g_best_test_accuracy = 0.0;

    preds = (double *)utils_safe_calloc((size_t)(num_outputs * sequence_length), sizeof(double));

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

            batch_loss_sum = 0.0;
            for (b = 0; b < actual_batch; b++) {
                idx = train_indices[sample + b];
                lstm_forward(network, train->features[idx],
                            cell_states, sequence_length, preds);
                lstm_backward(network, train->features[idx],
                              cell_states, preds, train->labels[idx], sequence_length);
                for (o = 0; o < num_outputs; o++) {
                    int st;
                    for (st = 0; st < sequence_length; st++) {
                        double p = preds[st * num_outputs + o];
                        double tgt = train->labels[idx][st * num_outputs + o];
                        if (network->output_activation == ACTIVATION_SIGMOID)
                            batch_loss_sum += utils_binary_cross_entropy(p, tgt);
                        else {
                            double err = p - tgt;
                            batch_loss_sum += 0.5 * err * err;
                        }
                    }
                }
            }

            lstm_update(network, learning_rate, momentum, actual_batch);
            batch_loss_avg = batch_loss_sum / ((double)actual_batch * (double)sequence_length * (double)num_outputs);
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

        /* Save best model based on test loss */
        if (test_loss < best_test_loss) {
            best_test_loss = test_loss;
            best_train_loss = train_loss;
            best_test_accuracy = test_acc;
            patience_counter = 0;

            /* Sync globals for signal handler */
            g_epochs_trained = (double)(epoch + 1);
            g_best_train_mse = best_train_loss;
            g_best_test_loss = best_test_loss;
            g_best_test_accuracy = best_test_accuracy;

            /* Save model only when test loss improves */
            model_save(model_path, network, (double)(epoch + 1),
                      train_loss, test_loss, test_acc, full_dataset);
        } else {
            patience_counter++;
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
        printf("] %.1f%% | tr_loss: %.6f | te_loss: %.6f | te_acc: %.2f%% | best_te_loss: %.6f\n",
               (double)(epoch + 1) / (double)num_epochs * 100.0,
               train_loss, test_loss, test_acc * 100.0, best_test_loss);
        fflush(stdout);

        /* Early stopping */
        if (patience > 0 && patience_counter >= patience) {
            printf("\n*** Early stopping: test loss hasn't improved for %d epochs ***\n", patience);
            break;
        }
    }

    utils_safe_free(train_indices);
    utils_safe_free(preds);

    printf("========================================\n");
    printf("Training complete!\n");
    printf("Best train loss:       %.6f\n", best_train_loss);
    printf("Best test loss:        %.6f\n", best_test_loss);
    printf("Best test accuracy:    %.4f\n", best_test_accuracy);
    printf("\n");

    /* Demo: predict on last few candles */
    if (mode == MODE_OHLC) {
        demo_ohlc(network, full_dataset, predict_count);
    } else {
        demo_direction(network, full_dataset, predict_count);
    }

    /* Save final model */
    printf("\nSaving final model to %s ...\n", model_path);
    model_save(model_path, network, (double)num_epochs,
               best_train_loss, best_test_loss, best_test_accuracy, full_dataset);

    /* Cleanup */
    g_saved_network = NULL;
    g_full_dataset = NULL;
    lstm_free(network);
    data_free_split(&split);
    data_free_dataset(full_dataset);
    utils_safe_free(raw_data);

    printf("\nDone.\n");
    return 0;
    }  /* end of sdata scope */
}
