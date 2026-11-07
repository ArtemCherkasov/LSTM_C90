#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================== Constants ==================== */
#define INITIAL_CAPACITY 1024
#define CAPACITY_MULTIPLIER 2
#define NORMALIZATION_EPSILON 1e-15

/* Configurable parameters (overridable via parameters file) */
double g_train_ratio = 0.8;
double g_volume_return_clip = 10.0;

/* ==================== Day of Week (Zeller) ==================== */
static int day_of_week(int year, int month, int day) {
    if (month < 3) { month += 12; year--; }
    return (day + 13*(month+1)/5 + year + year/4 - year/100 + year/400) % 7;
    /* Returns: 0=Saturday, 1=Sunday, 2=Monday, ..., 6=Friday */
    /* Convert to: 0=Monday, ..., 6=Sunday */
}

/* ==================== CSV Parser ==================== */

static char *skip_quotes(char *p) {
    while (*p == '"' || *p == ' ' || *p == '\t') p++;
    return p;
}

static char *find_field_end(char *p) {
    if (*p == '"') {
        p++;
        while (*p && *p != '"') p++;
        return (*p == '"') ? p + 1 : p;
    }
    while (*p && *p != '\t' && *p != '\n' && *p != '\r') p++;
    return p;
}

int data_load_raw(const char *filepath, DataRow **out_rows, int *out_count) {
    FILE *f;
    int capacity, count;
    DataRow *rows;
    char line[4096];
    char *p, *end;
    int len;

    f = fopen(filepath, "r");
    if (!f) { fprintf(stderr, "Error: Cannot open %s\n", filepath); return -1; }

    capacity = INITIAL_CAPACITY;
    count = 0;
    rows = (DataRow *)malloc((size_t)capacity * sizeof(DataRow));
    if (!rows) { fclose(f); return -1; }

    {
        int header_skipped = 0;
        while (fgets(line, sizeof(line), f)) {
            if (line[0] == '\n' || line[0] == '\r' || line[0] == 0) continue;
            if (!header_skipped) { header_skipped = 1; continue; }

            if (count >= capacity) {
                DataRow *tmp;
                capacity *= CAPACITY_MULTIPLIER;
                tmp = (DataRow *)realloc(rows, (size_t)capacity * sizeof(DataRow));
                if (!tmp) { free(rows); fclose(f); return -1; }
                rows = tmp;
            }

            p = line;
            p = skip_quotes(p); end = find_field_end(p); len = (int)(end - p);
            if (len >= 16) len = 15; memcpy(rows[count].date, p, (size_t)len); rows[count].date[len] = 0; p = end;
            p = skip_quotes(p); end = find_field_end(p); len = (int)(end - p);
            if (len >= 16) len = 15; memcpy(rows[count].time_, p, (size_t)len); rows[count].time_[len] = 0; p = end;
            p = skip_quotes(p); rows[count].open = atof(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].high = atof(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].low = atof(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].close = atof(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].tickvol = atoi(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].vol = atoi(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].spread = atoi(p);
            count++;
        }
    }
    fclose(f);
    *out_rows = rows;
    *out_count = count;
    return 0;
}

/* ==================== Timeframe Aggregation ==================== */

/* Compute group key for time-aligned aggregation.
 * For h_period N, bars are grouped by (date, (hour / N) * N).
 * This ensures H4 bars always start at 00:00, 04:00, 08:00, etc.,
 * even after gaps (weekends, holidays). */
static void compute_group_key(const char *date, const char *time_,
                              int h_period, char *out_key) {
    int hour;
    int group_hour;
    hour = atoi(time_);
    group_hour = (hour / h_period) * h_period;
    snprintf(out_key, 32, "%s %02d", date, group_hour);
}

void data_aggregate_raw(DataRow *raw, int num_raw, int h_period,
                        DataRow **out_raw, int *out_count) {
    int i, j;
    DataRow *agg;
    int agg_count, agg_cap;
    char cur_key[32], new_key[32];
    double max_high, min_low;
    double sum_tickvol, sum_vol;
    int group_start; /* index of first bar in current group */

    if (h_period <= 1 || !raw || num_raw <= 0) {
        *out_raw = raw;
        *out_count = num_raw;
        return;
    }

    agg_cap = num_raw; /* worst case: each bar is its own group */
    agg = (DataRow *)utils_safe_malloc((size_t)agg_cap * sizeof(DataRow));
    agg_count = 0;

    compute_group_key(raw[0].date, raw[0].time_, h_period, cur_key);
    group_start = 0;
    max_high = raw[0].high;
    min_low = raw[0].low;
    sum_tickvol = (double)raw[0].tickvol;
    sum_vol = (double)raw[0].vol;

    for (i = 1; i < num_raw; i++) {
        compute_group_key(raw[i].date, raw[i].time_, h_period, new_key);

        if (strcmp(new_key, cur_key) != 0) {
            /* Finalize previous group */
            int last = i - 1;
            int g_hour = (atoi(raw[group_start].time_) / h_period) * h_period;
            strcpy(agg[agg_count].date, raw[group_start].date);
            snprintf(agg[agg_count].time_, 16, "%02d:00", g_hour);
            agg[agg_count].open = raw[group_start].open;
            agg[agg_count].high = max_high;
            agg[agg_count].low = min_low;
            agg[agg_count].close = raw[last].close;
            agg[agg_count].tickvol = (int)sum_tickvol;
            agg[agg_count].vol = (int)sum_vol;
            agg[agg_count].spread = raw[group_start].spread;
            agg_count++;

            /* Start new group */
            strcpy(cur_key, new_key);
            group_start = i;
            max_high = raw[i].high;
            min_low = raw[i].low;
            sum_tickvol = (double)raw[i].tickvol;
            sum_vol = (double)raw[i].vol;
        } else {
            /* Same group: accumulate */
            if (raw[i].high > max_high) max_high = raw[i].high;
            if (raw[i].low < min_low) min_low = raw[i].low;
            sum_tickvol += (double)raw[i].tickvol;
            sum_vol += (double)raw[i].vol;
        }
    }

    /* Finalize last group */
    {
        int last = num_raw - 1;
        int g_hour = (atoi(raw[group_start].time_) / h_period) * h_period;
        strcpy(agg[agg_count].date, raw[group_start].date);
        snprintf(agg[agg_count].time_, 16, "%02d:00", g_hour);
        agg[agg_count].open = raw[group_start].open;
        agg[agg_count].high = max_high;
        agg[agg_count].low = min_low;
        agg[agg_count].close = raw[last].close;
        agg[agg_count].tickvol = (int)sum_tickvol;
        agg[agg_count].vol = (int)sum_vol;
        agg[agg_count].spread = raw[group_start].spread;
        agg_count++;
    }

    (void)j; /* silence unused warning */
    *out_raw = agg;
    *out_count = agg_count;
}

/* ==================== Feature Engineering (common) ==================== */

double **data_compute_raw_features(DataRow *raw, int num_raw, int num_features,
                                    double *feat_min, double *feat_max,
                                    char **feat_names) {
    int i, j;
    double **raw_features;
    double cr, rng, cret, vol_ret, ma_diff, vol;
    double sum, mean, std_dev;
    double *rsi_arr, *macd_hist_arr, *bb_pos_arr, *atr_norm_arr, *stoch_k_arr;
    const char names[][32] = {
        "candle_return", "candle_range", "close_return",
        "volume_return", "ma_diff", "volatility",
        "rsi", "macd_hist", "bb_position", "atr_norm", "stoch_k",
        "hour", "dow",
        "mtf_candle_return", "mtf_candle_range", "mtf_close_return",
        "mtf_volume_return", "mtf_ma_diff", "mtf_volatility",
        "mtf_rsi", "mtf_macd_hist", "mtf_bb_position", "mtf_atr_norm", "mtf_stoch_k",
        "mtf_hour", "mtf_dow"
    };

    raw_features = (double **)malloc((size_t)num_raw * sizeof(double *));
    for (i = 0; i < num_raw; i++)
        raw_features[i] = (double *)malloc((size_t)num_features * sizeof(double));

    for (j = 0; j < num_features; j++) {
        feat_names[j] = utils_strdup(names[j]);
        feat_min[j] = 1e30;
        feat_max[j] = -1e30;
    }

    /* Pre-compute technical indicators requiring sequential state */
    rsi_arr = (double *)calloc((size_t)num_raw, sizeof(double));
    macd_hist_arr = (double *)calloc((size_t)num_raw, sizeof(double));
    bb_pos_arr = (double *)calloc((size_t)num_raw, sizeof(double));
    atr_norm_arr = (double *)calloc((size_t)num_raw, sizeof(double));
    stoch_k_arr = (double *)calloc((size_t)num_raw, sizeof(double));

    /* RSI(14) using Wilder's smoothing */
    {
        double avg_gain = 0.0, avg_loss = 0.0, gain_sum = 0.0, loss_sum = 0.0;
        int rsi_period = 14, ti;
        for (ti = 0; ti < num_raw; ti++) {
            if (ti == 0) { rsi_arr[ti] = 0.0; continue; }
            {
                double change = raw[ti].close - raw[ti-1].close;
                double gain = (change > 0.0) ? change : 0.0;
                double loss = (change < 0.0) ? -change : 0.0;
                if (ti <= rsi_period) {
                    gain_sum += gain; loss_sum += loss;
                    if (ti == rsi_period) {
                        avg_gain = gain_sum / (double)rsi_period;
                        avg_loss = loss_sum / (double)rsi_period;
                    }
                } else {
                    avg_gain = (avg_gain * (rsi_period - 1) + gain) / (double)rsi_period;
                    avg_loss = (avg_loss * (rsi_period - 1) + loss) / (double)rsi_period;
                }
                if (ti >= rsi_period) {
                    if (avg_loss < 1e-15) rsi_arr[ti] = 100.0;
                    else { double rs = avg_gain / avg_loss; rsi_arr[ti] = 100.0 - 100.0 / (1.0 + rs); }
                }
            }
        }
    }

    /* MACD histogram: (EMA12 - EMA26) - EMA9(EMA12 - EMA26) */
    {
        double ema12 = raw[0].close, ema26 = raw[0].close, signal_ema = 0.0;
        double a12 = 2.0 / 13.0, a26 = 2.0 / 27.0, a9 = 2.0 / 10.0;
        int ti;
        for (ti = 0; ti < num_raw; ti++) {
            if (ti > 0) {
                ema12 = a12 * raw[ti].close + (1.0 - a12) * ema12;
                ema26 = a26 * raw[ti].close + (1.0 - a26) * ema26;
            }
            { double macd_line = ema12 - ema26;
            if (ti == 0) signal_ema = macd_line;
            else signal_ema = a9 * macd_line + (1.0 - a9) * signal_ema;
            macd_hist_arr[ti] = macd_line - signal_ema; }
        }
    }

    /* Bollinger Band position: (close - BB_lower) / (BB_upper - BB_lower) */
    {
        int bb_period = 20, ti;
        for (ti = 0; ti < num_raw; ti++) {
            if (ti < bb_period - 1) { bb_pos_arr[ti] = 0.5; continue; }
            { double sum_bb = 0.0, var_bb = 0.0; int bi;
            for (bi = 0; bi < bb_period; bi++) sum_bb += raw[ti - bi].close;
            { double ma_bb = sum_bb / (double)bb_period;
            for (bi = 0; bi < bb_period; bi++) { double dev = raw[ti - bi].close - ma_bb; var_bb += dev * dev; }
            { double std_bb = sqrt(var_bb / (double)bb_period);
            double bb_range = 4.0 * std_bb;
            if (bb_range < 1e-15) bb_pos_arr[ti] = 0.5;
            else bb_pos_arr[ti] = (raw[ti].close - (ma_bb - 2.0 * std_bb)) / bb_range; } } }
        }
    }

    /* ATR(14) normalized by close */
    {
        double prev_atr = 0.0, tr_sum = 0.0; int atr_period = 14, ti;
        for (ti = 0; ti < num_raw; ti++) {
            double tr;
            if (ti == 0) { tr = raw[ti].high - raw[ti].low; }
            else {
                double h_l = raw[ti].high - raw[ti].low;
                double h_pc = fabs(raw[ti].high - raw[ti-1].close);
                double l_pc = fabs(raw[ti].low - raw[ti-1].close);
                tr = h_l; if (h_pc > tr) tr = h_pc; if (l_pc > tr) tr = l_pc;
            }
            if (ti < atr_period) { tr_sum += tr; if (ti == atr_period - 1) prev_atr = tr_sum / (double)atr_period; }
            else prev_atr = (prev_atr * (atr_period - 1) + tr) / (double)atr_period;
            if (ti >= atr_period - 1 && raw[ti].close > 1e-15)
                atr_norm_arr[ti] = prev_atr / raw[ti].close;
        }
    }

    /* Stochastic %K(14): (close - low_14) / (high_14 - low_14) * 100 */
    {
        int stoch_period = 14, ti;
        for (ti = 0; ti < num_raw; ti++) {
            if (ti < stoch_period - 1) { stoch_k_arr[ti] = 50.0; continue; }
            { double lo = raw[ti].low, hi = raw[ti].high; int si;
            for (si = 0; si < stoch_period; si++) {
                if (raw[ti - si].low < lo) lo = raw[ti - si].low;
                if (raw[ti - si].high > hi) hi = raw[ti - si].high;
            }
            { double range = hi - lo;
            if (range < 1e-15) stoch_k_arr[ti] = 50.0;
            else stoch_k_arr[ti] = (raw[ti].close - lo) / range * 100.0; } }
        }
    }

    for (i = 0; i < num_raw; i++) {
        cr = (raw[i].close - raw[i].open) / raw[i].open;
        rng = (raw[i].high - raw[i].low) / raw[i].open;
        cret = (i > 0) ? (raw[i].close - raw[i - 1].close) / raw[i - 1].close : 0.0;
        vol_ret = (i > 0 && raw[i - 1].tickvol > 0) ?
            (raw[i].tickvol - raw[i - 1].tickvol) / (double)raw[i - 1].tickvol : 0.0;
        if (vol_ret > g_volume_return_clip) vol_ret = g_volume_return_clip;
        if (vol_ret < -g_volume_return_clip) vol_ret = -g_volume_return_clip;
        ma_diff = 0.0;
        if (i >= 20) {
            double ss = 0.0, sl = 0.0; int kk;
            for (kk = 0; kk < 5; kk++) ss += raw[i - kk].close;
            for (kk = 0; kk < 20; kk++) sl += raw[i - kk].close;
            ma_diff = (ss / 5.0 - sl / 20.0) / (sl / 20.0);
        }
        vol = 0.0;
        if (i >= 12) {
            double sr = 0.0; int rc = 0, m;
            for (m = i - 12; m < i; m++) {
                if (m > 0) { sr += (raw[m].close - raw[m-1].close) / raw[m-1].close; rc++; }
            }
            if (rc > 0) {
                mean = sr / (double)rc; sum = 0.0;
                for (m = i - 12; m < i; m++) {
                    if (m > 0) { double r = (raw[m].close - raw[m-1].close) / raw[m-1].close; sum += (r - mean) * (r - mean); }
                }
                vol = sqrt(sum / (double)rc);
            }
        }
        raw_features[i][0] = cr; raw_features[i][1] = rng; raw_features[i][2] = cret;
        raw_features[i][3] = vol_ret; raw_features[i][4] = ma_diff; raw_features[i][5] = vol;
        raw_features[i][6]  = rsi_arr[i] / 100.0;
        raw_features[i][7]  = macd_hist_arr[i];
        raw_features[i][8]  = bb_pos_arr[i];
        raw_features[i][9]  = atr_norm_arr[i];
        raw_features[i][10] = stoch_k_arr[i] / 100.0;
        /* Temporal features: hour of day, day of week */
        {
            int hour = atoi(raw[i].time_);
            int yr, mo, dy;
            sscanf(raw[i].date, "%d.%d.%d", &yr, &mo, &dy);
            int dow_z = day_of_week(yr, mo, dy);
            int dow = (dow_z >= 2) ? (dow_z - 2) : (dow_z + 5);
            raw_features[i][11] = (double)hour / 23.0;
            raw_features[i][12] = (double)dow / 6.0;
        }
        /* Only update min/max for base features (0-12); MTF features (13-25)
         * are not yet computed at this point and raw_features[i][13..25]
         * would be uninitialized garbage if read here. */
        for (j = 0; j < NUM_BASE_FEATURES; j++) {
            if (raw_features[i][j] < feat_min[j]) feat_min[j] = raw_features[i][j];
            if (raw_features[i][j] > feat_max[j]) feat_max[j] = raw_features[i][j];
        }
    }

    /* Multi-timeframe features: aggregate raw by 2, compute 13 base features, overlay */
    if (num_features > NUM_BASE_FEATURES) {
        DataRow *mtf_raw; int mtf_count;
        double **mtf_features;
        double *mtf_min, *mtf_max;
        char **mtf_names;
        int mtf_i;

        data_aggregate_raw(raw, num_raw, 2, &mtf_raw, &mtf_count);
        mtf_min = (double *)calloc(NUM_BASE_FEATURES, sizeof(double));
        mtf_max = (double *)calloc(NUM_BASE_FEATURES, sizeof(double));
        mtf_names = (char **)calloc(NUM_BASE_FEATURES, sizeof(char *));
        mtf_features = data_compute_raw_features(mtf_raw, mtf_count, NUM_BASE_FEATURES,
                                                   mtf_min, mtf_max, mtf_names);

        for (i = 0; i < num_raw; i++) {
            int mi = i / 2;
            if (mi >= mtf_count) mi = mtf_count - 1;
            for (j = 0; j < NUM_BASE_FEATURES; j++) {
                raw_features[i][NUM_BASE_FEATURES + j] = mtf_features[mi][j];
                if (mtf_features[mi][j] < feat_min[NUM_BASE_FEATURES + j]) feat_min[NUM_BASE_FEATURES + j] = mtf_features[mi][j];
                if (mtf_features[mi][j] > feat_max[NUM_BASE_FEATURES + j]) feat_max[NUM_BASE_FEATURES + j] = mtf_features[mi][j];
            }
        }

        data_free_raw_features(mtf_features, mtf_count);
        for (mtf_i = 0; mtf_i < NUM_BASE_FEATURES; mtf_i++) free(mtf_names[mtf_i]);
        free(mtf_names);
        free(mtf_min);
        free(mtf_max);
        free(mtf_raw);
    }

    free(rsi_arr);
    free(macd_hist_arr);
    free(bb_pos_arr);
    free(atr_norm_arr);
    free(stoch_k_arr);
    return raw_features;
}

void data_free_raw_features(double **raw_features, int num_raw) {
    int i;
    for (i = 0; i < num_raw; i++) free(raw_features[i]);
    free(raw_features);
}

/* ==================== Dataset Allocation (common) ==================== */

/* ==================== Intersect raw arrays by date+time ==================== */

int data_intersect_raw(DataRow *pri, int n_pri, DataRow *sec, int n_sec,
                       DataRow **out_pri, DataRow **out_sec, int *out_n) {
    int i, j, count, cap;
    DataRow *rp, *rs;
    char key_pri[32], key_sec[32];

    *out_pri = NULL; *out_sec = NULL; *out_n = 0;
    if (n_pri <= 0 || n_sec <= 0) return 0;

    cap = (n_pri < n_sec) ? n_pri : n_sec;
    rp = (DataRow *)utils_safe_malloc((size_t)cap * sizeof(DataRow));
    rs = (DataRow *)utils_safe_malloc((size_t)cap * sizeof(DataRow));

    i = 0; j = 0; count = 0;
    while (i < n_pri && j < n_sec) {
        int cmp;
        snprintf(key_pri, sizeof(key_pri), "%s %s", pri[i].date, pri[i].time_);
        snprintf(key_sec, sizeof(key_sec), "%s %s", sec[j].date, sec[j].time_);
        cmp = strcmp(key_pri, key_sec);
        if (cmp == 0) {
            rp[count] = pri[i];
            rs[count] = sec[j];
            count++; i++; j++;
        } else if (cmp < 0) {
            i++;
        } else {
            j++;
        }
    }

    *out_pri = rp; *out_sec = rs; *out_n = count;
    return 0;
}

/* ==================== Dataset Allocation (extended) ==================== */

Dataset *data_alloc_dataset_ext(DataRow *raw, int num_raw, int seq_len,
                                int prediction_horizon, int num_outputs, int mode,
                                double **sec_features, int num_sec_features,
                                double *sec_feat_min, double *sec_feat_max,
                                char **sec_feat_names) {
    int num_primary = NUM_RAW_FEATURES;
    int num_features = num_primary + (sec_features ? num_sec_features : 0);
    int num_samples;
    Dataset *ds;
    double **raw_features;
    int i, s, fi, ti;

    num_samples = num_raw - seq_len - prediction_horizon;
    if (num_samples <= 0) {
        fprintf(stderr, "Error: Not enough data. Need at least %d rows, have %d\n",
                seq_len + prediction_horizon + 1, num_raw);
        return NULL;
    }

    ds = (Dataset *)malloc(sizeof(Dataset));
    if (!ds) return NULL;

    ds->num_samples = num_samples;
    ds->num_features = num_features;
    ds->num_outputs = num_outputs;
    ds->sequence_length = seq_len;
    ds->mode = mode;
    ds->raw_data = raw;
    ds->num_raw = num_raw;

    ds->features = (double **)utils_safe_malloc((size_t)num_samples * sizeof(double *));
    ds->labels = (double **)utils_safe_malloc((size_t)num_samples * sizeof(double *));
    ds->feat_min = (double *)utils_safe_malloc((size_t)num_features * sizeof(double));
    ds->feat_max = (double *)utils_safe_malloc((size_t)num_features * sizeof(double));
    ds->feat_names = (char **)utils_safe_malloc((size_t)num_features * sizeof(char *));
    ds->label_min = (double *)utils_safe_malloc((size_t)num_outputs * sizeof(double));
    ds->label_max = (double *)utils_safe_malloc((size_t)num_outputs * sizeof(double));

    raw_features = data_compute_raw_features(raw, num_raw, num_primary,
                                              ds->feat_min, ds->feat_max, ds->feat_names);

    if (sec_features) {
        for (i = 0; i < num_sec_features; i++) {
            ds->feat_min[num_primary + i] = sec_feat_min[i];
            ds->feat_max[num_primary + i] = sec_feat_max[i];
            ds->feat_names[num_primary + i] = utils_strdup(sec_feat_names[i]);
        }
    }

    for (i = 0; i < num_outputs; i++) {
        ds->label_min[i] = 1e30;
        ds->label_max[i] = -1e30;
    }

    for (i = 0; i < num_samples; i++) {
        ds->features[i] = (double *)utils_safe_malloc((size_t)(seq_len * num_features) * sizeof(double));
        ds->labels[i] = (double *)utils_safe_malloc((size_t)(seq_len * num_outputs) * sizeof(double));
        for (s = 0; s < seq_len; s++) {
            ti = i + s;
            for (fi = 0; fi < num_primary; fi++)
                ds->features[i][s * num_features + fi] = raw_features[ti][fi];
            if (sec_features) {
                for (fi = 0; fi < num_sec_features; fi++)
                    ds->features[i][s * num_features + num_primary + fi] = sec_features[ti][fi];
            }
        }
    }

    data_free_raw_features(raw_features, num_raw);
    return ds;
}

/* ==================== Dataset Allocation (common, wrapper) ==================== */

Dataset *data_alloc_dataset(DataRow *raw, int num_raw, int seq_len,
                             int prediction_horizon, int num_outputs, int mode) {
    return data_alloc_dataset_ext(raw, num_raw, seq_len, prediction_horizon,
                                  num_outputs, mode,
                                  NULL, 0, NULL, NULL, NULL);
}

/* ==================== Free Dataset ==================== */

void data_free_dataset(Dataset *ds) {
    int i, j;
    if (!ds) return;
    if (ds->features) { for (i = 0; i < ds->num_samples; i++) free(ds->features[i]); free(ds->features); }
    if (ds->labels) { for (i = 0; i < ds->num_samples; i++) free(ds->labels[i]); free(ds->labels); }
    if (ds->feat_min) free(ds->feat_min);
    if (ds->feat_max) free(ds->feat_max);
    if (ds->feat_names) { for (j = 0; j < ds->num_features; j++) free(ds->feat_names[j]); free(ds->feat_names); }
    if (ds->label_min) free(ds->label_min);
    if (ds->label_max) free(ds->label_max);
    free(ds);
}

/* ==================== Feature Normalization (common) ==================== */

void data_normalize_features(Dataset *ds) {
    int i, j, s, fi;
    double range;
    for (i = 0; i < ds->num_samples; i++) {
        for (j = 0; j < ds->num_features; j++) {
            for (s = 0; s < ds->sequence_length; s++) {
                fi = s * ds->num_features + j;
                range = ds->feat_max[j] - ds->feat_min[j];
                if (range < NORMALIZATION_EPSILON) range = 1.0;
                ds->features[i][fi] = (ds->features[i][fi] - ds->feat_min[j]) / range;
            }
        }
    }
}

/* ==================== Train/Test Split ==================== */

DatasetSplit data_split_train_test(Dataset *ds, double train_ratio) {
    DatasetSplit split;
    int train_count, test_count, i, src;
    split.train = NULL; split.test = NULL;
    train_count = (int)((double)ds->num_samples * g_train_ratio);
    if (train_count < 1) train_count = 1;
    if (train_count >= ds->num_samples) train_count = ds->num_samples - 1;
    test_count = ds->num_samples - train_count;

    split.train = (Dataset *)malloc(sizeof(Dataset));
    split.test = (Dataset *)malloc(sizeof(Dataset));
    if (!split.train || !split.test) { free(split.train); free(split.test); split.train = NULL; split.test = NULL; return split; }
    *split.train = *ds; *split.test = *ds;
    split.train->num_samples = train_count; split.test->num_samples = test_count;

    split.train->features = (double **)malloc((size_t)train_count * sizeof(double *));
    split.train->labels = (double **)malloc((size_t)train_count * sizeof(double *));
    split.test->features = (double **)malloc((size_t)test_count * sizeof(double *));
    split.test->labels = (double **)malloc((size_t)test_count * sizeof(double *));
    if (!split.train->features || !split.train->labels || !split.test->features || !split.test->labels) {
        free(split.train->features); free(split.train->labels); free(split.test->features); free(split.test->labels);
        free(split.train); free(split.test); split.train = NULL; split.test = NULL; return split;
    }
    for (i = 0; i < train_count; i++) { split.train->features[i] = ds->features[i]; split.train->labels[i] = ds->labels[i]; }
    for (i = 0; i < test_count; i++) { src = train_count + i; split.test->features[i] = ds->features[src]; split.test->labels[i] = ds->labels[src]; }

    split.train->feat_min = ds->feat_min; split.train->feat_max = ds->feat_max; split.train->feat_names = ds->feat_names;
    split.train->label_min = ds->label_min; split.train->label_max = ds->label_max;
    split.test->feat_min = ds->feat_min; split.test->feat_max = ds->feat_max; split.test->feat_names = ds->feat_names;
    split.test->label_min = ds->label_min; split.test->label_max = ds->label_max;
    split.train->raw_data = ds->raw_data; split.test->raw_data = ds->raw_data;
    split.train->num_raw = ds->num_raw; split.test->num_raw = ds->num_raw;
    return split;
}

/* ==================== Normalization I/O ==================== */

int data_save_normalization(const char *filepath, Dataset *ds) {
    FILE *f; int j;
    f = fopen(filepath, "w"); if (!f) return -1;
    fprintf(f, "# Normalization Parameters\n# num_features: %d\n# num_outputs: %d\n# sequence_length: %d\n\n", ds->num_features, ds->num_outputs, ds->sequence_length);
    for (j = 0; j < ds->num_features; j++)
        fprintf(f, "feature_name: %s | min: %.8f | max: %.8f\n", ds->feat_names[j], ds->feat_min[j], ds->feat_max[j]);
    fprintf(f, "label_min:"); for (j = 0; j < ds->num_outputs; j++) fprintf(f, " %.8f", ds->label_min[j]); fprintf(f, "\n");
    fprintf(f, "label_max:"); for (j = 0; j < ds->num_outputs; j++) fprintf(f, " %.8f", ds->label_max[j]); fprintf(f, "\n");
    fclose(f); return 0;
}

int data_load_normalization(const char *filepath, Dataset *ds) {
    FILE *f; char line[8192]; int count = 0; char name[64]; double mn, mx;
    f = fopen(filepath, "r"); if (!f) return -1;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (strncmp(line, "label_min:", 10) == 0) { char *p = line + 10; int j; for (j = 0; j < ds->num_outputs && *p; j++) ds->label_min[j] = strtod(p, &p); continue; }
        if (strncmp(line, "label_max:", 10) == 0) { char *p = line + 10; int j; for (j = 0; j < ds->num_outputs && *p; j++) ds->label_max[j] = strtod(p, &p); continue; }
        if (sscanf(line, "feature_name: %63s | min: %lf | max: %lf", name, &mn, &mx) == 3) {
            if (count < ds->num_features) { free(ds->feat_names[count]); ds->feat_names[count] = utils_strdup(name); ds->feat_min[count] = mn; ds->feat_max[count] = mx; count++; }
        }
    }
    fclose(f); return (count == ds->num_features) ? 0 : -1;
}

/* ==================== Free Split ==================== */

void data_free_split(DatasetSplit *split) {
    if (!split) return;
    if (split->train) free(split->train);
    if (split->test) free(split->test);
}
