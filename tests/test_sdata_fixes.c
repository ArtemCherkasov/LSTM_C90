#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TEST_CSV_EUR "../resources/EURUSD_H1_201001040000_202611022100.csv"
#define TEST_CSV_GBP "../resources/GBPUSD_H1_201001040000_202611022100.csv"
#define DBL_TOL 1e-6

/* ==================== Test: MTF features are finite ==================== */

int test_mtf_features_finite(void) {
    DataRow *raw;
    int n = 100;
    double **feat;
    double *fmin, *fmax;
    char **fnames;
    int i, j;
    int bad = 0;

    printf("[TEST] MTF features finite (no garbage from uninitialized memory)...");
    fflush(stdout);

    raw = (DataRow *)utils_safe_malloc((size_t)n * sizeof(DataRow));
    for (i = 0; i < n; i++) {
        raw[i].open = 1.1000 + i * 0.001;
        raw[i].high = raw[i].open + 0.0005;
        raw[i].low = raw[i].open - 0.0005;
        raw[i].close = raw[i].open + 0.0002;
        raw[i].tickvol = 1000 + i;
        raw[i].vol = 0;
        raw[i].spread = 1;
        snprintf(raw[i].date, 16, "2024.01.%02d", (i % 28) + 1);
        snprintf(raw[i].time_, 16, "%02d:00", i % 24);
    }

    fmin = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fmax = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fnames = (char **)calloc(NUM_RAW_FEATURES, sizeof(char *));
    feat = data_compute_raw_features(raw, n, NUM_RAW_FEATURES, fmin, fmax, fnames);

    /* Check all MTF min/max (indices 13-25) are finite and reasonable */
    for (j = NUM_BASE_FEATURES; j < NUM_RAW_FEATURES; j++) {
        if (fmin[j] != fmin[j] || fabs(fmin[j]) > 1e10) {
            printf("\n  FAIL: feat_min[%d]=%.6g is NaN or extreme\n", j, fmin[j]);
            bad = 1; break;
        }
        if (fmax[j] != fmax[j] || fabs(fmax[j]) > 1e10) {
            printf("\n  FAIL: feat_max[%d]=%.6g is NaN or extreme\n", j, fmax[j]);
            bad = 1; break;
        }
    }

    /* Also check feature values themselves */
    if (!bad) {
        for (i = 0; i < n && !bad; i++) {
            for (j = NUM_BASE_FEATURES; j < NUM_RAW_FEATURES; j++) {
                if (feat[i][j] != feat[i][j] || fabs(feat[i][j]) > 1e10) {
                    printf("\n  FAIL: feat[%d][%d]=%.6g is NaN or extreme\n", i, j, feat[i][j]);
                    bad = 1; break;
                }
            }
        }
    }

    data_free_raw_features(feat, n);
    for (i = 0; i < NUM_RAW_FEATURES; i++) free(fnames[i]);
    free(fnames); free(fmin); free(fmax); utils_safe_free(raw);

    if (!bad) {
        printf("\n  PASS: all MTF features and min/max are finite\n");
        return 1;
    }
    return 0;
}

/* ==================== Test: MTF bounded features in [0,1] ==================== */

int test_mtf_features_bounded(void) {
    DataRow *raw;
    int n = 200;
    double **feat;
    double *fmin, *fmax;
    char **fnames;
    int i;
    int bad = 0;

    printf("[TEST] MTF bounded features (rsi, stoch_k, hour, dow) in [0,1]...");
    fflush(stdout);

    raw = (DataRow *)utils_safe_malloc((size_t)n * sizeof(DataRow));
    for (i = 0; i < n; i++) {
        raw[i].open = 1.1000 + (i % 50) * 0.001;
        raw[i].high = raw[i].open + 0.0005;
        raw[i].low = raw[i].open - 0.0005;
        raw[i].close = raw[i].open + 0.0002;
        raw[i].tickvol = 1000 + i;
        raw[i].vol = 0;
        raw[i].spread = 1;
        snprintf(raw[i].date, 16, "2024.%02d.%02d", (i / 28) % 12 + 1, (i % 28) + 1);
        snprintf(raw[i].time_, 16, "%02d:00", i % 24);
    }

    fmin = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fmax = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fnames = (char **)calloc(NUM_RAW_FEATURES, sizeof(char *));
    feat = data_compute_raw_features(raw, n, NUM_RAW_FEATURES, fmin, fmax, fnames);

    /* MTF RSI is index 19, stoch_k is 23, hour is 24, dow is 25 */
    {
        int bounded_idx[] = {19, 23, 24, 25};
        int bi;
        for (bi = 0; bi < 4 && !bad; bi++) {
            int idx = bounded_idx[bi];
            for (i = 20; i < n; i++) { /* skip warmup */
                if (feat[i][idx] < -0.01 || feat[i][idx] > 1.01) {
                    printf("\n  FAIL: feat[%d][%d]=%.6f outside [0,1]\n", i, idx, feat[i][idx]);
                    bad = 1; break;
                }
            }
        }
    }

    data_free_raw_features(feat, n);
    for (i = 0; i < NUM_RAW_FEATURES; i++) free(fnames[i]);
    free(fnames); free(fmin); free(fmax); utils_safe_free(raw);

    if (!bad) {
        printf("\n  PASS: MTF rsi/stoch_k/hour/dow all in [0,1]\n");
        return 1;
    }
    return 0;
}

/* ==================== Test: MTF min/max not garbage ==================== */

int test_mtf_minmax_no_garbage(void) {
    DataRow *raw;
    int n = 50;
    double **feat;
    double *fmin, *fmax;
    char **fnames;
    int j;
    int bad = 0;

    printf("[TEST] MTF min/max values reasonable (no astronomical garbage)...");
    fflush(stdout);

    raw = (DataRow *)utils_safe_malloc((size_t)n * sizeof(DataRow));
    for (j = 0; j < n; j++) {
        raw[j].open = 1.1000 + j * 0.001;
        raw[j].high = raw[j].open + 0.0005;
        raw[j].low = raw[j].open - 0.0005;
        raw[j].close = raw[j].open + 0.0002;
        raw[j].tickvol = 1000;
        raw[j].vol = 0;
        raw[j].spread = 1;
        snprintf(raw[j].date, 16, "2024.01.%02d", (j % 28) + 1);
        snprintf(raw[j].time_, 16, "%02d:00", j % 24);
    }

    fmin = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fmax = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fnames = (char **)calloc(NUM_RAW_FEATURES, sizeof(char *));
    feat = data_compute_raw_features(raw, n, NUM_RAW_FEATURES, fmin, fmax, fnames);

    for (j = NUM_BASE_FEATURES; j < NUM_RAW_FEATURES; j++) {
        if (fmin[j] > fmax[j]) {
            printf("\n  FAIL: feat_min[%d]=%.6g > feat_max[%d]=%.6g\n", j, fmin[j], j, fmax[j]);
            bad = 1; break;
        }
        if (fmax[j] - fmin[j] > 100.0) {
            printf("\n  FAIL: feat range[%d] = %.6g too wide\n", j, fmax[j] - fmin[j]);
            bad = 1; break;
        }
    }

    data_free_raw_features(feat, n);
    {
        int i;
        for (i = 0; i < NUM_RAW_FEATURES; i++) free(fnames[i]);
    }
    free(fnames); free(fmin); free(fmax); utils_safe_free(raw);

    if (!bad) {
        printf("\n  PASS: all MTF min/max reasonable\n");
        return 1;
    }
    return 0;
}

/* ==================== Test: sdata intersection chronological and matching ==================== */

int test_sdata_intersection_chronological(void) {
    DataRow *eur = NULL, *gbp = NULL;
    DataRow *out_pri, *out_sec;
    int n_eur = 0, n_gbp = 0, out_n;
    int i;
    int bad = 0;

    printf("[TEST] sdata intersection: chronological + date matching...");
    fflush(stdout);

    if (data_load_raw(TEST_CSV_EUR, &eur, &n_eur) != 0 ||
        data_load_raw(TEST_CSV_GBP, &gbp, &n_gbp) != 0) {
        printf("\n  FAIL: could not load CSVs\n");
        utils_safe_free(eur); utils_safe_free(gbp);
        return 0;
    }

    data_intersect_raw(eur, n_eur, gbp, n_gbp, &out_pri, &out_sec, &out_n);

    if (out_n == 0) {
        printf("\n  FAIL: intersection is empty\n");
        utils_safe_free(eur); utils_safe_free(gbp);
        utils_safe_free(out_pri); utils_safe_free(out_sec);
        return 0;
    }

    for (i = 0; i < out_n; i++) {
        char pri_key[32], sec_key[32];
        snprintf(pri_key, sizeof(pri_key), "%s %s", out_pri[i].date, out_pri[i].time_);
        snprintf(sec_key, sizeof(sec_key), "%s %s", out_sec[i].date, out_sec[i].time_);

        /* Every pri date must match sec date */
        if (strcmp(pri_key, sec_key) != 0) {
            printf("\n  FAIL: row %d pri='%s' != sec='%s'\n", i, pri_key, sec_key);
            bad = 1; break;
        }

        /* Chronological order */
        if (i > 0) {
            char prev_key[32];
            snprintf(prev_key, sizeof(prev_key), "%s %s", out_pri[i-1].date, out_pri[i-1].time_);
            if (strcmp(prev_key, pri_key) >= 0) {
                printf("\n  FAIL: row %d not chronological\n", i);
                bad = 1; break;
            }
        }
    }

    utils_safe_free(eur); utils_safe_free(gbp);
    utils_safe_free(out_pri); utils_safe_free(out_sec);

    if (!bad) {
        printf("\n  PASS: %d rows, all dates match and chronological\n", out_n);
        return 1;
    }
    return 0;
}

/* ==================== Test: sdata intersection size close to primary ==================== */

int test_sdata_intersection_size(void) {
    DataRow *eur = NULL, *gbp = NULL;
    DataRow *out_pri, *out_sec;
    int n_eur = 0, n_gbp = 0, out_n;
    int min_n;
    double ratio;

    printf("[TEST] sdata intersection: size >= 95%% of min(EURUSD, GBPUSD)...");
    fflush(stdout);

    if (data_load_raw(TEST_CSV_EUR, &eur, &n_eur) != 0 ||
        data_load_raw(TEST_CSV_GBP, &gbp, &n_gbp) != 0) {
        printf("\n  FAIL: could not load CSVs\n");
        utils_safe_free(eur); utils_safe_free(gbp);
        return 0;
    }

    data_intersect_raw(eur, n_eur, gbp, n_gbp, &out_pri, &out_sec, &out_n);

    min_n = (n_eur < n_gbp) ? n_eur : n_gbp;
    ratio = (min_n > 0) ? (double)out_n / (double)min_n : 0.0;

    utils_safe_free(eur); utils_safe_free(gbp);
    utils_safe_free(out_pri); utils_safe_free(out_sec);

    if (ratio < 0.95) {
        printf("\n  FAIL: intersection %d / min %d = %.1f%% < 95%%\n", out_n, min_n, ratio * 100.0);
        return 0;
    }

    printf("\n  PASS: intersection %d / min %d = %.1f%%\n", out_n, min_n, ratio * 100.0);
    return 1;
}

/* ==================== Test: 39-feature dataset with finite MTF ==================== */

int test_sdata_39_features_finite(void) {
    DataRow *eur = NULL, *gbp = NULL;
    DataRow *out_pri, *out_sec;
    int n_eur = 0, n_gbp = 0, out_n;
    double **sec_feat;
    double *sec_min, *sec_max;
    char **sec_names;
    Dataset *ds;
    int j;
    int bad = 0;

    printf("[TEST] 39-feature dataset: MTF min/max all finite...");
    fflush(stdout);

    if (data_load_raw(TEST_CSV_EUR, &eur, &n_eur) != 0 ||
        data_load_raw(TEST_CSV_GBP, &gbp, &n_gbp) != 0) {
        printf("\n  FAIL: could not load CSVs\n");
        utils_safe_free(eur); utils_safe_free(gbp);
        return 0;
    }

    data_intersect_raw(eur, n_eur, gbp, n_gbp, &out_pri, &out_sec, &out_n);
    utils_safe_free(eur); utils_safe_free(gbp);

    sec_min = (double *)calloc(NUM_BASE_FEATURES, sizeof(double));
    sec_max = (double *)calloc(NUM_BASE_FEATURES, sizeof(double));
    sec_names = (char **)calloc(NUM_BASE_FEATURES, sizeof(char *));
    sec_feat = data_compute_raw_features(out_sec, out_n, NUM_BASE_FEATURES,
                                          sec_min, sec_max, sec_names);

    ds = data_alloc_dataset_ext(out_pri, out_n, 32, 1, 1, MODE_DIRECTION,
                                sec_feat, NUM_BASE_FEATURES, sec_min, sec_max, sec_names);

    if (!ds) {
        printf("\n  FAIL: could not build dataset\n");
        goto cleanup;
    }

    if (ds->num_features != 39) {
        printf("\n  FAIL: num_features=%d, expected 39\n", ds->num_features);
        bad = 1;
    }

    /* Check all 39 min/max are finite */
    if (!bad) {
        for (j = 0; j < ds->num_features; j++) {
            if (ds->feat_min[j] != ds->feat_min[j] || fabs(ds->feat_min[j]) > 1e10) {
                printf("\n  FAIL: feat_min[%d]=%.6g NaN or extreme\n", j, ds->feat_min[j]);
                bad = 1; break;
            }
            if (ds->feat_max[j] != ds->feat_max[j] || fabs(ds->feat_max[j]) > 1e10) {
                printf("\n  FAIL: feat_max[%d]=%.6g NaN or extreme\n", j, ds->feat_max[j]);
                bad = 1; break;
            }
        }
    }

    data_free_dataset(ds);

cleanup:
    data_free_raw_features(sec_feat, out_n);
    for (j = 0; j < NUM_BASE_FEATURES; j++) free(sec_names[j]);
    free(sec_names); free(sec_min); free(sec_max);
    utils_safe_free(out_pri); utils_safe_free(out_sec);

    if (!bad) {
        printf("\n  PASS: 39 features, all min/max finite\n");
        return 1;
    }
    return 0;
}

/* ==================== Test: iteration count with sdata similar ==================== */

int test_sdata_iteration_count(void) {
    DataRow *eur = NULL, *gbp = NULL;
    DataRow *out_pri, *out_sec;
    int n_eur = 0, n_gbp = 0, out_n;
    Dataset *ds_no_sdata, *ds_with_sdata;
    double **sec_feat;
    double *sec_min, *sec_max;
    char **sec_names;
    int seq_len = 32, pred_horizon = 1;
    int samples_no, samples_with;
    double ratio;

    printf("[TEST] sdata iteration count: similar to 26-feature baseline...");
    fflush(stdout);

    if (data_load_raw(TEST_CSV_EUR, &eur, &n_eur) != 0 ||
        data_load_raw(TEST_CSV_GBP, &gbp, &n_gbp) != 0) {
        printf("\n  FAIL: could not load CSVs\n");
        utils_safe_free(eur); utils_safe_free(gbp);
        return 0;
    }

    /* Dataset without sdata */
    ds_no_sdata = data_alloc_dataset(eur, n_eur, seq_len, pred_horizon, 1, MODE_DIRECTION);
    if (!ds_no_sdata) {
        printf("\n  FAIL: could not build baseline dataset\n");
        utils_safe_free(eur); utils_safe_free(gbp);
        return 0;
    }
    samples_no = ds_no_sdata->num_samples;
    data_free_dataset(ds_no_sdata);

    /* Dataset with sdata (intersection) */
    data_intersect_raw(eur, n_eur, gbp, n_gbp, &out_pri, &out_sec, &out_n);

    sec_min = (double *)calloc(NUM_BASE_FEATURES, sizeof(double));
    sec_max = (double *)calloc(NUM_BASE_FEATURES, sizeof(double));
    sec_names = (char **)calloc(NUM_BASE_FEATURES, sizeof(char *));
    sec_feat = data_compute_raw_features(out_sec, out_n, NUM_BASE_FEATURES,
                                          sec_min, sec_max, sec_names);

    ds_with_sdata = data_alloc_dataset_ext(out_pri, out_n, seq_len, pred_horizon, 1,
                                            MODE_DIRECTION, sec_feat, NUM_BASE_FEATURES,
                                            sec_min, sec_max, sec_names);
    samples_with = ds_with_sdata ? ds_with_sdata->num_samples : 0;

    if (ds_with_sdata) data_free_dataset(ds_with_sdata);
    data_free_raw_features(sec_feat, out_n);
    {
        int j;
        for (j = 0; j < NUM_BASE_FEATURES; j++) free(sec_names[j]);
    }
    free(sec_names); free(sec_min); free(sec_max);
    utils_safe_free(out_pri); utils_safe_free(out_sec);
    utils_safe_free(eur); utils_safe_free(gbp);

    ratio = (samples_no > 0) ? (double)samples_with / (double)samples_no : 0.0;

    if (ratio < 0.95) {
        printf("\n  FAIL: with_sdata=%d / no_sdata=%d = %.1f%% < 95%%\n",
               samples_with, samples_no, ratio * 100.0);
        return 0;
    }

    printf("\n  PASS: with_sdata=%d / no_sdata=%d = %.1f%%\n",
           samples_with, samples_no, ratio * 100.0);
    return 1;
}
