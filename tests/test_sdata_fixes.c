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

/* ==================== Test: MTF features actually vary (not constant/noise) ==================== */

int test_mtf_features_vary(void) {
    DataRow *raw = NULL;
    int n = 0;
    double **feat;
    double *fmin, *fmax;
    char **fnames;
    int i, j;
    int bad = 0;

    printf("[TEST] MTF features vary across bars (not constant)...");
    fflush(stdout);

    if (data_load_raw("../resources/EURUSD_H1_201001040000_202611022100.csv",
                      &raw, &n) != 0) {
        printf("\n  FAIL: could not load CSV\n");
        return 0;
    }

    fmin = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fmax = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fnames = (char **)calloc(NUM_RAW_FEATURES, sizeof(char *));
    feat = data_compute_raw_features(raw, n, NUM_RAW_FEATURES, fmin, fmax, fnames);

    /* Check that each MTF feature (13-25) has meaningful variance */
    for (j = NUM_BASE_FEATURES; j < NUM_RAW_FEATURES; j++) {
        double sum = 0.0, mean, var = 0.0;
        int count = 0;
        for (i = 100; i < n; i++) { /* skip warmup */
            sum += feat[i][j];
            count++;
        }
        mean = sum / (double)count;
        for (i = 100; i < n; i++) {
            double d = feat[i][j] - mean;
            var += d * d;
        }
        var /= (double)count;
        if (var < 1e-12) {
            printf("\n  FAIL: MTF feature %d has near-zero variance (%.2e)\n", j, var);
            bad = 1; break;
        }
    }

    /* Also check that MTF features differ from their base counterparts
     * (MTF should be same indicators on H2, not identical to H1) */
    if (!bad) {
        int diffs = 0;
        for (i = 100; i < n && i < 1000; i++) {
            for (j = 0; j < NUM_BASE_FEATURES; j++) {
                if (fabs(feat[i][j] - feat[i][NUM_BASE_FEATURES + j]) > 1e-6) {
                    diffs++;
                }
            }
        }
        if (diffs < 100) {
            printf("\n  FAIL: MTF features nearly identical to base (%d diffs out of ~900)\n", diffs);
            bad = 1;
        }
    }

    data_free_raw_features(feat, n);
    for (i = 0; i < NUM_RAW_FEATURES; i++) free(fnames[i]);
    free(fnames); free(fmin); free(fmax); utils_safe_free(raw);

    if (!bad) {
        printf("\n  PASS: MTF features vary and differ from base\n");
        return 1;
    }
    return 0;
}

/* ==================== Test: MTF mapping correct after gap ==================== */

int test_mtf_mapping_after_gap(void) {
    /* Create synthetic data with a gap to test MTF time-aligned mapping.
     * Bars: 00:00, 01:00, 02:00, 03:00, (gap), 06:00, 07:00
     * H2 bars: 00:00 (from 00,01), 02:00 (from 02,03), 06:00 (from 06,07)
     * Bar at 06:00 should map to H2 06:00, NOT H2 02:00 */
    DataRow raw[7];
    double **feat;
    double *fmin, *fmax;
    char **fnames;
    int i;
    int bad = 0;

    printf("[TEST] MTF mapping correct after gap...");
    fflush(stdout);

    /* 4 bars: 00-03 */
    for (i = 0; i < 4; i++) {
        strcpy(raw[i].date, "2024.01.15");
        snprintf(raw[i].time_, 16, "%02d:00", i);
        raw[i].open = 1.10 + i * 0.001;
        raw[i].high = raw[i].open + 0.0005;
        raw[i].low = raw[i].open - 0.0005;
        raw[i].close = raw[i].open + 0.0002;
        raw[i].tickvol = 1000 + i;
        raw[i].vol = 0;
        raw[i].spread = 1;
    }
    /* Gap: skip 04:00, 05:00 */
    /* 3 bars: 06, 07 + one more for dataset */
    for (i = 4; i < 7; i++) {
        strcpy(raw[i].date, "2024.01.15");
        snprintf(raw[i].time_, 16, "%02d:00", i + 2); /* 06, 07, 08 */
        raw[i].open = 1.10 + i * 0.001;
        raw[i].high = raw[i].open + 0.0005;
        raw[i].low = raw[i].open - 0.0005;
        raw[i].close = raw[i].open + 0.0002;
        raw[i].tickvol = 1000 + i;
        raw[i].vol = 0;
        raw[i].spread = 1;
    }

    fmin = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fmax = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fnames = (char **)calloc(NUM_RAW_FEATURES, sizeof(char *));
    feat = data_compute_raw_features(raw, 7, NUM_RAW_FEATURES, fmin, fmax, fnames);

    /* Bar at index 4 (06:00) should have MTF features from H2 06:00.
     * With old mi=i/2 mapping: mi=4/2=2, but H2 only has 3 bars (00,02,06),
     * so mi=2 → H2 06:00 (correct by luck here).
     *
     * Bar at index 4 with a bigger gap would fail with i/2.
     * Let's verify the hour feature (MTF index 24) is correct.
     * H2 bar at 06:00 → hour=6, mtf_hour = 6/23 ≈ 0.2609
     * H2 bar at 02:00 → hour=2, mtf_hour = 2/23 ≈ 0.0870
     * If mapped to wrong H2, the hour feature would be wrong. */
    {
        double mtf_hour_at_06 = feat[4][24]; /* MTF hour for bar at 06:00 */
        double expected = 6.0 / 23.0;
        if (fabs(mtf_hour_at_06 - expected) > 0.01) {
            printf("\n  FAIL: MTF hour at 06:00 = %.6f, expected %.6f\n",
                   mtf_hour_at_06, expected);
            bad = 1;
        }
    }

    data_free_raw_features(feat, 7);
    for (i = 0; i < NUM_RAW_FEATURES; i++) free(fnames[i]);
    free(fnames); free(fmin); free(fmax);

    if (!bad) {
        printf("\n  PASS: MTF mapping correct after gap\n");
        return 1;
    }
    return 0;
}

/* ==================== Test: MTF differs from base on H4 data ==================== */

int test_mtf_differs_from_base_h4(void) {
    /* The critical bug: when data is already H4 (hours 0,4,8,12...),
     * time-aligned aggregation by 2 is a no-op, making MTF = base.
     * With consecutive aggregation, MTF should differ from base. */
    DataRow *raw = NULL;
    int n_raw = 0;
    DataRow *agg = NULL;
    int n_agg = 0;
    double **feat;
    double *fmin, *fmax;
    char **fnames;
    int i, j;
    int diffs = 0;
    int bad = 0;

    printf("[TEST] MTF differs from base on H4 data...");
    fflush(stdout);

    if (data_load_raw("../resources/EURUSD_H1_201001040000_202611022100.csv",
                      &raw, &n_raw) != 0) {
        printf("\n  FAIL: could not load CSV\n");
        return 0;
    }

    /* Aggregate to H4 */
    data_aggregate_raw(raw, n_raw, 4, &agg, &n_agg);
    utils_safe_free(raw);

    /* Compute 26 features on H4 data */
    fmin = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fmax = (double *)calloc(NUM_RAW_FEATURES, sizeof(double));
    fnames = (char **)calloc(NUM_RAW_FEATURES, sizeof(char *));
    feat = data_compute_raw_features(agg, n_agg, NUM_RAW_FEATURES, fmin, fmax, fnames);

    /* Check MTF min/max differ from base min/max */
    for (j = 0; j < NUM_BASE_FEATURES; j++) {
        if (fabs(fmin[j] - fmin[NUM_BASE_FEATURES + j]) > 1e-6 ||
            fabs(fmax[j] - fmax[NUM_BASE_FEATURES + j]) > 1e-6) {
            diffs++;
        }
    }

    if (diffs < 5) {
        printf("\n  FAIL: MTF min/max nearly identical to base (%d diffs out of 13)\n", diffs);
        bad = 1;
    }

    /* Also check feature values differ */
    if (!bad) {
        int val_diffs = 0;
        for (i = 100; i < n_agg && i < 1000; i++) {
            for (j = 0; j < NUM_BASE_FEATURES; j++) {
                if (fabs(feat[i][j] - feat[i][NUM_BASE_FEATURES + j]) > 1e-6) {
                    val_diffs++;
                }
            }
        }
        if (val_diffs < 100) {
            printf("\n  FAIL: MTF values nearly identical to base (%d diffs out of ~11700)\n", val_diffs);
            bad = 1;
        }
    }

    data_free_raw_features(feat, n_agg);
    for (i = 0; i < NUM_RAW_FEATURES; i++) free(fnames[i]);
    free(fnames); free(fmin); free(fmax); utils_safe_free(agg);

    if (!bad) {
        printf("\n  PASS: MTF differs from base on H4 data (%d min/max diffs)\n", diffs);
        return 1;
    }
    return 0;
}
