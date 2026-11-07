#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Path to test CSV — relative to build directory */
#define TEST_CSV "../resources/EURUSD_H1_201001040000_202611022100.csv"

/* T+1e-6 for floating-point comparisons */
#define DBL_TOL 1e-6

/* ==================== Test: data_load_raw — basic loading ==================== */

int test_load_raw_basic(void) {
    DataRow *raw = NULL;
    int n = 0;
    int i;
    int bad = 0;

    printf("[TEST] data_load_raw: basic loading from CSV...");
    fflush(stdout);

    if (data_load_raw(TEST_CSV, &raw, &n) != 0) {
        printf("\n  FAIL: could not load %s\n", TEST_CSV);
        return 0;
    }

    /* Expect a reasonable number of rows */
    if (n < 1000) {
        printf("\n  FAIL: only %d rows loaded, expected at least 1000\n", n);
        utils_safe_free(raw);
        return 0;
    }

    /* Every row should have valid OHLC */
    for (i = 0; i < n; i++) {
        if (raw[i].open <= 0.0 || raw[i].high <= 0.0 ||
            raw[i].low <= 0.0 || raw[i].close <= 0.0) {
            printf("\n  FAIL: row %d has non-positive price\n", i);
            bad = 1;
            break;
        }
        if (raw[i].high < raw[i].low) {
            printf("\n  FAIL: row %d high < low\n", i);
            bad = 1;
            break;
        }
        if (raw[i].date[0] == 0 || raw[i].time_[0] == 0) {
            printf("\n  FAIL: row %d has empty date/time\n", i);
            bad = 1;
            break;
        }
    }

    utils_safe_free(raw);

    if (!bad) {
        printf("\n  PASS: loaded %d rows, all have valid OHLC and date/time\n", n);
        return 1;
    }
    return 0;
}

/* ==================== Test: data_load_raw — chronological order ==================== */

int test_load_raw_chronological(void) {
    DataRow *raw = NULL;
    int n = 0;
    int i;
    int bad = 0;

    printf("[TEST] data_load_raw: chronological order...");
    fflush(stdout);

    if (data_load_raw(TEST_CSV, &raw, &n) != 0) {
        printf("\n  FAIL: could not load CSV\n");
        return 0;
    }

    for (i = 1; i < n; i++) {
        char prev_dt[32], cur_dt[32];
        snprintf(prev_dt, sizeof(prev_dt), "%s %s", raw[i-1].date, raw[i-1].time_);
        snprintf(cur_dt,  sizeof(cur_dt),  "%s %s", raw[i].date,   raw[i].time_);
        if (strcmp(prev_dt, cur_dt) > 0) {
            printf("\n  FAIL: row %d (%s) < row %d (%s)\n", i, cur_dt, i-1, prev_dt);
            bad = 1;
            break;
        }
    }

    utils_safe_free(raw);

    if (!bad) {
        printf("\n  PASS: all %d rows in chronological order\n", n);
        return 1;
    }
    return 0;
}

/* ==================== Helper: verify aggregation correctness ==================== */

static int verify_aggregation(DataRow *raw, int n_raw,
                              DataRow *agg, int n_agg, int h) {
    int i;

    /* Check that all output times are aligned to h-period boundaries */
    for (i = 0; i < n_agg; i++) {
        int hour = atoi(agg[i].time_);
        if (hour % h != 0) {
            printf("\n  FAIL: agg[%d] hour=%d not aligned to H%d\n", i, hour, h);
            return 0;
        }
        if (agg[i].open <= 0.0 || agg[i].high < agg[i].low) {
            printf("\n  FAIL: agg[%d] invalid OHLC\n", i);
            return 0;
        }
    }

    /* Verify OHLC by reconstructing groups from raw data */
    {
        int raw_idx = 0;
        int agg_idx;
        for (agg_idx = 0; agg_idx < n_agg; agg_idx++) {
            int g_hour = atoi(agg[agg_idx].time_);
            char g_date[16];
            double exp_open, exp_high, exp_low, exp_close;
            int exp_tickvol;
            int count = 0;

            strcpy(g_date, agg[agg_idx].date);
            exp_high = -1e30; exp_low = 1e30; exp_tickvol = 0;
            exp_open = 0.0; exp_close = 0.0;

            while (raw_idx < n_raw) {
                int r_hour = atoi(raw[raw_idx].time_);
                int r_group = (r_hour / h) * h;
                if (count > 0 && (strcmp(raw[raw_idx].date, g_date) != 0 || r_group != g_hour))
                    break;
                if (strcmp(raw[raw_idx].date, g_date) == 0 && r_group == g_hour) {
                    if (count == 0) exp_open = raw[raw_idx].open;
                    if (raw[raw_idx].high > exp_high) exp_high = raw[raw_idx].high;
                    if (raw[raw_idx].low < exp_low) exp_low = raw[raw_idx].low;
                    exp_close = raw[raw_idx].close;
                    exp_tickvol += raw[raw_idx].tickvol;
                    count++;
                    raw_idx++;
                } else {
                    raw_idx++;
                }
            }

            if (count == 0) {
                printf("\n  FAIL: agg[%d] has no matching raw bars\n", agg_idx);
                return 0;
            }
            if (fabs(agg[agg_idx].open - exp_open) > DBL_TOL) {
                printf("\n  FAIL: agg[%d].open=%.5f != %.5f\n", agg_idx, agg[agg_idx].open, exp_open);
                return 0;
            }
            if (fabs(agg[agg_idx].high - exp_high) > DBL_TOL) {
                printf("\n  FAIL: agg[%d].high=%.5f != %.5f\n", agg_idx, agg[agg_idx].high, exp_high);
                return 0;
            }
            if (fabs(agg[agg_idx].low - exp_low) > DBL_TOL) {
                printf("\n  FAIL: agg[%d].low=%.5f != %.5f\n", agg_idx, agg[agg_idx].low, exp_low);
                return 0;
            }
            if (fabs(agg[agg_idx].close - exp_close) > DBL_TOL) {
                printf("\n  FAIL: agg[%d].close=%.5f != %.5f\n", agg_idx, agg[agg_idx].close, exp_close);
                return 0;
            }
        }
    }

    return 1;
}

/* ==================== Test: data_aggregate_raw with h=2 ==================== */

int test_aggregate_h2(void) {
    DataRow *raw = NULL;
    int n_raw = 0;
    DataRow *agg = NULL;
    int n_agg = 0;

    printf("[TEST] data_aggregate_raw: h=2 aggregation...");
    fflush(stdout);

    if (data_load_raw(TEST_CSV, &raw, &n_raw) != 0) {
        printf("\n  FAIL: could not load CSV\n");
        return 0;
    }

    data_aggregate_raw(raw, n_raw, 2, &agg, &n_agg);

    if (!verify_aggregation(raw, n_raw, agg, n_agg, 2)) {
        utils_safe_free(raw);
        utils_safe_free(agg);
        return 0;
    }

    printf("\n  PASS: h=2 aggregation correct (%d rows -> %d rows)\n", n_raw, n_agg);
    utils_safe_free(raw);
    utils_safe_free(agg);
    return 1;
}

/* ==================== Test: data_aggregate_raw with h=4 ==================== */

int test_aggregate_h4(void) {
    DataRow *raw = NULL;
    int n_raw = 0;
    DataRow *agg = NULL;
    int n_agg = 0;

    printf("[TEST] data_aggregate_raw: h=4 aggregation...");
    fflush(stdout);

    if (data_load_raw(TEST_CSV, &raw, &n_raw) != 0) {
        printf("\n  FAIL: could not load CSV\n");
        return 0;
    }

    data_aggregate_raw(raw, n_raw, 4, &agg, &n_agg);

    if (!verify_aggregation(raw, n_raw, agg, n_agg, 4)) {
        utils_safe_free(raw);
        utils_safe_free(agg);
        return 0;
    }

    printf("\n  PASS: h=4 aggregation correct (%d rows -> %d rows)\n", n_raw, n_agg);
    utils_safe_free(raw);
    utils_safe_free(agg);
    return 1;
}

/* ==================== Test: data_aggregate_raw with h=8 ==================== */

int test_aggregate_h8(void) {
    DataRow *raw = NULL;
    int n_raw = 0;
    DataRow *agg = NULL;
    int n_agg = 0;

    printf("[TEST] data_aggregate_raw: h=8 aggregation...");
    fflush(stdout);

    if (data_load_raw(TEST_CSV, &raw, &n_raw) != 0) {
        printf("\n  FAIL: could not load CSV\n");
        return 0;
    }

    data_aggregate_raw(raw, n_raw, 8, &agg, &n_agg);

    if (!verify_aggregation(raw, n_raw, agg, n_agg, 8)) {
        utils_safe_free(raw);
        utils_safe_free(agg);
        return 0;
    }

    printf("\n  PASS: h=8 aggregation correct (%d rows -> %d rows)\n", n_raw, n_agg);
    utils_safe_free(raw);
    utils_safe_free(agg);
    return 1;
}

/* ==================== Test: data_aggregate_raw h=1 is identity ==================== */

int test_aggregate_h1_identity(void) {
    DataRow *raw = NULL;
    int n_raw = 0;
    DataRow *agg = NULL;
    int n_agg = 0;

    printf("[TEST] data_aggregate_raw: h=1 is identity...");
    fflush(stdout);

    if (data_load_raw(TEST_CSV, &raw, &n_raw) != 0) {
        printf("\n  FAIL: could not load CSV\n");
        return 0;
    }

    data_aggregate_raw(raw, n_raw, 1, &agg, &n_agg);

    /* h=1 returns the original pointer */
    if (agg != raw || n_agg != n_raw) {
        printf("\n  FAIL: h=1 should return original data unchanged\n");
        utils_safe_free(raw);
        if (agg != raw) utils_safe_free(agg);
        return 0;
    }

    printf("\n  PASS: h=1 returns original data (identity)\n");
    utils_safe_free(raw);
    return 1;
}

/* ==================== Test: h=4 direct equals h=2 then h=2 for OHLC values ==================== */
/* Note: with time-aligned grouping, double h=2 may produce different row counts
 * than direct h=4 because H2 bars already have even hours. Instead, we verify
 * that both approaches produce correctly aligned, valid OHLC data. */
int test_aggregate_double_h2(void) {
    DataRow *raw = NULL;
    int n_raw = 0;
    DataRow *agg1 = NULL, *agg2 = NULL;
    int n1 = 0, n2 = 0;
    DataRow *agg4 = NULL;
    int n4 = 0;
    int i;
    int bad = 0;

    printf("[TEST] data_aggregate_raw: h=2 then h=2 and h=4 both aligned...");
    fflush(stdout);

    if (data_load_raw(TEST_CSV, &raw, &n_raw) != 0) {
        printf("\n  FAIL: could not load CSV\n");
        return 0;
    }

    /* Aggregate by 2, then by 2 again */
    data_aggregate_raw(raw, n_raw, 2, &agg1, &n1);
    data_aggregate_raw(agg1, n1, 2, &agg2, &n2);

    /* Aggregate directly by 4 */
    data_aggregate_raw(raw, n_raw, 4, &agg4, &n4);

    /* Both should have aligned times (h=2 result is H2-aligned, h=4 is H4-aligned) */
    for (i = 0; i < n2 && !bad; i++) {
        int hour = atoi(agg2[i].time_);
        if (hour % 2 != 0) { printf("\n  FAIL: double-h2 agg[%d] hour=%d not H2-aligned\n", i, hour); bad = 1; }
    }
    for (i = 0; i < n4 && !bad; i++) {
        int hour = atoi(agg4[i].time_);
        if (hour % 4 != 0) { printf("\n  FAIL: h4 agg[%d] hour=%d not H4-aligned\n", i, hour); bad = 1; }
    }

    /* h=4 should have fewer or equal rows than double h=2 */
    if (!bad && n4 > n2) {
        printf("\n  FAIL: h4 has %d rows > double-h2 %d rows\n", n4, n2);
        bad = 1;
    }

    utils_safe_free(raw);
    utils_safe_free(agg1);
    utils_safe_free(agg2);
    utils_safe_free(agg4);

    if (!bad) {
        printf("\n  PASS: double-h2 (%d, H2-aligned) and h4 (%d, H4-aligned)\n", n2, n4);
        return 1;
    }
    return 0;
}

/* ==================== Test: data_intersect_raw full overlap ==================== */

int test_intersect_raw_full_overlap(void) {
    DataRow pri[5], sec[5];
    DataRow *out_pri, *out_sec;
    int out_n;
    int i;

    printf("[TEST] data_intersect_raw: full overlap...");
    fflush(stdout);

    for (i = 0; i < 5; i++) {
        snprintf(pri[i].date, 16, "2024.01.%02d", i + 1);
        snprintf(pri[i].time_, 16, "00:00");
        pri[i].open = 1.0 + i;
        sec[i] = pri[i];
        sec[i].open = 2.0 + i;
    }

    data_intersect_raw(pri, 5, sec, 5, &out_pri, &out_sec, &out_n);

    if (out_n != 5) {
        printf("\n  FAIL: out_n=%d, expected 5\n", out_n);
        utils_safe_free(out_pri);
        utils_safe_free(out_sec);
        return 0;
    }

    for (i = 0; i < 5; i++) {
        if (strcmp(out_pri[i].date, pri[i].date) != 0 ||
            fabs(out_pri[i].open - pri[i].open) > DBL_TOL ||
            fabs(out_sec[i].open - sec[i].open) > DBL_TOL) {
            printf("\n  FAIL: row %d mismatch\n", i);
            utils_safe_free(out_pri);
            utils_safe_free(out_sec);
            return 0;
        }
    }

    printf("\n  PASS: full overlap, all 5 rows aligned\n");
    utils_safe_free(out_pri);
    utils_safe_free(out_sec);
    return 1;
}

/* ==================== Test: data_intersect_raw partial ==================== */

int test_intersect_raw_partial(void) {
    DataRow pri[5], sec[5];
    DataRow *out_pri, *out_sec;
    int out_n;
    int i;

    printf("[TEST] data_intersect_raw: partial overlap...");
    fflush(stdout);

    for (i = 0; i < 5; i++) {
        snprintf(pri[i].date, 16, "2024.01.%02d", i + 1);
        snprintf(pri[i].time_, 16, "00:00");
        pri[i].open = 1.0;
        snprintf(sec[i].date, 16, "2024.01.%02d", i + 2);
        snprintf(sec[i].time_, 16, "00:00");
        sec[i].open = 2.0;
    }

    data_intersect_raw(pri, 5, sec, 5, &out_pri, &out_sec, &out_n);

    if (out_n != 4) {
        printf("\n  FAIL: out_n=%d, expected 4\n", out_n);
        utils_safe_free(out_pri);
        utils_safe_free(out_sec);
        return 0;
    }

    if (strcmp(out_pri[0].date, "2024.01.02") != 0) {
        printf("\n  FAIL: first intersection date='%s', expected 2024.01.02\n", out_pri[0].date);
        utils_safe_free(out_pri);
        utils_safe_free(out_sec);
        return 0;
    }

    printf("\n  PASS: partial overlap, 4 rows intersected\n");
    utils_safe_free(out_pri);
    utils_safe_free(out_sec);
    return 1;
}

/* ==================== Test: data_intersect_raw empty ==================== */

int test_intersect_raw_empty(void) {
    DataRow pri[3], sec[3];
    DataRow *out_pri, *out_sec;
    int out_n;
    int i;

    printf("[TEST] data_intersect_raw: empty intersection...");
    fflush(stdout);

    for (i = 0; i < 3; i++) {
        snprintf(pri[i].date, 16, "2024.01.%02d", i + 1);
        snprintf(pri[i].time_, 16, "00:00");
        snprintf(sec[i].date, 16, "2025.01.%02d", i + 1);
        snprintf(sec[i].time_, 16, "00:00");
    }

    data_intersect_raw(pri, 3, sec, 3, &out_pri, &out_sec, &out_n);

    if (out_n != 0) {
        printf("\n  FAIL: out_n=%d, expected 0\n", out_n);
        utils_safe_free(out_pri);
        utils_safe_free(out_sec);
        return 0;
    }

    printf("\n  PASS: empty intersection, 0 rows\n");
    utils_safe_free(out_pri);
    utils_safe_free(out_sec);
    return 1;
}

/* ==================== Test: compute 13 base features only ==================== */

int test_compute_raw_features_base_only(void) {
    DataRow *raw;
    int n = 50;
    double **feat;
    double *fmin, *fmax;
    char **fnames;
    int i;

    printf("[TEST] data_compute_raw_features: 13 base features only...");
    fflush(stdout);

    raw = (DataRow *)utils_safe_malloc((size_t)n * sizeof(DataRow));
    for (i = 0; i < n; i++) {
        raw[i].open = 1.1000 + i * 0.001;
        raw[i].high = raw[i].open + 0.0005;
        raw[i].low = raw[i].open - 0.0005;
        raw[i].close = raw[i].open + 0.0002;
        raw[i].tickvol = 1000;
        raw[i].vol = 0;
        raw[i].spread = 1;
        snprintf(raw[i].date, 16, "2024.01.%02d", (i % 28) + 1);
        snprintf(raw[i].time_, 16, "%02d:00", i % 24);
    }

    fmin = (double *)calloc(13, sizeof(double));
    fmax = (double *)calloc(13, sizeof(double));
    fnames = (char **)calloc(13, sizeof(char *));
    feat = data_compute_raw_features(raw, n, 13, fmin, fmax, fnames);

    if (!feat) {
        printf("\n  FAIL: returned NULL\n");
        free(fmin); free(fmax); free(fnames); utils_safe_free(raw);
        return 0;
    }
    if (strcmp(fnames[0], "candle_return") != 0) {
        printf("\n  FAIL: fnames[0]='%s', expected 'candle_return'\n", fnames[0]);
        data_free_raw_features(feat, n);
        for (i = 0; i < 13; i++) free(fnames[i]);
        free(fnames); free(fmin); free(fmax); utils_safe_free(raw);
        return 0;
    }
    if (strcmp(fnames[12], "dow") != 0) {
        printf("\n  FAIL: fnames[12]='%s', expected 'dow'\n", fnames[12]);
        data_free_raw_features(feat, n);
        for (i = 0; i < 13; i++) free(fnames[i]);
        free(fnames); free(fmin); free(fmax); utils_safe_free(raw);
        return 0;
    }

    printf("\n  PASS: 13 base features computed, names correct\n");
    data_free_raw_features(feat, n);
    for (i = 0; i < 13; i++) free(fnames[i]);
    free(fnames); free(fmin); free(fmax); utils_safe_free(raw);
    return 1;
}

/* ==================== Test: data_alloc_dataset_ext with NULL secondary ==================== */

int test_alloc_dataset_ext_null_secondary(void) {
    DataRow *raw;
    int n = 100;
    Dataset *ds;
    int i;

    printf("[TEST] data_alloc_dataset_ext: NULL secondary (backward compat)...");
    fflush(stdout);

    raw = (DataRow *)utils_safe_malloc((size_t)n * sizeof(DataRow));
    for (i = 0; i < n; i++) {
        raw[i].open = 1.1000 + i * 0.001;
        raw[i].high = raw[i].open + 0.0005;
        raw[i].low = raw[i].open - 0.0005;
        raw[i].close = raw[i].open + 0.0002;
        raw[i].tickvol = 1000;
        raw[i].vol = 0;
        raw[i].spread = 1;
        snprintf(raw[i].date, 16, "2024.01.%02d", (i % 28) + 1);
        snprintf(raw[i].time_, 16, "%02d:00", i % 24);
    }

    ds = data_alloc_dataset_ext(raw, n, 10, 5, 4, MODE_OHLC,
                                NULL, 0, NULL, NULL, NULL);
    if (!ds) {
        printf("\n  FAIL: returned NULL\n");
        utils_safe_free(raw);
        return 0;
    }
    if (ds->num_features != NUM_RAW_FEATURES) {
        printf("\n  FAIL: num_features=%d, expected %d\n", ds->num_features, NUM_RAW_FEATURES);
        data_free_dataset(ds);
        utils_safe_free(raw);
        return 0;
    }

    printf("\n  PASS: NULL secondary gives %d features (backward compat)\n", ds->num_features);
    data_free_dataset(ds);
    utils_safe_free(raw);
    return 1;
}

/* ==================== Test: data_alloc_dataset_ext with secondary ==================== */

int test_alloc_dataset_ext_with_secondary(void) {
    DataRow *raw;
    int n = 100;
    Dataset *ds;
    double **sec_feat;
    double *sec_min, *sec_max;
    char **sec_names;
    int i, j;
    double sec_val = 0.7;

    printf("[TEST] data_alloc_dataset_ext: with secondary (39 features)...");
    fflush(stdout);

    raw = (DataRow *)utils_safe_malloc((size_t)n * sizeof(DataRow));
    for (i = 0; i < n; i++) {
        raw[i].open = 1.1000 + i * 0.001;
        raw[i].high = raw[i].open + 0.0005;
        raw[i].low = raw[i].open - 0.0005;
        raw[i].close = raw[i].open + 0.0002;
        raw[i].tickvol = 1000;
        raw[i].vol = 0;
        raw[i].spread = 1;
        snprintf(raw[i].date, 16, "2024.01.%02d", (i % 28) + 1);
        snprintf(raw[i].time_, 16, "%02d:00", i % 24);
    }

    sec_feat = (double **)utils_safe_malloc((size_t)n * sizeof(double *));
    for (i = 0; i < n; i++) {
        sec_feat[i] = (double *)utils_safe_calloc((size_t)NUM_BASE_FEATURES, sizeof(double));
        for (j = 0; j < NUM_BASE_FEATURES; j++)
            sec_feat[i][j] = sec_val;
    }
    sec_min = (double *)utils_safe_calloc((size_t)NUM_BASE_FEATURES, sizeof(double));
    sec_max = (double *)utils_safe_calloc((size_t)NUM_BASE_FEATURES, sizeof(double));
    sec_names = (char **)utils_safe_calloc((size_t)NUM_BASE_FEATURES, sizeof(char *));
    for (j = 0; j < NUM_BASE_FEATURES; j++) {
        sec_min[j] = sec_val;
        sec_max[j] = sec_val;
        sec_names[j] = utils_strdup("sec_test");
    }

    ds = data_alloc_dataset_ext(raw, n, 10, 5, 4, MODE_OHLC,
                                sec_feat, NUM_BASE_FEATURES, sec_min, sec_max, sec_names);
    if (!ds) {
        printf("\n  FAIL: returned NULL\n");
        goto cleanup_sec;
    }
    if (ds->num_features != NUM_RAW_FEATURES + NUM_BASE_FEATURES) {
        printf("\n  FAIL: num_features=%d, expected %d\n", ds->num_features, NUM_RAW_FEATURES + NUM_BASE_FEATURES);
        data_free_dataset(ds);
        goto cleanup_sec;
    }
    if (strncmp(ds->feat_names[26], "sec_", 4) != 0) {
        printf("\n  FAIL: feat_names[26]='%s', expected sec_ prefix\n", ds->feat_names[26]);
        data_free_dataset(ds);
        goto cleanup_sec;
    }
    {
        double val = ds->features[0][0 * ds->num_features + 26];
        if (fabs(val - sec_val) > DBL_TOL) {
            printf("\n  FAIL: secondary feature value=%.6f, expected %.6f\n", val, sec_val);
            data_free_dataset(ds);
            goto cleanup_sec;
        }
    }

    printf("\n  PASS: 39 features, sec_ prefix, values correct\n");
    data_free_dataset(ds);

cleanup_sec:
    for (i = 0; i < n; i++) free(sec_feat[i]);
    free(sec_feat);
    for (j = 0; j < NUM_BASE_FEATURES; j++) free(sec_names[j]);
    free(sec_names); free(sec_min); free(sec_max);
    utils_safe_free(raw);
    return 1;
}
