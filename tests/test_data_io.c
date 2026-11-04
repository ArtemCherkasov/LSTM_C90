#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Path to test CSV — relative to build directory */
#define TEST_CSV "../resources/EURUSD_H1_200906120000_202509251100.csv"

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
    int i, j;

    /* Check row count */
    if (n_agg != n_raw / h) {
        printf("\n  FAIL: n_agg=%d, expected %d (%d / %d)\n",
               n_agg, n_raw / h, n_raw, h);
        return 0;
    }

    for (i = 0; i < n_agg; i++) {
        int base = i * h;
        double max_high, min_low, sum_vol;

        /* Open = first candle's open */
        if (fabs(agg[i].open - raw[base].open) > DBL_TOL) {
            printf("\n  FAIL: agg[%d].open=%.5f != raw[%d].open=%.5f\n",
                   i, agg[i].open, base, raw[base].open);
            return 0;
        }

        /* High = max of highs */
        max_high = raw[base].high;
        for (j = 1; j < h; j++)
            if (raw[base + j].high > max_high) max_high = raw[base + j].high;
        if (fabs(agg[i].high - max_high) > DBL_TOL) {
            printf("\n  FAIL: agg[%d].high=%.5f != max_high=%.5f\n",
                   i, agg[i].high, max_high);
            return 0;
        }

        /* Low = min of lows */
        min_low = raw[base].low;
        for (j = 1; j < h; j++)
            if (raw[base + j].low < min_low) min_low = raw[base + j].low;
        if (fabs(agg[i].low - min_low) > DBL_TOL) {
            printf("\n  FAIL: agg[%d].low=%.5f != min_low=%.5f\n",
                   i, agg[i].low, min_low);
            return 0;
        }

        /* Close = last candle's close */
        if (fabs(agg[i].close - raw[base + h - 1].close) > DBL_TOL) {
            printf("\n  FAIL: agg[%d].close=%.5f != raw[%d].close=%.5f\n",
                   i, agg[i].close, base + h - 1, raw[base + h - 1].close);
            return 0;
        }

        /* TickVol = sum of tickvols */
        sum_vol = 0.0;
        for (j = 0; j < h; j++)
            sum_vol += (double)raw[base + j].tickvol;
        if (abs(agg[i].tickvol - (int)sum_vol) > 0) {
            printf("\n  FAIL: agg[%d].tickvol=%d != sum=%d\n",
                   i, agg[i].tickvol, (int)sum_vol);
            return 0;
        }

        /* Date/Time = first candle's date/time */
        if (strcmp(agg[i].date, raw[base].date) != 0) {
            printf("\n  FAIL: agg[%d].date='%s' != raw[%d].date='%s'\n",
                   i, agg[i].date, base, raw[base].date);
            return 0;
        }
        if (strcmp(agg[i].time_, raw[base].time_) != 0) {
            printf("\n  FAIL: agg[%d].time_='%s' != raw[%d].time_='%s'\n",
                   i, agg[i].time_, base, raw[base].time_);
            return 0;
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

/* ==================== Test: double aggregation h=2 then h=2 equals h=4 ==================== */

int test_aggregate_double_h2(void) {
    DataRow *raw = NULL;
    int n_raw = 0;
    DataRow *agg1 = NULL, *agg2 = NULL;
    int n1 = 0, n2 = 0;
    DataRow *agg4 = NULL;
    int n4 = 0;
    int i;
    int bad = 0;

    printf("[TEST] data_aggregate_raw: h=2 then h=2 equals h=4...");
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

    if (n2 != n4) {
        printf("\n  FAIL: double h=2 gives %d rows, h=4 gives %d rows\n", n2, n4);
        bad = 1;
    } else {
        for (i = 0; i < n2 && !bad; i++) {
            if (fabs(agg2[i].open  - agg4[i].open)  > DBL_TOL ||
                fabs(agg2[i].high  - agg4[i].high)  > DBL_TOL ||
                fabs(agg2[i].low   - agg4[i].low)   > DBL_TOL ||
                fabs(agg2[i].close - agg4[i].close) > DBL_TOL) {
                printf("\n  FAIL: row %d differs between double-h2 and h4\n", i);
                bad = 1;
            }
        }
    }

    utils_safe_free(raw);
    utils_safe_free(agg1);
    utils_safe_free(agg2);
    utils_safe_free(agg4);

    if (!bad) {
        printf("\n  PASS: h=2 then h=2 equals h=4 (%d rows)\n", n2);
        return 1;
    }
    return 0;
}
