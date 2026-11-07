#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DBL_TOL 1e-6

/* ==================== Helper: verify time alignment ==================== */

static int verify_time_alignment(DataRow *agg, int n_agg, int h) {
    int i;
    for (i = 0; i < n_agg; i++) {
        int hour = atoi(agg[i].time_);
        if (hour % h != 0) {
            printf("\n  FAIL: agg[%d] hour=%d not aligned to H%d\n", i, hour, h);
            return 0;
        }
    }
    return 1;
}

/* ==================== Test: H4 aggregation aligned to 00/04/08/12/16/20 ==================== */

int test_aggregate_h4_aligned(void) {
    DataRow *raw = NULL;
    int n_raw = 0;
    DataRow *agg = NULL;
    int n_agg = 0;

    printf("[TEST] H4 aggregation: times aligned to 00/04/08/12/16/20...");
    fflush(stdout);

    if (data_load_raw("../resources/EURUSD_H1_201001040000_202611022100.csv",
                      &raw, &n_raw) != 0) {
        printf("\n  FAIL: could not load CSV\n");
        return 0;
    }

    data_aggregate_raw(raw, n_raw, 4, &agg, &n_agg);

    if (!verify_time_alignment(agg, n_agg, 4)) {
        utils_safe_free(raw); utils_safe_free(agg);
        return 0;
    }

    printf("\n  PASS: all %d H4 bars aligned to 4-hour boundaries\n", n_agg);
    utils_safe_free(raw); utils_safe_free(agg);
    return 1;
}

/* ==================== Test: H2 aggregation aligned ==================== */

int test_aggregate_h2_aligned(void) {
    DataRow *raw = NULL;
    int n_raw = 0;
    DataRow *agg = NULL;
    int n_agg = 0;

    printf("[TEST] H2 aggregation: times aligned to even hours...");
    fflush(stdout);

    if (data_load_raw("../resources/EURUSD_H1_201001040000_202611022100.csv",
                      &raw, &n_raw) != 0) {
        printf("\n  FAIL: could not load CSV\n");
        return 0;
    }

    data_aggregate_raw(raw, n_raw, 2, &agg, &n_agg);

    if (!verify_time_alignment(agg, n_agg, 2)) {
        utils_safe_free(raw); utils_safe_free(agg);
        return 0;
    }

    printf("\n  PASS: all %d H2 bars aligned to 2-hour boundaries\n", n_agg);
    utils_safe_free(raw); utils_safe_free(agg);
    return 1;
}

/* ==================== Test: H8 aggregation aligned ==================== */

int test_aggregate_h8_aligned(void) {
    DataRow *raw = NULL;
    int n_raw = 0;
    DataRow *agg = NULL;
    int n_agg = 0;

    printf("[TEST] H8 aggregation: times aligned to 00/08/16...");
    fflush(stdout);

    if (data_load_raw("../resources/EURUSD_H1_201001040000_202611022100.csv",
                      &raw, &n_raw) != 0) {
        printf("\n  FAIL: could not load CSV\n");
        return 0;
    }

    data_aggregate_raw(raw, n_raw, 8, &agg, &n_agg);

    if (!verify_time_alignment(agg, n_agg, 8)) {
        utils_safe_free(raw); utils_safe_free(agg);
        return 0;
    }

    printf("\n  PASS: all %d H8 bars aligned to 8-hour boundaries\n", n_agg);
    utils_safe_free(raw); utils_safe_free(agg);
    return 1;
}

/* ==================== Test: H4 after synthetic gap stays aligned ==================== */

int test_aggregate_h4_after_gap(void) {
    /* Create synthetic data: 8 bars at 00-03, gap, 4 bars at 08-11 */
    DataRow raw[12];
    DataRow *agg = NULL;
    int n_agg = 0;
    int i;

    printf("[TEST] H4 aggregation: stays aligned after gap...");
    fflush(stdout);

    /* 4 bars: 00:00, 01:00, 02:00, 03:00 */
    for (i = 0; i < 4; i++) {
        strcpy(raw[i].date, "2024.01.15");
        snprintf(raw[i].time_, 16, "%02d:00", i);
        raw[i].open = 1.10 + i * 0.001;
        raw[i].high = raw[i].open + 0.0005;
        raw[i].low = raw[i].open - 0.0005;
        raw[i].close = raw[i].open + 0.0002;
        raw[i].tickvol = 1000;
        raw[i].vol = 0;
        raw[i].spread = 1;
    }
    /* Gap: skip 04:00, 05:00, 06:00, 07:00 (weekend) */
    /* 4 bars: 08:00, 09:00, 10:00, 11:00 */
    for (i = 4; i < 8; i++) {
        strcpy(raw[i].date, "2024.01.15");
        snprintf(raw[i].time_, 16, "%02d:00", i + 4);
        raw[i].open = 1.10 + i * 0.001;
        raw[i].high = raw[i].open + 0.0005;
        raw[i].low = raw[i].open - 0.0005;
        raw[i].close = raw[i].open + 0.0002;
        raw[i].tickvol = 1000;
        raw[i].vol = 0;
        raw[i].spread = 1;
    }
    /* More bars: 12:00, 13:00, 14:00, 15:00 */
    for (i = 8; i < 12; i++) {
        strcpy(raw[i].date, "2024.01.15");
        snprintf(raw[i].time_, 16, "%02d:00", i + 4);
        raw[i].open = 1.10 + i * 0.001;
        raw[i].high = raw[i].open + 0.0005;
        raw[i].low = raw[i].open - 0.0005;
        raw[i].close = raw[i].open + 0.0002;
        raw[i].tickvol = 1000;
        raw[i].vol = 0;
        raw[i].spread = 1;
    }

    data_aggregate_raw(raw, 12, 4, &agg, &n_agg);

    /* Expect: H4 bars at 00:00, 08:00, 12:00 */
    if (n_agg != 3) {
        printf("\n  FAIL: n_agg=%d, expected 3\n", n_agg);
        utils_safe_free(agg);
        return 0;
    }

    if (atoi(agg[0].time_) != 0) {
        printf("\n  FAIL: agg[0] hour=%d, expected 0\n", atoi(agg[0].time_));
        utils_safe_free(agg);
        return 0;
    }
    if (atoi(agg[1].time_) != 8) {
        printf("\n  FAIL: agg[1] hour=%d, expected 8\n", atoi(agg[1].time_));
        utils_safe_free(agg);
        return 0;
    }
    if (atoi(agg[2].time_) != 12) {
        printf("\n  FAIL: agg[2] hour=%d, expected 12\n", atoi(agg[2].time_));
        utils_safe_free(agg);
        return 0;
    }

    printf("\n  PASS: H4 bars at 00:00, 08:00, 12:00 (aligned after gap)\n");
    utils_safe_free(agg);
    return 1;
}

/* ==================== Test: partial group kept after gap ==================== */

int test_aggregate_partial_group_kept(void) {
    /* 5 bars: 00:00, 01:00, 02:00, 03:00, 05:00 (skip 04:00) */
    DataRow raw[5];
    DataRow *agg = NULL;
    int n_agg = 0;
    int i;

    printf("[TEST] H4 aggregation: partial group kept (2 bars at 04-05)...");
    fflush(stdout);

    for (i = 0; i < 4; i++) {
        strcpy(raw[i].date, "2024.01.15");
        snprintf(raw[i].time_, 16, "%02d:00", i);
        raw[i].open = 1.10 + i * 0.001;
        raw[i].high = raw[i].open + 0.0005;
        raw[i].low = raw[i].open - 0.0005;
        raw[i].close = raw[i].open + 0.0002;
        raw[i].tickvol = 1000;
        raw[i].vol = 0;
        raw[i].spread = 1;
    }
    /* Bar at 05:00 (04:00 is missing — partial group at 04:00 boundary) */
    strcpy(raw[4].date, "2024.01.15");
    strcpy(raw[4].time_, "05:00");
    raw[4].open = 1.1050;
    raw[4].high = 1.1055;
    raw[4].low = 1.1045;
    raw[4].close = 1.1052;
    raw[4].tickvol = 1000;
    raw[4].vol = 0;
    raw[4].spread = 1;

    data_aggregate_raw(raw, 5, 4, &agg, &n_agg);

    /* Expect: H4 bar at 00:00 (4 bars) + H4 bar at 04:00 (1 bar, partial) */
    if (n_agg != 2) {
        printf("\n  FAIL: n_agg=%d, expected 2\n", n_agg);
        utils_safe_free(agg);
        return 0;
    }
    if (atoi(agg[0].time_) != 0 || atoi(agg[1].time_) != 4) {
        printf("\n  FAIL: times %s, %s expected 00:00, 04:00\n", agg[0].time_, agg[1].time_);
        utils_safe_free(agg);
        return 0;
    }

    printf("\n  PASS: partial group at 04:00 kept (1 bar)\n");
    utils_safe_free(agg);
    return 1;
}
