#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================== Test: shift H1 datetime back by 1 hour ==================== */

int test_datetime_shift_h1(void) {
    char out[32];

    printf("[TEST] datetime shift: H1 (subtract 1 hour)...");
    fflush(stdout);

    /* 09:00 - 1h = 08:00 → candle at 08:00 closes at 09:00 */
    utils_shift_datetime_back("2024.01.15T09:00:00", 1, out, sizeof(out));
    if (strcmp(out, "2024.01.15T08:00:00") != 0) {
        printf("\n  FAIL: got '%s', expected '2024.01.15T08:00:00'\n", out);
        return 0;
    }

    /* 00:00 - 1h = 23:00 previous day */
    utils_shift_datetime_back("2024.01.15T00:00:00", 1, out, sizeof(out));
    if (strcmp(out, "2024.01.14T23:00:00") != 0) {
        printf("\n  FAIL: got '%s', expected '2024.01.14T23:00:00'\n", out);
        return 0;
    }

    printf("\n  PASS: H1 shift correct (09:00→08:00, 00:00→prev 23:00)\n");
    return 1;
}

/* ==================== Test: shift H4 datetime back by 4 hours ==================== */

int test_datetime_shift_h4(void) {
    char out[32];

    printf("[TEST] datetime shift: H4 (subtract 4 hours)...");
    fflush(stdout);

    /* 08:00 - 4h = 04:00 → H4 candle at 04:00 closes at 08:00 */
    utils_shift_datetime_back("2024.01.15T08:00:00", 4, out, sizeof(out));
    if (strcmp(out, "2024.01.15T04:00:00") != 0) {
        printf("\n  FAIL: got '%s', expected '2024.01.15T04:00:00'\n", out);
        return 0;
    }

    /* 00:00 - 4h = 20:00 previous day */
    utils_shift_datetime_back("2024.01.15T00:00:00", 4, out, sizeof(out));
    if (strcmp(out, "2024.01.14T20:00:00") != 0) {
        printf("\n  FAIL: got '%s', expected '2024.01.14T20:00:00'\n", out);
        return 0;
    }

    printf("\n  PASS: H4 shift correct (08:00→04:00, 00:00→prev 20:00)\n");
    return 1;
}

/* ==================== Test: shift across month boundary ==================== */

int test_datetime_shift_month_boundary(void) {
    char out[32];

    printf("[TEST] datetime shift: across month boundary...");
    fflush(stdout);

    /* 2024.02.01T02:00:00 - 4h = 2024.01.31T22:00:00 */
    utils_shift_datetime_back("2024.02.01T02:00:00", 4, out, sizeof(out));
    if (strcmp(out, "2024.01.31T22:00:00") != 0) {
        printf("\n  FAIL: got '%s', expected '2024.01.31T22:00:00'\n", out);
        return 0;
    }

    /* 2024.03.01T02:00:00 - 4h = 2024.02.29T22:00:00 (leap year) */
    utils_shift_datetime_back("2024.03.01T02:00:00", 4, out, sizeof(out));
    if (strcmp(out, "2024.02.29T22:00:00") != 0) {
        printf("\n  FAIL: got '%s', expected '2024.02.29T22:00:00'\n", out);
        return 0;
    }

    /* 2025.03.01T02:00:00 - 4h = 2025.02.28T22:00:00 (non-leap year) */
    utils_shift_datetime_back("2025.03.01T02:00:00", 4, out, sizeof(out));
    if (strcmp(out, "2025.02.28T22:00:00") != 0) {
        printf("\n  FAIL: got '%s', expected '2025.02.28T22:00:00'\n", out);
        return 0;
    }

    /* 2024.01.01T02:00:00 - 4h = 2023.12.31T22:00:00 (year boundary) */
    utils_shift_datetime_back("2024.01.01T02:00:00", 4, out, sizeof(out));
    if (strcmp(out, "2023.12.31T22:00:00") != 0) {
        printf("\n  FAIL: got '%s', expected '2023.12.31T22:00:00'\n", out);
        return 0;
    }

    printf("\n  PASS: month/year boundary shifts correct\n");
    return 1;
}

/* ==================== Test: shift with h=1 is identity minus 1 hour ==================== */

int test_datetime_shift_h1_identity(void) {
    char out[32];

    printf("[TEST] datetime shift: h=1 on real data (08:00 → 07:00)...");
    fflush(stdout);

    /* With h=1, datetime 08:00 should shift to 07:00 (candle at 07:00 closes at 08:00) */
    utils_shift_datetime_back("2024.01.15T08:00:00", 1, out, sizeof(out));
    if (strcmp(out, "2024.01.15T07:00:00") != 0) {
        printf("\n  FAIL: got '%s', expected '2024.01.15T07:00:00'\n", out);
        return 0;
    }

    printf("\n  PASS: h=1 shift gives previous hour\n");
    return 1;
}
