#include "../test_utils.h"
#include "../../include/direction/data_direction.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ==================== Test: Direction labels are 0/1 ==================== */

int test_direction_labels(void) {
    DataRow *raw_data;
    int num_raw;
    Dataset *ds;
    int i, s, k;
    int bad = 0;

    printf("[TEST] Direction labels are 0/1...");
    fflush(stdout);

    num_raw = 300;
    raw_data = (DataRow *)utils_safe_malloc((size_t)num_raw * sizeof(DataRow));
    for (i = 0; i < num_raw; i++) {
        double cret = ((i % 6 < 3) ? 1.0 : -1.0) * 0.003;
        raw_data[i].open = 1.10000;
        raw_data[i].close = 1.10000 * (1.0 + cret);
        raw_data[i].high = (raw_data[i].close > raw_data[i].open) ?
            raw_data[i].close + 0.0005 : raw_data[i].open + 0.0005;
        raw_data[i].low = (raw_data[i].close < raw_data[i].open) ?
            raw_data[i].close - 0.0005 : raw_data[i].open - 0.0005;
        raw_data[i].tickvol = 1000 + (int)(utils_rand() * 500);
        raw_data[i].vol = 0;
        raw_data[i].spread = 1;
        snprintf(raw_data[i].date, 16, "2024.01.%02d", (i % 28) + 1);
        snprintf(raw_data[i].time_, 16, "%02d:00", i % 24);
    }

    ds = data_build_dataset_direction(raw_data, num_raw, 5, 5);
    if (!ds) {
        printf("  FAIL: could not build dataset\n");
        utils_safe_free(raw_data);
        return 0;
    }

    for (i = 0; i < ds->num_samples && !bad; i++) {
        for (s = 0; s < ds->sequence_length && !bad; s++) {
            for (k = 0; k < ds->num_outputs && !bad; k++) {
                double val = ds->labels[i][s * ds->num_outputs + k];
                if (val != 0.0 && val != 1.0) {
                    printf("\n  FAIL: label[%d][%d][%d]=%.6f, expected 0 or 1\n", i, s, k, val);
                    bad = 1;
                }
            }
        }
    }

    data_free_dataset(ds);
    utils_safe_free(raw_data);

    if (!bad) {
        printf("  PASS: all direction labels are 0 or 1\n");
        return 1;
    }
    return 0;
}
