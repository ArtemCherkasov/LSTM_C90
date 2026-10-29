#include "../test_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ==================== Test: Volume Return Clipping & Label Normalization === */

int test_volume_clip_and_label_norm_ohlc(void) {
    DataRow *raw_data;
    int num_raw;
    Dataset *ds;
    int i, o, s;
    double max_abs_vol_ret;
    double max_abs_label;

    printf("[TEST] OHLC volume clip & label normalization...");
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

    /* Inject extreme volume outlier at row 50 */
    raw_data[49].tickvol = 1;
    raw_data[50].tickvol = 50000;

    ds = data_build_dataset_ohlc(raw_data, num_raw, 5, 5);
    if (!ds) {
        printf("  FAIL: could not build dataset\n");
        utils_safe_free(raw_data);
        return 0;
    }

    max_abs_vol_ret = fabs(ds->feat_max[3]);
    if (fabs(ds->feat_min[3]) > max_abs_vol_ret)
        max_abs_vol_ret = fabs(ds->feat_min[3]);

    if (max_abs_vol_ret > 10.0 + 1e-6) {
        printf("\n  FAIL: volume_return max_abs=%.4f exceeds clip=10.0\n",
               max_abs_vol_ret);
        data_free_dataset(ds);
        utils_safe_free(raw_data);
        return 0;
    }

    data_normalize_features(ds);
    data_normalize_labels_ohlc(ds);;

    max_abs_label = 0.0;
    for (i = 0; i < ds->num_samples; i++) {
        for (s = 0; s < ds->sequence_length; s++) {
            for (o = 0; o < ds->num_outputs; o++) {
                double a = fabs(ds->labels[i][s * ds->num_outputs + o]);
                if (a > max_abs_label) max_abs_label = a;
            }
        }
    }

    if (max_abs_label > 1.0 + 1e-6) {
        printf("\n  FAIL: max |label|=%.6f exceeds 1.0 after normalization\n",
               max_abs_label);
        data_free_dataset(ds);
        utils_safe_free(raw_data);
        return 0;
    }

    printf("  PASS: vol_ret clipped to %.4f, labels in [-1,1] (max=%.4f)\n",
           max_abs_vol_ret, max_abs_label);
    data_free_dataset(ds);
    utils_safe_free(raw_data);
    return 1;
}
