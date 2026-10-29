#include "../../include/direction/predict_direction.h"
#include <stdio.h>

void predict_direction(double *pred_outputs, int num_outputs, int seq_len,
                        int ohlc_predict_count) {
    int last_base = (seq_len - 1) * num_outputs;
    int k;

    for (k = 0; k < ohlc_predict_count; k++) {
        double raw = pred_outputs[last_base + k];
        int direction = (raw > 0.5) ? 1 : -1;
        printf("PRE [%03d] %d  (raw=%.6f)\n", k + 1, direction, raw);
    }
}

void demo_direction(LSTMNetwork *network, Dataset *ds, int ohlc_predict_count) {
    int demo_start, d, k;
    double *preds;
    int num_outputs = ds->num_outputs;
    int seq_len = ds->sequence_length;

    printf("Demo: Predicting on last 5 sequences ...\n");
    demo_start = ds->num_samples - 5;
    if (demo_start < 0) demo_start = 0;
    preds = (double *)utils_safe_calloc((size_t)(num_outputs * seq_len), sizeof(double));
    for (d = demo_start; d < ds->num_samples; d++) {
        lstm_predict(network, ds->features[d], seq_len, preds);
        printf("  Sample %d directions (last step, %d steps ahead):\n", d, ohlc_predict_count);
        for (k = 0; k < ohlc_predict_count; k++) {
            int lb = (seq_len - 1) * num_outputs;
            double raw = preds[lb + k];
            int direction = (raw > 0.5) ? 1 : -1;
            printf("    step %d: %d  (raw=%.6f)\n", k + 1, direction, raw);
        }
    }
    utils_safe_free(preds);
}
