#include "../../include/ohlc/predict_ohlc.h"
#include <stdio.h>
#include <math.h>

void predict_ohlc(double *pred_outputs, int num_outputs, int seq_len,
                  double *lbl_min_arr, double *lbl_max_arr,
                  double ref_close, int ohlc_predict_count) {
    int last_base = (seq_len - 1) * num_outputs;
    int kk, o;

    for (kk = 0; kk < ohlc_predict_count; kk++) {
        double pred_open, pred_high, pred_low, pred_close;
        double label_max_abs;

        for (o = 0; o < 4; o++) {
            int out_idx = kk * 4 + o;
            label_max_abs = fabs(lbl_min_arr[out_idx]);
            if (fabs(lbl_max_arr[out_idx]) > label_max_abs)
                label_max_abs = fabs(lbl_max_arr[out_idx]);
            if (label_max_abs < 1e-15) label_max_abs = 1.0;
            pred_outputs[last_base + out_idx] *= label_max_abs;
        }

        pred_open  = ref_close * (1.0 + pred_outputs[last_base + kk * 4 + 0]);
        pred_high  = ref_close * (1.0 + pred_outputs[last_base + kk * 4 + 1]);
        pred_low   = ref_close * (1.0 + pred_outputs[last_base + kk * 4 + 2]);
        pred_close = ref_close * (1.0 + pred_outputs[last_base + kk * 4 + 3]);

        printf("PRE [%03d] %.5f %.5f %.5f %.5f\n",
               kk + 1, pred_open, pred_high, pred_low, pred_close);
    }
}

void demo_ohlc(LSTMNetwork *network, Dataset *ds, int ohlc_predict_count) {
    int demo_start, d, kk;
    double *preds;
    int num_outputs = ds->num_outputs;
    int seq_len = ds->sequence_length;

    printf("Demo: Predicting on last 5 sequences ...\n");
    demo_start = ds->num_samples - 5;
    if (demo_start < 0) demo_start = 0;
    preds = (double *)utils_safe_calloc((size_t)(num_outputs * seq_len), sizeof(double));
    for (d = demo_start; d < ds->num_samples; d++) {
        lstm_predict(network, ds->features[d], seq_len, preds);
        printf("  Sample %d predictions (last step OHLC x %d candles):\n", d, ohlc_predict_count);
        for (kk = 0; kk < ohlc_predict_count; kk++) {
            int lb = (seq_len - 1) * num_outputs;
            printf("    candle %d: O=%.6f H=%.6f L=%.6f C=%.6f\n",
                   kk + 1,
                   preds[lb + kk * 4 + 0], preds[lb + kk * 4 + 1],
                   preds[lb + kk * 4 + 2], preds[lb + kk * 4 + 3]);
        }
    }
    utils_safe_free(preds);
}
