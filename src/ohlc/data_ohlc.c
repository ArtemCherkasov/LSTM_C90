#include "../../include/ohlc/data_ohlc.h"
#include <stdlib.h>
#include <math.h>

Dataset *data_build_dataset_ohlc(DataRow *raw, int num_raw, int seq_len, int prediction_horizon) {
    int num_outputs = prediction_horizon * 4;
    Dataset *ds;
    int i, s, k, j;
    int label_idx;

    ds = data_alloc_dataset(raw, num_raw, seq_len, prediction_horizon, num_outputs, MODE_OHLC);
    if (!ds) return NULL;

    /* Fill OHLC return labels for each step */
    for (i = 0; i < ds->num_samples; i++) {
        for (s = 0; s < seq_len; s++) {
            int si = i + s;
            double step_ref = raw[si].close;
            for (k = 0; k < prediction_horizon; k++) {
                label_idx = si + 1 + k;
                ds->labels[i][s * num_outputs + k * 4 + 0] = (raw[label_idx].open - step_ref) / step_ref;
                ds->labels[i][s * num_outputs + k * 4 + 1] = (raw[label_idx].high - step_ref) / step_ref;
                ds->labels[i][s * num_outputs + k * 4 + 2] = (raw[label_idx].low - step_ref) / step_ref;
                ds->labels[i][s * num_outputs + k * 4 + 3] = (raw[label_idx].close - step_ref) / step_ref;
            }
        }
        /* Track per-output min/max */
        for (s = 0; s < seq_len; s++) {
            for (j = 0; j < num_outputs; j++) {
                double val = ds->labels[i][s * num_outputs + j];
                if (val < ds->label_min[j]) ds->label_min[j] = val;
                if (val > ds->label_max[j]) ds->label_max[j] = val;
            }
        }
    }
    return ds;
}

void data_normalize_labels_ohlc(Dataset *ds) {
    int i, o, s;
    double label_max_abs;
    for (o = 0; o < ds->num_outputs; o++) {
        label_max_abs = fabs(ds->label_min[o]);
        if (fabs(ds->label_max[o]) > label_max_abs) label_max_abs = fabs(ds->label_max[o]);
        if (label_max_abs < 1e-15) label_max_abs = 1.0;
        for (i = 0; i < ds->num_samples; i++) {
            for (s = 0; s < ds->sequence_length; s++) {
                ds->labels[i][s * ds->num_outputs + o] /= label_max_abs;
            }
        }
    }
}
