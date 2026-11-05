#include "../../include/ohlc/data_ohlc.h"
#include <stdlib.h>
#include <math.h>

Dataset *data_build_dataset_ohlc_ext(DataRow *raw, int num_raw, int seq_len, int prediction_horizon,
                                     double **sec_features, int num_sec_features,
                                     double *sec_feat_min, double *sec_feat_max,
                                     char **sec_feat_names) {
    int num_outputs = prediction_horizon * 4;
    Dataset *ds;
    int i, s, k, j;
    int label_idx;

    ds = data_alloc_dataset_ext(raw, num_raw, seq_len, prediction_horizon, num_outputs, MODE_OHLC,
                                sec_features, num_sec_features, sec_feat_min, sec_feat_max, sec_feat_names);
    if (!ds) return NULL;

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

Dataset *data_build_dataset_ohlc(DataRow *raw, int num_raw, int seq_len, int prediction_horizon) {
    return data_build_dataset_ohlc_ext(raw, num_raw, seq_len, prediction_horizon,
                                       NULL, 0, NULL, NULL, NULL);
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
