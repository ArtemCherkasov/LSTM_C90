#include "../../include/direction/data_direction.h"
#include <stdlib.h>

Dataset *data_build_dataset_direction_ext(DataRow *raw, int num_raw, int seq_len, int prediction_horizon,
                                           double **sec_features, int num_sec_features,
                                           double *sec_feat_min, double *sec_feat_max,
                                           char **sec_feat_names) {
    int num_outputs = prediction_horizon;
    Dataset *ds;
    int i, s, k;

    ds = data_alloc_dataset_ext(raw, num_raw, seq_len, prediction_horizon, num_outputs, MODE_DIRECTION,
                                sec_features, num_sec_features, sec_feat_min, sec_feat_max, sec_feat_names);
    if (!ds) return NULL;

    for (i = 0; i < ds->num_samples; i++) {
        for (s = 0; s < seq_len; s++) {
            int si = i + s;
            double step_ref = raw[si].close;
            for (k = 0; k < prediction_horizon; k++) {
                int label_idx = si + 1 + k;
                ds->labels[i][s * num_outputs + k] =
                    (raw[label_idx].close > step_ref) ? 1.0 : 0.0;
            }
        }
    }

    for (k = 0; k < num_outputs; k++) {
        ds->label_min[k] = 0.0;
        ds->label_max[k] = 1.0;
    }

    return ds;
}

Dataset *data_build_dataset_direction(DataRow *raw, int num_raw, int seq_len, int prediction_horizon) {
    return data_build_dataset_direction_ext(raw, num_raw, seq_len, prediction_horizon,
                                            NULL, 0, NULL, NULL, NULL);
}

void data_normalize_labels_direction(Dataset *ds) {
    /* No-op: labels are already 0/1, no normalization needed */
    (void)ds;
}
