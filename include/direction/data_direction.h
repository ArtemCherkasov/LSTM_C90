#ifndef DATA_DIRECTION_H
#define DATA_DIRECTION_H

#include "data.h"

/* Build dataset with direction labels: 1 if close_{t+k} > close_t, 0 otherwise */
Dataset *data_build_dataset_direction(DataRow *raw, int num_raw, int seq_len, int prediction_horizon);

Dataset *data_build_dataset_direction_ext(DataRow *raw, int num_raw, int seq_len, int prediction_horizon,
                                           double **sec_features, int num_sec_features,
                                           double *sec_feat_min, double *sec_feat_max,
                                           char **sec_feat_names);

/* Normalize direction labels (no-op, already in [0, 1]) */
void data_normalize_labels_direction(Dataset *ds);

#endif /* DATA_DIRECTION_H */
