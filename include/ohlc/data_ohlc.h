#ifndef DATA_OHLC_H
#define DATA_OHLC_H

#include "data.h"

/* Build dataset with OHLC return labels (many-to-many) */
Dataset *data_build_dataset_ohlc(DataRow *raw, int num_raw, int seq_len, int prediction_horizon);

Dataset *data_build_dataset_ohlc_ext(DataRow *raw, int num_raw, int seq_len, int prediction_horizon,
                                     double **sec_features, int num_sec_features,
                                     double *sec_feat_min, double *sec_feat_max,
                                     char **sec_feat_names);

/* Normalize OHLC labels: per-output symmetric scaling */
void data_normalize_labels_ohlc(Dataset *ds);

#endif /* DATA_OHLC_H */
