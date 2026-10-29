#ifndef DATA_OHLC_H
#define DATA_OHLC_H

#include "data.h"

/* Build dataset with OHLC return labels (many-to-many) */
Dataset *data_build_dataset_ohlc(DataRow *raw, int num_raw, int seq_len, int prediction_horizon);

/* Normalize OHLC labels: per-output symmetric scaling */
void data_normalize_labels_ohlc(Dataset *ds);

#endif /* DATA_OHLC_H */
