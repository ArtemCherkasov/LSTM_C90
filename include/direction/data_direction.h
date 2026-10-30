#ifndef DATA_DIRECTION_H
#define DATA_DIRECTION_H

#include "data.h"

/* Build dataset with direction labels: 1 if close_{t+k} > close_t, 0 otherwise */
Dataset *data_build_dataset_direction(DataRow *raw, int num_raw, int seq_len, int prediction_horizon);

/* Normalize direction labels (no-op, already in [0, 1]) */
void data_normalize_labels_direction(Dataset *ds);

#endif /* DATA_DIRECTION_H */
